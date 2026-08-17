"""Tile a single-cycle mjlab motion NPZ into a ~20 s style clip.

Preserves the mjlab archive schema (``fps``, ``joint_names``, ``body_names``,
``joint_pos`` / ``joint_vel``, ``body_*_w``) and adds ``cycle_len`` plus
``cmd_vx`` for the style task.

Pipeline
--------
1. Heading-align so mean base yaw faces +X.
2. Detect one gait cycle from joint-position autocorrelation.
3. If the first and last cycle frames are nearly identical, drop the last
   frame before tiling (avoids a repeated pose at the seam).
4. Repeat the cycle to ``num_out`` frames (default 1000 @ 50 Hz = 20.0 s).
   Root xy accumulates ``Δ = p_xy[-1] - p_xy[0]`` each lap; z / quat / joints
   / velocities cycle. Seam velocity jumps are left as-is (same as APEX).

Usage::

    uv run python scripts/tools/tile_motion_npz.py \\
        --npz /tmp/motion.npz \\
        --out /tmp/xingzou_style.npz \\
        --num-out 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_FPS = 50.0
DEFAULT_NUM_OUT = 1000
SEAM_DROP_THRESHOLD = 1.0e-3
FOOT_BODY_NAMES: tuple[str, ...] = (
  "fl_foot_link",
  "fr_foot_link",
  "rl_foot_link",
  "rr_foot_link",
)


def _as_str_list(values: np.ndarray) -> list[str]:
  return [
    v.decode() if isinstance(v, bytes) else str(v)
    for v in np.asarray(values).reshape(-1)
  ]


def quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> np.ndarray:
  w, x, y, z = np.moveaxis(np.asarray(quat_wxyz, dtype=np.float64), -1, 0)
  return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
  """Hamilton product, both (..., 4) in wxyz."""
  w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
  w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
  return np.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    axis=-1,
  )


def yaw_quat_wxyz(yaw: np.ndarray | float) -> np.ndarray:
  half = 0.5 * np.asarray(yaw, dtype=np.float64)
  zeros = np.zeros_like(half)
  return np.stack([np.cos(half), zeros, zeros, np.sin(half)], axis=-1)


def rotate_xy(vec_xy: np.ndarray, yaw: float) -> np.ndarray:
  c, s = np.cos(yaw), np.sin(yaw)
  x, y = vec_xy[..., 0], vec_xy[..., 1]
  out = np.empty_like(vec_xy, dtype=np.float64)
  out[..., 0] = c * x - s * y
  out[..., 1] = s * x + c * y
  return out


def rotate_vec(vec: np.ndarray, yaw: float) -> np.ndarray:
  out = np.array(vec, dtype=np.float64, copy=True)
  out[..., :2] = rotate_xy(out[..., :2], yaw)
  return out


def quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
  w, x, y, z = np.moveaxis(np.asarray(quat_wxyz, dtype=np.float64), -1, 0)
  rot = np.empty(quat_wxyz.shape[:-1] + (3, 3), dtype=np.float64)
  rot[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
  rot[..., 0, 1] = 2.0 * (x * y - z * w)
  rot[..., 0, 2] = 2.0 * (x * z + y * w)
  rot[..., 1, 0] = 2.0 * (x * y + z * w)
  rot[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
  rot[..., 1, 2] = 2.0 * (y * z - x * w)
  rot[..., 2, 0] = 2.0 * (x * z - y * w)
  rot[..., 2, 1] = 2.0 * (y * z + x * w)
  rot[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
  return rot


def world_to_body_vel(quat_wxyz: np.ndarray, vel_w: np.ndarray) -> np.ndarray:
  rot = quat_wxyz_to_rotmat(quat_wxyz)
  return np.einsum("...ji,...j->...i", rot, vel_w)


def feet_in_yaw_frame(
  base_pos: np.ndarray, base_yaw: np.ndarray, foot_pos_w: np.ndarray
) -> np.ndarray:
  """Feet in the yaw-heading frame used by the style end-effector reward."""
  rel = foot_pos_w - base_pos[:, None, :]
  c = np.cos(base_yaw)[:, None]
  s = np.sin(base_yaw)[:, None]
  x, y, z = rel[..., 0], rel[..., 1], rel[..., 2]
  return np.stack([c * x + s * y, -s * x + c * y, z], axis=-1)


def detect_cycle_length(
  joint_pos: np.ndarray, min_lag: int = 12, max_lag: int = 40
) -> int:
  """Return the lag that best loops the clip; 0 means use the whole clip."""
  n = joint_pos.shape[0]
  max_lag = min(max_lag, n // 2)
  if n < min_lag + 2:
    return 0
  best_lag = 0
  best_score = np.inf
  for lag in range(min_lag, max_lag + 1):
    score = float(np.linalg.norm(joint_pos[lag:] - joint_pos[:-lag], axis=1).mean())
    seam = float(np.linalg.norm(joint_pos[0] - joint_pos[lag - 1]))
    combined = score + 0.25 * seam
    if combined < best_score:
      best_score = combined
      best_lag = lag
  rms = float(np.sqrt(np.mean(joint_pos * joint_pos))) + 1e-6
  if best_score > 0.35 * rms * np.sqrt(joint_pos.shape[1]):
    return 0
  return best_lag


def resolve_cycle_length(joint_pos: np.ndarray, cycle_len: int) -> tuple[int, float]:
  """Drop a duplicated last frame when the cycle seam is nearly closed."""
  n = joint_pos.shape[0]
  length = n if cycle_len <= 0 else min(int(cycle_len), n)
  seam_q = float(np.linalg.norm(joint_pos[0] - joint_pos[length - 1]))
  if length > 2 and seam_q < SEAM_DROP_THRESHOLD:
    length -= 1
    seam_q = float(np.linalg.norm(joint_pos[0] - joint_pos[length - 1]))
  return length, seam_q


def tile_motion(
  joint_pos: np.ndarray,
  joint_vel: np.ndarray,
  body_pos_w: np.ndarray,
  body_quat_w: np.ndarray,
  body_lin_vel_w: np.ndarray,
  body_ang_vel_w: np.ndarray,
  *,
  num_out: int,
  cycle_len: int,
  base_idx: int,
) -> dict[str, np.ndarray | float | int]:
  """Repeat one cycle to ``num_out`` frames, accumulating root xy each lap."""
  length, seam_q = resolve_cycle_length(joint_pos, cycle_len)
  src = slice(0, length)
  delta_xy = body_pos_w[length - 1, base_idx, :2] - body_pos_w[0, base_idx, :2]
  idx = np.arange(num_out)
  phase = idx % length
  laps = idx // length
  body_pos = np.array(body_pos_w[src][phase], dtype=np.float64, copy=True)
  body_pos[..., :2] = body_pos[..., :2] + laps[:, None, None] * delta_xy[None, None, :]
  return {
    "joint_pos": joint_pos[src][phase],
    "joint_vel": joint_vel[src][phase],
    "body_pos_w": body_pos,
    "body_quat_w": body_quat_w[src][phase],
    "body_lin_vel_w": body_lin_vel_w[src][phase],
    "body_ang_vel_w": body_ang_vel_w[src][phase],
    "cycle_len": length,
    "delta_xy": delta_xy,
    "seam_q": seam_q,
  }


def heading_align_archive(
  body_pos_w: np.ndarray,
  body_quat_w: np.ndarray,
  body_lin_vel_w: np.ndarray,
  body_ang_vel_w: np.ndarray,
  base_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
  """Rotate the clip so mean base yaw faces +X."""
  yaw = quat_wxyz_to_yaw(body_quat_w[:, base_idx])
  yaw_offset = -float(np.mean(np.unwrap(yaw)))
  body_pos_w = rotate_vec(body_pos_w, yaw_offset)
  body_lin_vel_w = rotate_vec(body_lin_vel_w, yaw_offset)
  body_ang_vel_w = rotate_vec(body_ang_vel_w, yaw_offset)
  q_align = yaw_quat_wxyz(yaw_offset)
  body_quat_w = quat_multiply_wxyz(q_align, body_quat_w)
  return body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, yaw_offset


def tile_motion_npz(
  npz_path: Path,
  out_path: Path,
  num_out: int = DEFAULT_NUM_OUT,
  cmd_vx: float | None = None,
) -> dict[str, float]:
  """Load a single-cycle mjlab NPZ, tile it, and write a style archive."""
  with np.load(npz_path, allow_pickle=False) as data:
    files = set(data.files)
    required = {
      "joint_pos",
      "joint_vel",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    }
    missing = sorted(required - files)
    if missing:
      raise KeyError(f"NPZ missing fields: {missing}")
    fps = (
      float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in files else DEFAULT_FPS
    )
    joint_names = _as_str_list(data["joint_names"]) if "joint_names" in files else []
    body_names = _as_str_list(data["body_names"]) if "body_names" in files else []
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
    body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float64)
    body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
    body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float64)

  if not body_names:
    raise KeyError("NPZ has no body_names; cannot locate the base body.")
  name_to_b = {n: i for i, n in enumerate(body_names)}
  if "base_link" not in name_to_b:
    raise KeyError("NPZ body_names has no 'base_link'.")
  base_idx = name_to_b["base_link"]

  body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, yaw_offset = (
    heading_align_archive(
      body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, base_idx
    )
  )
  yaw_aligned = quat_wxyz_to_yaw(body_quat_w[:, base_idx])
  vel_body = world_to_body_vel(body_quat_w[:, base_idx], body_lin_vel_w[:, base_idx])
  mean_vx = float(vel_body[:, 0].mean())
  if cmd_vx is None:
    cmd_vx = max(0.1, round(mean_vx, 1))

  detected = detect_cycle_length(joint_pos)
  tiled = tile_motion(
    joint_pos,
    joint_vel,
    body_pos_w,
    body_quat_w,
    body_lin_vel_w,
    body_ang_vel_w,
    num_out=num_out,
    cycle_len=detected,
    base_idx=base_idx,
  )

  foot_idx = [name_to_b[n] for n in FOOT_BODY_NAMES if n in name_to_b]
  seam_feet = 0.0
  if foot_idx:
    cycle_len = int(tiled["cycle_len"])
    feet_body = feet_in_yaw_frame(
      body_pos_w[:cycle_len, base_idx],
      yaw_aligned[:cycle_len],
      body_pos_w[:cycle_len, foot_idx],
    )
    seam_feet = float(np.linalg.norm(feet_body[0] - feet_body[cycle_len - 1]))

  out_path.parent.mkdir(parents=True, exist_ok=True)
  payload: dict[str, np.ndarray] = {
    "fps": np.asarray(fps, dtype=np.float64),
    "joint_pos": np.asarray(tiled["joint_pos"], dtype=np.float32),
    "joint_vel": np.asarray(tiled["joint_vel"], dtype=np.float32),
    "body_pos_w": np.asarray(tiled["body_pos_w"], dtype=np.float32),
    "body_quat_w": np.asarray(tiled["body_quat_w"], dtype=np.float32),
    "body_lin_vel_w": np.asarray(tiled["body_lin_vel_w"], dtype=np.float32),
    "body_ang_vel_w": np.asarray(tiled["body_ang_vel_w"], dtype=np.float32),
    "cycle_len": np.asarray(tiled["cycle_len"], dtype=np.int64),
    "cmd_vx": np.asarray(cmd_vx, dtype=np.float64),
  }
  if joint_names:
    payload["joint_names"] = np.asarray(joint_names)
  if body_names:
    payload["body_names"] = np.asarray(body_names)
  np.savez_compressed(out_path, **payload)  # type: ignore[arg-type]

  delta_xy = np.asarray(tiled["delta_xy"])
  root_xy = np.asarray(tiled["body_pos_w"])[:, base_idx, :2]
  return {
    "input_frames": float(joint_pos.shape[0]),
    "input_fps": fps,
    "output_frames": float(num_out),
    "cycle_len": float(tiled["cycle_len"]),
    "yaw_offset_deg": float(np.rad2deg(yaw_offset)),
    "mean_yaw_after_deg": float(np.rad2deg(np.mean(yaw_aligned))),
    "mean_vx": mean_vx,
    "cmd_vx": float(cmd_vx),
    "delta_x": float(delta_xy[0]),
    "delta_y": float(delta_xy[1]),
    "com_x_end": float(root_xy[-1, 0]),
    "seam_q": float(tiled["seam_q"]),
    "seam_feet": seam_feet,
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--npz",
    type=Path,
    required=True,
    help="Single-cycle mjlab motion archive from csv_to_npz.",
  )
  parser.add_argument(
    "--out",
    type=Path,
    required=True,
    help="Output tiled style archive (~20 s).",
  )
  parser.add_argument("--num-out", type=int, default=DEFAULT_NUM_OUT)
  parser.add_argument(
    "--cmd-vx",
    type=float,
    default=None,
    help="Override nominal forward command. Default: round(mean body vx, 1).",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if not args.npz.is_file():
    raise FileNotFoundError(f"NPZ not found: {args.npz}")
  stats = tile_motion_npz(args.npz, args.out, num_out=args.num_out, cmd_vx=args.cmd_vx)
  print(f"Wrote {args.out}")
  for key, value in stats.items():
    print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
  main()
