"""Script to play RL agent with RSL-RL."""

import csv
import os
import sys
import time as _time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.style.mdp.commands import StyleCommandCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.base import PolicyProtocol
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago


def _parse_wandb_dt(value: str | datetime) -> datetime:
  """Parse a W&B datetime string (or pass through a datetime object)."""
  if isinstance(value, str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
  return value


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  save_rollout_csv: str | None = None
  """If set, save actor observations and actions to this CSV path during play."""
  save_torque_csv: str | None = None
  """If set, save actuated-joint output torques to this CSV path during play."""
  rollout_obs_group: str = "actor"
  """Observation group to record when ``save_rollout_csv`` is set."""
  rollout_env_id: int = 0
  """Environment index to record when ``save_rollout_csv`` is set."""
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def _wrap_policy_for_rollout_csv(
  policy: PolicyProtocol,
  *,
  obs_group: str,
  env_id: int,
) -> tuple[PolicyProtocol, Callable[[str], None]]:
  """Wrap policy to buffer obs/action rows; flush writes a CSV on play exit."""
  rows: list[tuple[list[float], list[float]]] = []

  class RecordingPolicy:
    def __call__(self, obs: Any) -> torch.Tensor:
      actions = policy(obs)
      if obs_group not in obs:
        raise KeyError(
          f"Observation group '{obs_group}' not found. "
          f"Available groups: {list(obs.keys())}"
        )
      obs_row = obs[obs_group][env_id].detach().cpu().tolist()
      action_row = actions[env_id].detach().cpu().tolist()
      rows.append((obs_row, action_row))
      return actions

    def reset(self) -> None:
      rows.clear()
      reset_fn = getattr(policy, "reset", None)
      if reset_fn is not None:
        reset_fn()

  def flush(path: str) -> None:
    if not rows:
      print("[WARN]: No rollout rows recorded; CSV not written.")
      return
    obs_dim = len(rows[0][0])
    action_dim = len(rows[0][1])
    header = [f"obs_{i}" for i in range(obs_dim)] + [
      f"action_{i}" for i in range(action_dim)
    ]
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(header)
      for obs_row, action_row in rows:
        writer.writerow(obs_row + action_row)
    print(
      f"[INFO]: Saved rollout CSV to {output_path} "
      f"(steps={len(rows)}, obs_dim={obs_dim}, action_dim={action_dim})"
    )

  return RecordingPolicy(), flush


def _wrap_policy_for_torque_csv(
  policy: PolicyProtocol,
  *,
  joint_names: tuple[str, ...],
  read_torques: Callable[[], torch.Tensor],
  step_dt: float,
) -> tuple[PolicyProtocol, Callable[[str], None]]:
  """Wrap a policy to record joint-side applied torques at each control step."""
  rows: list[list[float]] = []

  class TorqueRecordingPolicy:
    def __call__(self, obs: Any) -> torch.Tensor:
      torques = read_torques().detach().cpu().tolist()
      if len(torques) != len(joint_names):
        raise ValueError(
          f"Torque dimension mismatch: got {len(torques)}, expected {len(joint_names)}"
        )
      rows.append(torques)
      return policy(obs)

    def reset(self) -> None:
      rows.clear()
      reset_fn = getattr(policy, "reset", None)
      if reset_fn is not None:
        reset_fn()

  def flush(path: str) -> None:
    if not rows:
      print("[WARN]: No torque rows recorded; CSV not written.")
      return
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(
        ["step", "time_s"]
        + [f"torque_{i}_{name}_Nm" for i, name in enumerate(joint_names)]
      )
      for step, torque_row in enumerate(rows):
        writer.writerow([step, step * step_dt] + torque_row)
    print(
      f"[INFO]: Saved joint torque CSV to {output_path} "
      f"(steps={len(rows)}, joints={len(joint_names)})"
    )

  return TorqueRecordingPolicy(), flush


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  is_style_task = "style" in env_cfg.commands and isinstance(
    env_cfg.commands["style"], StyleCommandCfg
  )
  if is_style_task:
    style_cmd = env_cfg.commands["style"]
    assert isinstance(style_cmd, StyleCommandCfg)
    if cfg.motion_file is None or not Path(cfg.motion_file).exists():
      raise ValueError(
        "Style tasks require a local tiled motion npz:\n"
        "  --motion-file /path/to/xingzou_style.npz"
      )
    print(f"[INFO]: Using style motion file: {cfg.motion_file}")
    style_cmd.motion_file = cfg.motion_file

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Build checkpoint manager for hot-swapping checkpoints in the viewer.
  ckpt_manager: CheckpointManager | None = None
  if TRAINED_MODE and resume_path is not None:
    _ckpt_runner = runner  # pyright: ignore[reportPossiblyUnboundVariable]

    def _reload_policy(path: str):
      _ckpt_runner.load(
        path,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
      )
      return _ckpt_runner.get_inference_policy(device=device)

    if cfg.wandb_run_path is None:
      ckpt_dir = resume_path.parent

      def fetch_available_local() -> list[tuple[str, str]]:
        now = _time.time()
        entries: list[tuple[str, str, int]] = []
        for f in sorted(ckpt_dir.glob("*.pt")):
          try:
            step = int(f.stem.split("_")[1])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(int(now - f.stat().st_mtime))
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_local,
        load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
      )
    else:
      import wandb

      api = wandb.Api()
      run_path = str(cfg.wandb_run_path)
      wandb_run = api.run(run_path)
      _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

      def fetch_available_wandb() -> list[tuple[str, str]]:
        wandb_run.load()
        now = datetime.now(tz=timezone.utc)
        entries: list[tuple[str, str, int]] = []
        for f in wandb_run.files():
          if not f.name.endswith(".pt"):
            continue
          try:
            step = int(f.name.split("_")[1].split(".")[0])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(
            int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
          )
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_wandb,
        load_checkpoint=lambda name: _reload_policy(
          str(get_wandb_checkpoint_path(_log_root, Path(run_path), name)[0])
        ),
        run_name=_parse_wandb_dt(wandb_run.created_at).strftime("%Y-%m-%d_%H-%M-%S"),
        run_url=wandb_run.url,
        run_status=wandb_run.state,
      )

  flush_rollout_csv: Callable[[str], None] | None = None
  flush_torque_csv: Callable[[str], None] | None = None
  if cfg.save_rollout_csv is not None or cfg.save_torque_csv is not None:
    if not (0 <= cfg.rollout_env_id < env.num_envs):
      env.close()
      raise ValueError(
        f"rollout_env_id out of range: {cfg.rollout_env_id} (num_envs={env.num_envs})"
      )
  if cfg.save_rollout_csv is not None:
    policy, flush_rollout_csv = _wrap_policy_for_rollout_csv(
      policy,
      obs_group=cfg.rollout_obs_group,
      env_id=cfg.rollout_env_id,
    )
  if cfg.save_torque_csv is not None:
    robot = env.unwrapped.scene["robot"]
    joint_ids, joint_names = robot.find_joints_by_actuator_names((".*",))
    torque_env_id = cfg.rollout_env_id

    def read_torques() -> torch.Tensor:
      # qfrc_actuator is the applied low-side joint torque, including MuJoCo's
      # actuator force clipping. This is the quantity comparable to real joints.
      return robot.data.qfrc_actuator[torque_env_id, joint_ids]

    policy, flush_torque_csv = _wrap_policy_for_torque_csv(
      policy,
      joint_names=tuple(joint_names),
      read_torques=read_torques,
      step_dt=env.unwrapped.step_dt,
    )

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  try:
    if resolved_viewer == "native":
      NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
      ViserPlayViewer(env, policy, checkpoint_manager=ckpt_manager).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    if flush_rollout_csv is not None and cfg.save_rollout_csv is not None:
      flush_rollout_csv(cfg.save_rollout_csv)
    if flush_torque_csv is not None and cfg.save_torque_csv is not None:
      flush_torque_csv(cfg.save_torque_csv)

  env.close()


def main():
  maybe_print_top_level_help("play")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
