"""myrobot 站立/参考一致性诊断脚本（只读，不改任何配置）。

功能：
  1. home 位姿（INIT_STATE）下的整机质心 subtree_com、总质量、质心高度；
  2. 双脚 collision geom 的世界坐标与最低点 z（判断悬空 / 穿地 / 单脚触地）；
  3. 质心 xy 相对双脚支撑域中心的横向偏移（解释“固定往某侧倒”）；
  4. 参考动作 frame-0 的基座线/角速度、关节初速度（判断复位是否被“踹一脚”）；
  5. default_joint_pos（home） 与 motion[0] 的逐关节差（判断零动作目标是否= 参考首帧）。

同时对 home 位姿 与 motion frame-0 位姿 各算一遍 (1)(2)(3)，方便对比。

运行：
  uv run python diagnose_stance.py --motion-file /tmp/motion.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.myrobot.myrobot_constants import get_myrobot_robot_cfg
from mjlab.entity.entity import Entity

FOOT_GEOMS = (
  "r_foot_1_collision",
  "r_foot_2_collision",
  "l_foot_1_collision",
  "l_foot_2_collision",
)


def _com_and_feet(mjm: mujoco.MjModel, mjd: mujoco.MjData, base_id: int) -> dict:
  """前向运动学后，提取质心与双脚信息。"""
  mujoco.mj_forward(mjm, mjd)

  com = mjd.subtree_com[base_id].copy()  # 整机质心（世界系）
  total_mass = float(mjm.body_subtreemass[base_id])

  feet = {}
  lowest_pts = []
  centers_xy = []
  for name in FOOT_GEOMS:
    gid = mjm.geom(name).id
    pos = mjd.geom_xpos[gid].copy()
    radius = float(mjm.geom_size[gid][0])  # 圆柱半径
    lowest_z = pos[2] - radius  # 轴沿水平 x，最低点≈中心 z - 半径
    feet[name] = (pos, lowest_z)
    lowest_pts.append(lowest_z)
    centers_xy.append(pos[:2])

  centers_xy = np.array(centers_xy)
  support_centroid = centers_xy.mean(axis=0)
  return {
    "com": com,
    "total_mass": total_mass,
    "feet": feet,
    "min_foot_z": float(min(lowest_pts)),
    "max_foot_z": float(max(lowest_pts)),
    "support_centroid_xy": support_centroid,
    "support_x_range": (float(centers_xy[:, 0].min()), float(centers_xy[:, 0].max())),
    "support_y_range": (float(centers_xy[:, 1].min()), float(centers_xy[:, 1].max())),
  }


def _print_pose_report(tag: str, info: dict) -> None:
  com = info["com"]
  c = info["support_centroid_xy"]
  print(f"\n===== {tag} =====")
  print(f"  总质量 total_mass        : {info['total_mass']:.4f} kg")
  print(f"  整机质心 com (x,y,z)     : ({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:+.4f}) m")
  print(
    f"  双脚最低点 z (min/max)   : {info['min_foot_z']:+.4f} / {info['max_foot_z']:+.4f} m  (地面 z=0)"
  )
  print(f"  支撑域中心 xy            : ({c[0]:+.4f}, {c[1]:+.4f}) m")
  print(
    f"  支撑域 x 范围            : [{info['support_x_range'][0]:+.4f}, {info['support_x_range'][1]:+.4f}] m"
  )
  print(
    f"  支撑域 y 范围            : [{info['support_y_range'][0]:+.4f}, {info['support_y_range'][1]:+.4f}] m"
  )
  dx = com[0] - c[0]
  dy = com[1] - c[1]
  print(
    f"  质心相对支撑中心偏移     : 前后 dx={dx:+.4f} m, 横向 dy={dy:+.4f} m  (+y=左, -y=右)"
  )
  # 横向裕度：质心 y 是否落在两脚 y 范围内
  ylo, yhi = info["support_y_range"]
  inside_y = ylo <= com[1] <= yhi
  print(
    f"  质心 y 是否落在双脚 y 区间: {'是' if inside_y else '否 (已越界 → 必往该侧倒)'}"
  )
  print("  各脚 geom 世界坐标 / 最低 z:")
  for name, (pos, lz) in info["feet"].items():
    print(
      f"    {name:20s} pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f})  lowest_z={lz:+.4f}"
    )


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--motion-file", default="/tmp/motion.npz")
  args = ap.parse_args()

  robot = Entity(get_myrobot_robot_cfg())
  mjm = robot.spec.compile()
  mjd = mujoco.MjData(mjm)
  base_id = mjm.body("base_link").id

  joint_names = list(robot.joint_names)  # entity 关节顺序（= qpos[7:] 顺序）
  nq = mjm.nq
  njnt = len(joint_names)
  print(f"[INFO] nq={nq}, 关节数={njnt}, base_id={base_id}")

  key = mjm.key("init_state")
  home_qpos = np.array(key.qpos, dtype=float)
  home_joint_pos = home_qpos[7:].copy()  # entity 关节顺序

  # ---- (A) home 位姿 ----
  mjd.qpos[:] = home_qpos
  home_info = _com_and_feet(mjm, mjd, base_id)
  _print_pose_report("HOME 位姿 (INIT_STATE / 零动作目标)", home_info)

  # ---- 读取 motion ----
  motion_path = Path(args.motion_file)
  if not motion_path.exists():
    print(f"\n[WARN] 找不到 motion 文件: {motion_path}，跳过参考相关诊断。")
    return
  m = np.load(motion_path)
  m_joint_pos0 = m["joint_pos"][0]  # entity 关节顺序
  m_joint_vel0 = m["joint_vel"][0]
  base_pos0 = m["body_pos_w"][0, 0]  # base_link = body 0
  base_quat0 = m["body_quat_w"][0, 0]  # wxyz
  base_linvel0 = m["body_lin_vel_w"][0, 0]
  base_angvel0 = m["body_ang_vel_w"][0, 0]

  # ---- (B) motion frame-0 位姿 ----
  ref_qpos = home_qpos.copy()
  ref_qpos[0:3] = base_pos0
  ref_qpos[3:7] = base_quat0
  ref_qpos[7:] = m_joint_pos0
  mjd.qpos[:] = ref_qpos
  ref_info = _com_and_feet(mjm, mjd, base_id)
  _print_pose_report("MOTION frame-0 位姿 (参考首帧)", ref_info)

  # ---- (C) frame-0 速度 ----
  print("\n===== 参考 frame-0 速度（复位时写入仿真）=====")
  print(
    f"  基座线速度 |v|           : {np.linalg.norm(base_linvel0):.4f} m/s  {base_linvel0}"
  )
  print(
    f"  基座角速度 |w|           : {np.linalg.norm(base_angvel0):.4f} rad/s {base_angvel0}"
  )
  print(f"  关节初速度 max|qd|       : {np.abs(m_joint_vel0).max():.4f} rad/s")

  # ---- (C2) 全程扫描：脚离地间隙 / 横向基底宽度 / 质心横向越界 ----
  print("\n===== 全程扫描 (逐帧静态运动学) =====")
  n_frames = m["joint_pos"].shape[0]
  bp = m["body_pos_w"]  # (T, nbody, 3)
  bq = m["body_quat_w"]
  jp = m["joint_pos"]
  foot_ids = [mjm.geom(n).id for n in FOOT_GEOMS]
  radii = np.array([mjm.geom_size[g][0] for g in foot_ids])

  min_clear = np.full(n_frames, np.nan)
  lat_width = np.full(n_frames, np.nan)
  com_y_margin = np.full(n_frames, np.nan)  # >0 表示质心 y 越出双脚 y 区间
  step = max(1, n_frames // 2000)  # 大动作下采样，最多 ~2000 帧
  for t in range(0, n_frames, step):
    q = home_qpos.copy()
    q[0:3] = bp[t, 0]
    q[3:7] = bq[t, 0]
    q[7:] = jp[t]
    mjd.qpos[:] = q
    mujoco.mj_forward(mjm, mjd)
    fxy = mjd.geom_xpos[foot_ids][:, :2]
    fz_low = mjd.geom_xpos[foot_ids][:, 2] - radii
    com = mjd.subtree_com[base_id]
    min_clear[t] = fz_low.min()  # 最低脚点 z（相对地面 0）
    lat_width[t] = fxy[:, 1].max() - fxy[:, 1].min()  # 双脚横向跨度
    ylo, yhi = fxy[:, 1].min(), fxy[:, 1].max()
    com_y_margin[t] = max(ylo - com[1], com[1] - yhi)  # >0 = 越界

  valid = ~np.isnan(min_clear)
  mc, lw, cy = min_clear[valid], lat_width[valid], com_y_margin[valid]
  out_frac = float((cy > 0).mean())
  print(f"  扫描帧数(下采样)         : {valid.sum()} / {n_frames}")
  print(
    f"  最低脚点 z  min/mean/max : {mc.min():+.4f} / {mc.mean():+.4f} / {mc.max():+.4f} m"
  )
  print(f"    (远<0=持续穿地, 远>0=持续悬空; 正常落脚应 ~0)")
  print(
    f"  双脚横向跨度 min/mean/max: {lw.min():.4f} / {lw.mean():.4f} / {lw.max():.4f} m"
  )
  print(f"    (髋间距 ~0.05m; 跨度持续 <~0.03m 说明步态近乎单线, 横向极不稳)")
  print(f"  质心 y 越界量 max        : {cy.max():+.4f} m  (>0 即质心落在双脚 y 区间外)")
  print(f"  质心 y 越界帧占比        : {out_frac * 100:.1f} %")

  # ---- (D) default_joint_pos vs motion[0] 逐关节差 ----
  print("\n===== default_joint_pos (home) vs motion[0] 逐关节差 (按 |diff| 降序) =====")
  diffs = home_joint_pos - m_joint_pos0
  order = np.argsort(-np.abs(diffs))
  print(f"  {'joint':24s} {'home':>9s} {'motion[0]':>10s} {'diff':>9s}")
  for i in order:
    print(
      f"  {joint_names[i]:24s} {home_joint_pos[i]:+9.4f} "
      f"{m_joint_pos0[i]:+10.4f} {diffs[i]:+9.4f}"
    )
  print(
    f"\n  |diff| 最大 = {np.abs(diffs).max():.4f} rad, "
    f"L2 = {np.linalg.norm(diffs):.4f} rad"
  )
  print("  说明: 若该差异显著(>0.1 rad)，零动作目标 != 参考首帧，")
  print("        且策略训练时需长期对抗该固定偏置。")


if __name__ == "__main__":
  main()
