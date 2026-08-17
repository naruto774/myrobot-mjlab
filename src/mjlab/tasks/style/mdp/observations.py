"""Style-task observations.

Actor (72-d, deployable): ω + g + cmd + q + q̇ + a, no phase, no v_lin.
Critic adds privileged style features: v_lin, (sin φ, cos φ), q*, feet*, g_xy*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_mul,
  yaw_quat,
)

from .commands import StyleCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def gait_phase(env: ManagerBasedRlEnv, command_name: str = "style") -> torch.Tensor:
  command = cast(StyleCommand, env.command_manager.get_term(command_name))
  return command.gait_phase


def style_joint_pos(
  env: ManagerBasedRlEnv,
  command_name: str = "style",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  command = cast(StyleCommand, env.command_manager.get_term(command_name))
  return command.style_joint_pos[:, asset_cfg.joint_ids]


def style_feet_yaw_b(
  env: ManagerBasedRlEnv, command_name: str = "style"
) -> torch.Tensor:
  command = cast(StyleCommand, env.command_manager.get_term(command_name))
  return command.style_feet_yaw_b.reshape(env.num_envs, -1)


def style_g_xy(env: ManagerBasedRlEnv, command_name: str = "style") -> torch.Tensor:
  command = cast(StyleCommand, env.command_manager.get_term(command_name))
  return command.style_g_xy


def robot_feet_yaw_b(
  env: ManagerBasedRlEnv,
  command_name: str = "style",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Robot feet in the yaw body frame, matching the style reward."""
  asset: Entity = env.scene[asset_cfg.name]
  command = cast(StyleCommand, env.command_manager.get_term(command_name))
  root_pos = asset.data.root_link_pos_w
  root_quat = asset.data.root_link_quat_w
  foot_pos = asset.data.body_link_pos_w[:, command.foot_body_ids]
  rel = foot_pos - root_pos[:, None, :]
  yaw_q = yaw_quat(root_quat)
  n_feet = rel.shape[1]
  feet_b = quat_apply_inverse(yaw_q[:, None, :].expand(-1, n_feet, -1), rel)
  return feet_b.reshape(env.num_envs, -1)


def projected_gravity_biased(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Projected gravity after a constant-per-episode IMU roll/pitch bias."""
  asset: Entity = env.scene[asset_cfg.name]
  bias = asset.data.imu_ori_bias
  delta_quat = quat_from_euler_xyz(bias[:, 0], bias[:, 1], torch.zeros_like(bias[:, 0]))
  biased_quat = quat_mul(delta_quat, asset.data.root_link_quat_w)
  gravity_w = torch.zeros(env.num_envs, 3, device=env.device)
  gravity_w[:, 2] = -1.0
  return quat_apply_inverse(biased_quat, gravity_w)


def base_ang_vel_biased(
  env: ManagerBasedRlEnv,
  sensor_name: str = "robot/imu_ang_vel",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """IMU gyro plus a constant-per-episode zero-bias."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  asset: Entity = env.scene[asset_cfg.name]
  return sensor.data + asset.data.imu_gyro_bias
