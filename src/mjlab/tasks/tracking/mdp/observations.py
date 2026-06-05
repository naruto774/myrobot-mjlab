from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_from_euler_xyz,
  quat_mul,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_ori_b_biased(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Anchor orientation error with a persistent IMU roll/pitch bias injected.

  Same as :func:`motion_anchor_ori_b`, but the robot's base orientation reading is
  first pre-rotated by a constant-per-episode roll/pitch bias (``imu_ori_bias``,
  sampled by the ``dr.imu_bias`` event). This trains the actor to be invariant to a
  small persistent IMU tilt offset, the dominant Sim-to-Real error on hardware.

  Mathematically, with ``ΔR`` the bias rotation and ``R_robot`` the true base
  orientation, the biased reading is ``R̂_robot = ΔR · R_robot`` and the observed
  error becomes ``R̂_robotᵀ · R_ref``. Only the actor observation is affected;
  rewards/terminations keep using the true state (asymmetric actor-critic).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  bias = command.robot.data.imu_ori_bias  # (num_envs, 2): roll, pitch.
  delta_quat = quat_from_euler_xyz(bias[:, 0], bias[:, 1], torch.zeros_like(bias[:, 0]))
  robot_anchor_quat_biased = quat_mul(delta_quat, command.robot_anchor_quat_w)

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    robot_anchor_quat_biased,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def base_ang_vel_biased(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Gyro (base angular velocity) sensor reading with a persistent zero-bias added.

  Reads the built-in IMU gyro sensor and adds the constant-per-episode
  ``imu_gyro_bias`` (sampled by the ``dr.imu_bias`` event). Only the actor sees the
  bias; the critic reads the clean sensor.
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  asset: Entity = env.scene[asset_cfg.name]
  return sensor.data + asset.data.imu_gyro_bias


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)
