from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )


def bad_anchor_ori(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def bad_anchor_ori_angle(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  """Terminate when reference and robot tilt differ by more than ``threshold``.

  Projected gravity makes this roll/pitch-aware and yaw-invariant. Unlike comparing
  only the gravity z components, the dot product is a true angular distance.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_gravity_b = quat_apply_inverse(command.anchor_quat_w, asset.data.gravity_vec_w)
  robot_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )
  cosine = torch.sum(motion_gravity_b * robot_gravity_b, dim=-1).clamp(-1.0, 1.0)
  return torch.acos(cosine) > threshold


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Terminate when a protected body contacts terrain above a force threshold."""
  sensor: ContactSensor = env.scene[sensor_name]
  if sensor.data.force_history is not None:
    force = torch.norm(sensor.data.force_history, dim=-1)
    return (force > force_threshold).any(dim=-1).any(dim=-1)
  assert sensor.data.force is not None
  return (torch.norm(sensor.data.force, dim=-1) > force_threshold).any(dim=-1)


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)
