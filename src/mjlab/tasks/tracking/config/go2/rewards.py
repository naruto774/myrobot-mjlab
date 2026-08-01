"""Marsdog-specific rewards for motion tracking.

The generic tracking rewards treat every tracked body with the same weight.  For
Marsdog, locomotion quality is dominated by base stability, foot placement, and
contact behavior, so these terms keep the imitation signal focused on the parts
that matter for quadruped gait control and sim-to-real transfer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def motion_anchor_height_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Track only the reference base height.

  Absolute x/y tracking is deliberately excluded: the policy should learn the
  reference gait and body height without overfitting to global translation drift
  in the motion clip.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.square(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
  return torch.exp(-error / std**2)


def motion_joint_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track Marsdog reference joint positions for selected joints."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene[asset_cfg.name]
  error = torch.square(
    command.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.joint_pos[:, asset_cfg.joint_ids]
  )
  return torch.exp(-error.mean(dim=1) / std**2)


def motion_joint_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track Marsdog reference joint velocities for selected joints."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene[asset_cfg.name]
  error = torch.square(
    command.joint_vel[:, asset_cfg.joint_ids]
    - asset.data.joint_vel[:, asset_cfg.joint_ids]
  )
  return torch.exp(-error.mean(dim=1) / std**2)


def feet_slip_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize horizontal foot velocity while the foot is in contact."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None

  in_contact = (contact_sensor.data.found > 0).float()
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  slip_speed_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)
  return torch.mean(slip_speed_sq * in_contact, dim=1)


def soft_landing_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 0.0,
) -> torch.Tensor:
  """Penalize impact impulses on first contact.

  The threshold leaves normal support forces untouched and only shapes sharp
  touchdown impacts, which helps keep four-foot contacts smooth in hardware.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.force is not None

  force = torch.norm(contact_sensor.data.force, dim=-1)
  impact = torch.clamp(force - force_threshold, min=0.0)
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt).float()
  return torch.mean(impact * first_contact, dim=1)
