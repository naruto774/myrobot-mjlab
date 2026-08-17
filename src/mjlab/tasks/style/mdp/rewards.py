"""Style and task reward terms.

Group 1 (style) matches cyclic joint angles, yaw-frame feet, and roll/pitch
tilt. Group 2 tracks commanded vx / ωz and applies Go2-flat penalties.

Standing (‖v_cmd,xy‖ < 0.1) swaps style targets to the default pose inside
:class:`StyleCommand`; these terms just consume those targets.

Style kernels follow APEX MarsDog: ``exp(-mean((x-x*)²) / σ)``. Using
``sum / σ²`` saturates 14-DoF leg terms at 0 while the robot is still far
from the clip (RMS ~0.26 rad → exp(-380)).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from .commands import StyleCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _exp_kernel(mean_sq: torch.Tensor, std: float) -> torch.Tensor:
  """APEX imitation kernel: exp(-mean_i (x_i-x*_i)² / σ)."""
  return torch.exp(-mean_sq / std)


def imitate_joint_pos_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """exp(-mean((q-q*_φ)²) / σ) on a joint subset (legs / waist / head)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = cast(StyleCommand, env.command_manager.get_term(command_name))
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q_star = command.style_joint_pos[:, asset_cfg.joint_ids]
  error = torch.mean(torch.square(q - q_star), dim=-1)
  return _exp_kernel(error, std)


def imitate_feet_yaw_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """exp(-mean((p_feet^B - p*_φ^B)²) / σ) in the yaw body frame."""
  asset: Entity = env.scene[asset_cfg.name]
  command = cast(StyleCommand, env.command_manager.get_term(command_name))
  root_pos = asset.data.root_link_pos_w
  root_quat = asset.data.root_link_quat_w
  foot_pos = asset.data.body_link_pos_w[:, command.foot_body_ids]
  rel = foot_pos - root_pos[:, None, :]
  yaw_q = yaw_quat(root_quat)
  n_feet = rel.shape[1]
  feet_b = quat_apply_inverse(yaw_q[:, None, :].expand(-1, n_feet, -1), rel)
  error = torch.mean(torch.square(feet_b - command.style_feet_yaw_b), dim=(1, 2))
  return _exp_kernel(error, std)


def imitate_tilt_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Match roll/pitch via projected gravity xy; yaw is intentionally omitted."""
  asset: Entity = env.scene[asset_cfg.name]
  command = cast(StyleCommand, env.command_manager.get_term(command_name))
  g_xy = asset.data.projected_gravity_b[:, :2]
  error = torch.mean(torch.square(g_xy - command.style_g_xy), dim=-1)
  return _exp_kernel(error, std)


def imitation_height_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """-(h - h*)² relative to the expert (or default when standing)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = cast(StyleCommand, env.command_manager.get_term(command_name))
  height = asset.data.root_link_pos_w[:, 2]
  return torch.square(height - command.style_root_height)
