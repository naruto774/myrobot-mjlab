"""Domain randomization functions for sensor (IMU) readings.

These randomizers do not touch the MuJoCo model; they only write per-environment
bias buffers stored on ``EntityData`` (``imu_ori_bias`` / ``imu_gyro_bias``). The
buffers are read by the (biased) observation functions so the policy is trained to
be robust against a persistent, low-frequency IMU bias/drift — the dominant error
mode on real hardware, which a zero-mean per-step observation noise cannot model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

from ._core import _DEFAULT_ASSET_CFG

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def imu_bias(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ori_range: tuple[float, float] = (0.0, 0.0),
  gyro_range: tuple[float, float] = (0.0, 0.0),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Randomize IMU orientation (roll/pitch) and gyro biases.

  Samples a constant-per-episode bias for each environment and writes it to the
  entity's ``imu_ori_bias`` (roll/pitch, rad) and ``imu_gyro_bias`` (3-axis, rad/s)
  buffers. Use with ``mode="reset"`` so the bias is resampled every episode.

  Args:
    env: The RL environment.
    env_ids: Environment indices to randomize. If ``None``, all envs.
    ori_range: Uniform range (rad) for the roll and pitch orientation bias.
    gyro_range: Uniform range (rad/s) for the per-axis gyro bias.
    asset_cfg: Asset selection (defaults to ``"robot"``).
  """
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  num = len(env_ids)
  asset.data.imu_ori_bias[env_ids] = sample_uniform(
    torch.tensor(ori_range[0], device=env.device),
    torch.tensor(ori_range[1], device=env.device),
    (num, 2),
    env.device,
  )
  asset.data.imu_gyro_bias[env_ids] = sample_uniform(
    torch.tensor(gyro_range[0], device=env.device),
    torch.tensor(gyro_range[1], device=env.device),
    (num, 3),
    env.device,
  )
