"""Mimic joint actuators.

These actuators are useful for passive or mechanically coupled joints that should
not appear in the policy action space. A target joint receives an internal motor
command computed from a source joint state, approximating a kinematic linkage in
backends where MuJoCo equality constraints may not be available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class MimicJointPositionActuatorCfg(IdealPdActuatorCfg):
  """PD actuator whose target position is derived from another joint.

  For each target joint, the desired position and velocity are:

  ``q_target_des = multiplier * q_source + offset``
  ``qd_target_des = multiplier * qd_source``

  This keeps the mimic joint out of the policy action space while still applying
  actuator forces that approximate a passive mechanical linkage.
  """

  source_joint_names: tuple[str, ...]
  """Source joints that drive each target joint."""

  multipliers: tuple[float, ...]
  """Linear mimic coefficients, one per target joint."""

  offsets: tuple[float, ...] = ()
  """Position offsets, one per target joint. Defaults to zero for every target."""

  def __post_init__(self) -> None:
    super().__post_init__()
    if len(self.source_joint_names) != len(self.target_names_expr):
      raise ValueError(
        "source_joint_names must have the same length as target_names_expr."
      )
    if len(self.multipliers) != len(self.target_names_expr):
      raise ValueError("multipliers must have the same length as target_names_expr.")
    if self.offsets and len(self.offsets) != len(self.target_names_expr):
      raise ValueError("offsets must be empty or match target_names_expr length.")

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> MimicJointPositionActuator:
    return MimicJointPositionActuator(self, entity, target_ids, target_names)


class MimicJointPositionActuator(IdealPdActuator[MimicJointPositionActuatorCfg]):
  """Position mimic actuator implemented as an internal motor PD loop."""

  def __init__(
    self,
    cfg: MimicJointPositionActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self._source_ids: torch.Tensor | None = None
    self._multipliers: torch.Tensor | None = None
    self._offsets: torch.Tensor | None = None

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)

    name_to_joint_id = {name: i for i, name in enumerate(self.entity.joint_names)}
    missing = [
      name for name in self.cfg.source_joint_names if name not in name_to_joint_id
    ]
    if missing:
      raise ValueError(
        f"Source joint(s) not found for {self.__class__.__name__}: {missing}"
      )

    self._source_ids = torch.tensor(
      [name_to_joint_id[name] for name in self.cfg.source_joint_names],
      dtype=torch.long,
      device=device,
    )
    self._multipliers = torch.tensor(
      self.cfg.multipliers,
      dtype=torch.float,
      device=device,
    )
    offsets = self.cfg.offsets or (0.0,) * len(self.cfg.target_names_expr)
    self._offsets = torch.tensor(offsets, dtype=torch.float, device=device)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self.stiffness is not None
    assert self.damping is not None
    assert self._source_ids is not None
    assert self._multipliers is not None
    assert self._offsets is not None

    source_pos = self.entity.data.joint_pos[:, self._source_ids]
    source_vel = self.entity.data.joint_vel[:, self._source_ids]

    desired_pos = source_pos * self._multipliers + self._offsets
    desired_vel = source_vel * self._multipliers

    pos_error = desired_pos - cmd.pos
    vel_error = desired_vel - cmd.vel

    computed_torques = self.stiffness * pos_error
    computed_torques += self.damping * vel_error
    computed_torques += cmd.effort_target

    return self._clip_effort(computed_torques)
