"""Decaying Action Priors (DecAP) on top of joint-position PD targets.

APEX applies a decaying torque prior

    τ = Kp (q_π - q) - Kd q̇ + λ_t Kp (q* - q)

which is algebraically the same as shifting the position target

    q_cmd = q_π + λ_t (q* - q)

Marsdog style follows APEX's default **exp** schedule: λ=1 at step 0
(full prior, the robot can walk immediately) then

    λ = γ^{s/k}    (γ=0.99, k=500 env-steps)

so control is handed to the 72-d actor from the first update. Cosine
(hold then wean) is kept for ablation. Play / ONNX sets
``decap_enabled=False``. Standing uses the default pose as q*, matching
the style-reward gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg

from .commands import StyleCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

DecapSchedule = Literal["cosine", "exp"]


def decap_lambda(
  step: torch.Tensor | int | float,
  gamma: float = 0.99,
  k: float = 500.0,
  *,
  schedule: DecapSchedule = "exp",
  hold_iterations: float = 500.0,
  decay_iterations: float = 6000.0,
  steps_per_iteration: float = 24.0,
) -> torch.Tensor:
  """DecAP gain λ_t.

  ``step`` is ``env.common_step_counter`` (one tick per control step).
  Exp uses that directly; cosine maps it to PPO iters via
  ``steps_per_iteration``.
  """
  step_t = torch.as_tensor(step, dtype=torch.float32)
  if schedule == "exp":
    return torch.pow(torch.as_tensor(gamma, dtype=step_t.dtype), step_t / k)
  iteration = step_t / steps_per_iteration
  phase = ((iteration - hold_iterations) / decay_iterations).clamp(0.0, 1.0)
  return 0.5 * (1.0 + torch.cos(torch.as_tensor(math.pi, dtype=step_t.dtype) * phase))


@dataclass(kw_only=True)
class DecapJointPositionActionCfg(JointPositionActionCfg):
  """Joint-position action with a decaying expert prior."""

  command_name: str = "style"
  """Style command that supplies q*."""
  twist_command_name: str = "twist"
  """Unused; standing is read from the style command's twist gate."""
  decap_enabled: bool = True
  """If False, this is a plain joint-position action (play / export)."""
  schedule: DecapSchedule = "exp"
  """``exp``: APEX default, γ^{s/k} from step 0. ``cosine``: hold then wean."""
  gamma: float = 0.99
  """Decay base for ``schedule='exp'`` (APEX)."""
  k: float = 500.0
  """Exp time-constant in env steps (APEX MarsDog). Unused for cosine."""
  hold_iterations: float = 500.0
  """Cosine-only: PPO iters with λ=1. Unused for exp."""
  decay_iterations: float = 6000.0
  """Cosine-only: PPO iters of wean after the hold. Unused for exp."""
  steps_per_iteration: float = 24.0
  """Must match ``RslRlOnPolicyRunnerCfg.num_steps_per_env``."""

  def build(self, env: ManagerBasedRlEnv) -> DecapJointPositionAction:
    return DecapJointPositionAction(self, env)


class DecapJointPositionAction(JointPositionAction):
  """q_cmd = q_π + λ_t (q*_style - q), then the existing BuiltinPosition PD."""

  def apply_actions(self) -> None:
    cfg = cast(DecapJointPositionActionCfg, self.cfg)
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._processed_actions - encoder_bias
    if cfg.decap_enabled:
      scale = decap_lambda(
        self._env.common_step_counter,
        cfg.gamma,
        cfg.k,
        schedule=cfg.schedule,
        hold_iterations=cfg.hold_iterations,
        decay_iterations=cfg.decay_iterations,
        steps_per_iteration=cfg.steps_per_iteration,
      ).to(device=target.device, dtype=target.dtype)
      style = cast(StyleCommand, self._env.command_manager.get_term(cfg.command_name))
      q_star = style.style_joint_pos[:, self._target_ids]
      q = self._entity.data.joint_pos[:, self._target_ids]
      target = target + scale * (q_star - q)
      self._env.extras.setdefault("log", {})["Metrics/decap_lambda"] = float(scale)
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
