"""Tests for the Marsdog style-imitation task."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

import mjlab.tasks  # noqa: F401
from mjlab.asset_zoo.robots.marsdog.marsdog_constants import MARSDOG_JOINT_NAMES
from mjlab.rl import RslRlMultiCriticPpoAlgorithmCfg, RslRlOnPolicyRunnerCfg
from mjlab.rl.multi_critic import fuse_group_advantages
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.style.mdp.actions import DecapJointPositionActionCfg, decap_lambda
from mjlab.tasks.style.mdp.commands import StyleCommandCfg, StyleVelocityCommandCfg
from mjlab.tasks.style.mdp.rewards import _exp_kernel

TASK_ID = "Mjlab-Style-Flat-Marsdog"
_TILE_PATH = (
  Path(__file__).resolve().parents[1] / "scripts" / "tools" / "tile_motion_npz.py"
)


def _load_tile_module():
  spec = importlib.util.spec_from_file_location("tile_motion_npz", _TILE_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _synthetic_cycle_npz(path: Path, n: int = 20, fps: float = 50.0) -> None:
  joint_names = list(MARSDOG_JOINT_NAMES) + ["rl_tarsus_joint", "rr_tarsus_joint"]
  body_names = [
    "base_link",
    "fl_foot_link",
    "fr_foot_link",
    "rl_foot_link",
    "rr_foot_link",
  ]
  t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
  joint_pos = np.zeros((n, len(joint_names)), dtype=np.float64)
  joint_pos[:, 0] = 0.2 * np.sin(t)
  joint_vel = np.zeros_like(joint_pos)
  body_pos = np.zeros((n, len(body_names), 3), dtype=np.float64)
  body_pos[:, 0, 0] = 0.01 * np.arange(n)
  body_pos[:, 0, 2] = 0.28
  for i, foot in enumerate(range(1, 5)):
    body_pos[:, foot, 0] = body_pos[:, 0, 0] + (0.1 if i % 2 == 0 else -0.1)
    body_pos[:, foot, 1] = 0.08 if i < 2 else -0.08
  body_quat = np.zeros((n, len(body_names), 4), dtype=np.float64)
  body_quat[..., 0] = 1.0
  body_lin = np.zeros((n, len(body_names), 3), dtype=np.float64)
  body_lin[:, 0, 0] = 0.5
  body_ang = np.zeros_like(body_lin)
  np.savez(
    path,
    fps=np.asarray(fps),
    joint_names=np.asarray(joint_names),
    body_names=np.asarray(body_names),
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=body_pos,
    body_quat_w=body_quat,
    body_lin_vel_w=body_lin,
    body_ang_vel_w=body_ang,
  )


def test_tile_motion_npz_accumulates_xy_and_writes_metadata(tmp_path: Path) -> None:
  tile = _load_tile_module()
  src = tmp_path / "cycle.npz"
  out = tmp_path / "style.npz"
  _synthetic_cycle_npz(src, n=20)
  stats = tile.tile_motion_npz(src, out, num_out=100)
  assert out.is_file()
  assert stats["output_frames"] == 100.0
  assert stats["cycle_len"] == 20.0
  data = np.load(out)
  assert data["joint_pos"].shape[0] == 100
  assert int(data["cycle_len"]) == 20
  assert float(data["cmd_vx"]) == pytest.approx(0.5)
  # Five laps of Δx = 0.01 * 19 along the base x axis.
  base_x = data["body_pos_w"][:, 0, 0]
  assert base_x[0] == pytest.approx(0.0)
  assert base_x[20] == pytest.approx(0.19)
  assert base_x[-1] > base_x[20]


def test_style_task_registered_with_go2_twist_and_grouped_rewards() -> None:
  cfg = load_env_cfg(TASK_ID)
  assert "style" in cfg.commands
  assert "twist" in cfg.commands
  style = cfg.commands["style"]
  twist = cfg.commands["twist"]
  assert isinstance(style, StyleCommandCfg)
  assert isinstance(twist, StyleVelocityCommandCfg)
  assert twist.heading_command is False
  assert twist.ranges.lin_vel_y == (0.0, 0.0)
  assert twist.ranges.ang_vel_z[0] < 0.0 < twist.ranges.ang_vel_z[1]
  assert twist.resampling_time_range == (5.0, 5.0)
  assert cfg.episode_length_s == 19.0

  groups = {term.group for term in cfg.rewards.values()}
  assert groups == {0, 1}

  actor_terms = cfg.observations["actor"].terms
  assert list(actor_terms) == [
    "base_ang_vel",
    "projected_gravity",
    "command",
    "joint_pos",
    "joint_vel",
    "actions",
  ]
  assert "gait_phase" not in actor_terms
  assert "base_lin_vel" not in actor_terms
  assert actor_terms["command"].scale == (2.0, 2.0, 0.25)
  assert actor_terms["joint_vel"].scale == 0.05
  joint_names = actor_terms["joint_pos"].params["asset_cfg"].joint_names
  assert joint_names == MARSDOG_JOINT_NAMES
  assert len(MARSDOG_JOINT_NAMES) == 21
  # ω(3)+g(3)+cmd(3)+q(21)+dq(21)+a(21) = 72
  assert 3 + 3 + 3 + 21 + 21 + 21 == 72

  critic_terms = cfg.observations["critic"].terms
  assert "gait_phase" in critic_terms
  assert "style_joint_pos" in critic_terms
  assert "base_lin_vel" in critic_terms

  assert "push_robot" in cfg.events
  assert "imu_bias" in cfg.events
  action = cfg.actions["joint_pos"]
  assert isinstance(action, DecapJointPositionActionCfg)
  assert action.decap_enabled is True
  assert action.schedule == "exp"
  assert action.gamma == 0.99
  assert action.k == 500.0


def test_style_play_disables_dr_noise_and_decap() -> None:
  cfg = load_env_cfg(TASK_ID, play=True)
  assert cfg.observations["actor"].enable_corruption is False
  for key in (
    "push_robot",
    "base_com",
    "encoder_bias",
    "foot_friction_slide",
    "imu_bias",
    "pd_gains",
    "base_mass",
    "link_mass",
  ):
    assert key not in cfg.events
  action = cfg.actions["joint_pos"]
  assert isinstance(action, DecapJointPositionActionCfg)
  assert action.decap_enabled is False
  twist = cfg.commands["twist"]
  assert isinstance(twist, StyleVelocityCommandCfg)
  assert twist.play_mode is True
  assert twist.vx_noise == 0.0


def test_style_rl_uses_multi_critic() -> None:
  rl_cfg = load_rl_cfg(TASK_ID)
  assert isinstance(rl_cfg, RslRlOnPolicyRunnerCfg)
  assert isinstance(rl_cfg.algorithm, RslRlMultiCriticPpoAlgorithmCfg)
  assert rl_cfg.algorithm.class_name == "mjlab.rl.multi_critic:MultiCriticPPO"
  assert rl_cfg.algorithm.advantage_weights == (0.5, 0.5)
  assert rl_cfg.max_iterations == 10_000
  assert rl_cfg.save_interval == 200


def test_decap_lambda_exp_decays_from_one() -> None:
  """APEX exp: full prior at step 0, then γ^{s/k} (k in env-steps)."""
  spp = 24.0
  assert float(decap_lambda(0, 0.99, 500.0, schedule="exp")) == pytest.approx(1.0)
  assert float(decap_lambda(500, 0.99, 500.0, schedule="exp")) == pytest.approx(0.99)
  # Iter 1000 (Go2 budget): λ ≈ 0.62; cosine-hold would still be ~1.
  at_1k = float(decap_lambda(1000 * spp, 0.99, 500.0, schedule="exp"))
  assert at_1k == pytest.approx(0.99 ** (1000 * spp / 500.0), rel=1e-5)
  assert 0.60 < at_1k < 0.64
  later = float(decap_lambda(5000, 0.99, 500.0, schedule="exp"))
  assert later < 0.99


def test_decap_cosine_holds_then_weans() -> None:
  """Ablation: λ=1 through the hold, 0.5 at mid-decay, 0 after hold+decay."""
  hold, decay, spp = 500.0, 6000.0, 24.0

  def lam(step: float) -> float:
    return float(
      decap_lambda(
        step,
        schedule="cosine",
        hold_iterations=hold,
        decay_iterations=decay,
        steps_per_iteration=spp,
      )
    )

  assert lam(0.0) == pytest.approx(1.0)
  assert lam(hold * spp) == pytest.approx(1.0)
  assert lam((hold + 0.5 * decay) * spp) == pytest.approx(0.5, abs=1e-5)
  assert lam((hold + decay) * spp) == pytest.approx(0.0, abs=1e-5)
  # Early training must not have started decaying (unlike exp).
  early = 320 * spp
  assert lam(early) == pytest.approx(1.0)
  assert float(decap_lambda(early, 0.99, 500.0, schedule="exp")) < 0.87


def test_fuse_group_advantages_shape_and_weights() -> None:
  adv = torch.zeros(4, 3, 2)
  adv[..., 0] = torch.arange(12, dtype=torch.float32).reshape(4, 3)
  adv[..., 1] = -torch.arange(12, dtype=torch.float32).reshape(4, 3)
  weights = torch.tensor([0.5, 0.5])
  fused = fuse_group_advantages(adv, weights)
  assert fused.shape == (4, 3, 1)
  assert torch.isfinite(fused).all()


def test_style_exp_kernel_uses_apex_mean_over_sigma() -> None:
  """RMS 0.26 rad must stay on the gradient (old sum/σ² kernel was ~0)."""
  mean_sq = torch.tensor(0.26**2)
  value = float(_exp_kernel(mean_sq, 0.05))
  assert value == pytest.approx(float(torch.exp(torch.tensor(-(0.26**2) / 0.05))))
  assert value > 0.2
  old_sum_kernel = float(torch.exp(torch.tensor(-(14.0 * 0.26**2) / (0.05**2))))
  assert old_sum_kernel < 1e-10


def test_standing_gate_swaps_style_targets_to_default() -> None:
  """Standing is ‖v_xy‖ < 0.1; StyleCommand then returns default q*."""
  cmd_xy = torch.tensor([[0.5, 0.0], [0.0, 0.0], [0.05, 0.0]])
  standing = torch.norm(cmd_xy, dim=-1) < 0.1
  assert standing.tolist() == [False, True, True]
