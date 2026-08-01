"""Tests specific to motion tracking tasks."""

from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg, MotionLoader


@pytest.fixture(scope="module")
def tracking_task_ids() -> list[str]:
  """Get all tracking task IDs."""
  return [t for t in list_tasks() if "Tracking" in t]


@pytest.fixture(scope="module")
def g1_tracking_task_ids(tracking_task_ids: list[str]) -> list[str]:
  """Get all G1 tracking task IDs."""
  return [t for t in tracking_task_ids if "G1" in t]


def test_tracking_tasks_have_motion_command(tracking_task_ids: list[str]) -> None:
  """All tracking tasks should have a 'motion' command of type MotionCommandCfg."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "motion" in cfg.commands, f"Task {task_id} missing 'motion' command"

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} motion command is not MotionCommandCfg"
    )


def test_tracking_tasks_have_self_collision_sensor(
  tracking_task_ids: list[str],
) -> None:
  """All tracking tasks should have a self_collision sensor."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_tracking_no_state_estimation_observations() -> None:
  """No-state-estimation tasks remove observations that depend on state estimation."""
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"

  # Test both training and play modes
  for play_mode in [False, True]:
    cfg = load_env_cfg(task_id, play=play_mode)
    mode_str = "play mode" if play_mode else "training mode"

    assert "actor" in cfg.observations, (
      f"Task {task_id} ({mode_str}) missing policy observations"
    )
    actor_terms = cfg.observations["actor"].terms

    assert "motion_anchor_pos_b" not in actor_terms, (
      f"Task {task_id} ({mode_str}) has motion_anchor_pos_b in policy, "
      "expected it to be removed for no-state-estimation variant"
    )
    assert "base_lin_vel" not in actor_terms, (
      f"Task {task_id} ({mode_str}) has base_lin_vel in policy, "
      "expected it to be removed for no-state-estimation variant"
    )


def test_tracking_play_disables_rsi_randomization() -> None:
  """Tracking play tasks should disable RSI randomization."""
  tracking_tasks = [
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  ]

  for task_id in tracking_tasks:
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.pose_range == {}, (
      f"Task {task_id} (play mode) has non-empty pose_range={motion_cmd.pose_range}, "
      "expected empty dict for disabled RSI"
    )
    assert motion_cmd.velocity_range == {}, (
      f"Task {task_id} (play mode) has non-empty velocity_range={motion_cmd.velocity_range}, "
      "expected empty dict for disabled RSI"
    )


def test_tracking_play_uses_start_sampling_mode() -> None:
  """Tracking play tasks should use sampling_mode='start'."""
  tracking_tasks = [
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  ]

  for task_id in tracking_tasks:
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.sampling_mode == "start", (
      f"Task {task_id} (play mode) sampling_mode={motion_cmd.sampling_mode}, expected 'start'"
    )


def test_g1_tracking_has_correct_action_scale(g1_tracking_task_ids: list[str]) -> None:
  """G1 tracking tasks should use G1_ACTION_SCALE."""
  for task_id in g1_tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )


def test_marsdog_tracking_uses_quadruped_safety_config() -> None:
  """Marsdog tracking should cover each leg chain and reject fallen states."""
  cfg = load_env_cfg("Mjlab-Tracking-Flat-marsdog-No-State-Estimation")
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  assert motion_cmd.align_episode_length_to_motion
  assert {
    "rl_hip_link",
    "rr_hip_link",
    "fl_thigh_roll_link",
    "fr_thigh_roll_link",
  }.issubset(motion_cmd.body_names)
  assert cfg.rewards["self_collisions"].weight == -0.5
  assert cfg.rewards["motion_head_neck_joint_pos"].weight == 0.4
  assert cfg.rewards["motion_head_neck_joint_vel"].weight == 0.05
  assert cfg.terminations["anchor_pos"].params["threshold"] == 0.15
  assert cfg.terminations["ee_body_pos"].params["threshold"] == 0.15
  assert "illegal_body_contact" in cfg.terminations
  assert "nan_detection" in cfg.terminations

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  assert joint_pos_action.clip is not None
  assert joint_pos_action.clip["head_roll_joint"] == (-0.8, 0.4)


def test_motion_loader_reorders_named_archive(tmp_path: Path) -> None:
  """Name metadata should make motion archives independent of model ordering."""
  archive = tmp_path / "motion.npz"
  body_vector = np.array([[[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
  np.savez(
    archive,
    fps=np.array([50.0]),
    joint_names=np.array(["joint_b", "joint_a"]),
    body_names=np.array(["body_b", "body_a"]),
    joint_pos=np.array([[2.0, 1.0]]),
    joint_vel=np.array([[20.0, 10.0]]),
    body_pos_w=body_vector,
    body_quat_w=np.zeros((1, 2, 4)),
    body_lin_vel_w=body_vector,
    body_ang_vel_w=body_vector,
  )

  loader = MotionLoader(
    str(archive),
    torch.tensor([0, 1]),
    joint_names=("joint_a", "joint_b"),
    body_names=("body_a", "body_b"),
  )

  torch.testing.assert_close(loader.joint_pos, torch.tensor([[1.0, 2.0]]))
  torch.testing.assert_close(loader.joint_vel, torch.tensor([[10.0, 20.0]]))
  torch.testing.assert_close(loader.body_pos_w[..., 0], torch.tensor([[1.0, 2.0]]))
