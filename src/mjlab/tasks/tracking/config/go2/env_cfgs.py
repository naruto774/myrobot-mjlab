"""go2 flat tracking environment configurations."""

import mujoco

from mjlab.asset_zoo.robots import (
  GO2_ACTION_SCALE,
  get_go2_robot_cfg,
)
from mjlab.asset_zoo.robots.unitree_go2.go2_constants import GO2_JOINT_NAMES
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from . import rewards as go2_rewards


def _add_go2_step(spec: mujoco.MjSpec) -> None:
  """Add the fixed step used by the Marsdog reference motion."""
  spec.add_material(name="step", rgba=(0.35, 0.35, 0.35, 1.0))
  spec.worldbody.add_body(name="step").add_geom(
    name="step1",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.8, 0.40, 0.15),
    pos=(0.0, -1.9, 0.15),
    material="step",
    friction=(0.8, 0.005, 0.0001),
  )


def go2_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create go2 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()
  cfg.sim.nconmax = None
  cfg.sim.njmax = 600
  cfg.sim.contact_sensor_maxmatch = 192
  # Entity specs are attached into the scene, so XML <option> fields do not
  # propagate automatically. Preserve the Go2 contact solver settings explicitly.
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.mujoco.impratio = 100.0
  # cfg.scene.spec_fn = _add_go2_step
  cfg.scene.entities = {"robot": get_go2_robot_cfg()}
  # The reference motion uses a single fixed step at a world-space location, but
  # the step geom lives in the shared model and is not replicated per env. With a
  # nonzero env_spacing the grid layout offsets each env's robot/reference motion
  # away from that single step, so only one env actually lands on it. Collapse the
  # env grid to the origin (all envs overlap) so every env shares the one step.
  # mjwarp simulates each env in an isolated world, so overlapping robots never
  # collide across envs.
  cfg.scene.env_spacing = 0.0

  foot_names = ("FL", "FR", "RL", "RR")
  feet_site_names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
  feet_body_names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
  leg_body_names = (
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
  )
  leg_joint_names = (r".*(FL|FR|RL|RR).*(hip|thigh|calf).*",)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=tuple(f"{name}_foot_collision" for name in foot_names),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  illegal_body_contact_cfg = ContactSensorCfg(
    name="illegal_body_contact",
    primary=ContactMatch(
      mode="body",
      pattern=(
        "base_link",
        "FL_thigh",
        "FR_thigh",
        "RL_thigh",
        "RR_thigh",
      ),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (
    feet_ground_cfg,
    self_collision_cfg,
    illegal_body_contact_cfg,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO2_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "base_link"
  motion_cmd.sampling_mode = "start"
  motion_cmd.align_episode_length_to_motion = True
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.joint_position_range = (0.0, 0.0)
  # Expert clip: 2002 frames @ 30 Hz → T = 2002/30 ≈ 66.7 s, then resampled
  # to the 50 Hz control rate. align_episode_length_to_motion overwrites this
  # with the exact NPZ duration (time_step_total * step_dt) at env build time.
  cfg.episode_length_s = 66.7

  motion_cmd.body_names = (
    "base_link",  # anchor
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
  )

  # Go2 tracking reward layout:
  # - base/root terms keep the body height and attitude aligned with the motion;
  # - foot terms dominate spatial imitation because contacts define quadruped gait;
  # - leg joint terms preserve the expert kinematic phase without constraining
  cfg.rewards["motion_global_root_pos"] = RewardTermCfg(
    func=go2_rewards.motion_anchor_height_error_exp,
    weight=0.8,
    params={"command_name": "motion", "std": 0.05},
  )
  cfg.rewards["motion_global_root_ori"].weight = 0.8
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.35
  cfg.rewards["motion_body_pos"].weight = 1.2
  cfg.rewards["motion_body_pos"].params = {
    "command_name": "motion",
    "std": 0.08,
    "body_names": feet_body_names,
  }
  cfg.rewards["motion_body_ori"].weight = 0.2
  cfg.rewards["motion_body_ori"].params = {
    "command_name": "motion",
    "std": 0.6,
    "body_names": leg_body_names,
  }
  cfg.rewards["motion_body_lin_vel"].weight = 0.4
  cfg.rewards["motion_body_lin_vel"].params = {
    "command_name": "motion",
    "std": 1.0,
    "body_names": feet_body_names,
  }
  cfg.rewards["motion_body_ang_vel"].weight = 0.1
  cfg.rewards["motion_body_ang_vel"].params = {
    "command_name": "motion",
    "std": 3.14,
    "body_names": ("base_link",),
  }
  cfg.rewards["motion_joint_pos"] = RewardTermCfg(
    func=go2_rewards.motion_joint_position_error_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 0.35,
      "asset_cfg": SceneEntityCfg("robot", joint_names=leg_joint_names),
    },
  )
  cfg.rewards["motion_joint_vel"] = RewardTermCfg(
    func=go2_rewards.motion_joint_velocity_error_exp,
    weight=0.2,
    params={
      "command_name": "motion",
      "std": 5.0,
      "asset_cfg": SceneEntityCfg("robot", joint_names=leg_joint_names),
    },
  )

  cfg.rewards["feet_slip"] = RewardTermCfg(
    func=go2_rewards.feet_slip_penalty,
    weight=-0.15,
    params={
      "sensor_name": feet_ground_cfg.name,
      "asset_cfg": SceneEntityCfg("robot", site_names=feet_site_names),
    },
  )
  cfg.rewards["soft_landing"] = RewardTermCfg(
    func=go2_rewards.soft_landing_penalty,
    weight=-1.0e-4,
    params={"sensor_name": feet_ground_cfg.name, "force_threshold": 20.0},
  )
  cfg.rewards["action_rate_l2"].weight = -0.05
  # Treat self-contact as a soft safety objective, not a reward-scale cliff.
  cfg.rewards["self_collisions"].weight = -0.5

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(FL|FR|RL|RR)_foot_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  # Conservative per-environment domain randomization for Go2. Model parameters
  # are sampled once at startup, providing a stationary plant within each rollout.
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.01, 0.01),
    1: (-0.005, 0.005),
    2: (-0.01, 0.01),
  }
  cfg.events["encoder_bias"].params["bias_range"] = (-0.01, 0.01)
  cfg.events["foot_friction"].params["ranges"] = (0.45, 1.10)

  # Scale base mass and inertia together by exp(2 * alpha), which preserves a
  # physically valid pseudo-inertia. alpha ±0.025 gives approximately ±5%.
  cfg.events["base_inertia"] = EventTermCfg(
    func=dr.pseudo_inertia,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "alpha_range": (-0.025, 0.025),
    },
  )

  # Account for gain identification error and structural compliance.
  cfg.events["pd_gains"] = EventTermCfg(
    func=dr.pd_gains,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.85, 1.15),
      "kd_range": (0.80, 1.20),
      "operation": "scale",
    },
  )

  # Sparse velocity kicks train recovery without dominating motion imitation.
  cfg.events["push_robot"] = EventTermCfg(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(4.0, 8.0),
    params={
      "velocity_range": {
        "x": (-0.15, 0.15),
        "y": (-0.10, 0.10),
        "roll": (-0.10, 0.10),
        "pitch": (-0.10, 0.10),
        "yaw": (-0.15, 0.15),
      }
    },
  )

  # Persistent IMU bias is resampled every episode. The critic and task objectives
  # retain privileged, unbiased state while only the actor sees this corruption.
  cfg.events["imu_bias"] = EventTermCfg(
    func=dr.imu_bias,
    mode="reset",
    params={
      "ori_range": (-0.035, 0.035),
      "gyro_range": (-0.02, 0.02),
    },
  )

  # Route the actor's orientation/gyro observations through the biased variants so
  # the IMU bias is actually seen by the policy. The critic keeps the clean readings
  # (asymmetric actor-critic), and rewards/terminations use the true state.
  actor_terms = cfg.observations["actor"].terms
  actor_terms["motion_anchor_ori_b"].func = mdp.motion_anchor_ori_b_biased
  actor_terms["base_ang_vel"].func = mdp.base_ang_vel_biased

  # Random observation delay on the proprioceptive/IMU channels. Control runs at
  # 50 Hz (decimation 4 x 5 ms), so 1 step = 20 ms. A 0-2 step random lag
  # (0-40 ms) makes the policy stop over-trusting single-frame feedback and learn
  # to act on the recent trend, which suppresses high-frequency reactions to
  # backlash-induced jitter. Delay does NOT change observation dimensionality, so
  # the network architecture is unchanged. Kept modest on base_ang_vel since gyro
  # is the key fall-detection signal. Delaying observations in the closed loop is
  # the practical equivalent of action delay (which the framework lacks natively).
  _DELAYED_OBS_TERMS = (
    "base_ang_vel",
    "motion_anchor_ori_b",
    "joint_pos",
    "joint_vel",
  )
  for _term in _DELAYED_OBS_TERMS:
    actor_terms[_term].delay_min_lag = 0
    actor_terms[_term].delay_max_lag = 2

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
  )
  cfg.terminations["anchor_pos"].params["threshold"] = 0.15
  cfg.terminations["anchor_ori"].func = mdp.bad_anchor_ori_angle
  cfg.terminations["anchor_ori"].params["threshold"] = 1.0472  # 60 degrees.
  cfg.terminations["ee_body_pos"].params["threshold"] = 0.15
  cfg.terminations["illegal_body_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={
      "sensor_name": illegal_body_contact_cfg.name,
      "force_threshold": 10.0,
    },
  )
  cfg.terminations["nan_detection"] = TerminationTermCfg(func=mdp.nan_detection)
  cfg.rewards["joint_limit"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=GO2_JOINT_NAMES
  )

  cfg.viewer.body_name = "base_link"

  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    # Disable all training-time randomization and disturbances for deterministic eval.
    cfg.events.pop("push_robot", None)
    cfg.events.pop("base_com", None)
    cfg.events.pop("encoder_bias", None)
    cfg.events.pop("foot_friction", None)
    cfg.events.pop("imu_bias", None)
    cfg.events.pop("base_inertia", None)
    cfg.events.pop("pd_gains", None)
    # Observation delay is not gated by enable_corruption, so zero it explicitly
    # for deterministic eval.
    for _term in _DELAYED_OBS_TERMS:
      if _term in cfg.observations["actor"].terms:
        cfg.observations["actor"].terms[_term].delay_min_lag = 0
        cfg.observations["actor"].terms[_term].delay_max_lag = 0
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"

  return cfg
