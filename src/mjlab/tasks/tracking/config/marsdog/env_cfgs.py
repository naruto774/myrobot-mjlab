"""marsdog flat tracking environment configurations."""

import mujoco

from mjlab.asset_zoo.robots import (
  MARSDOG_ACTION_SCALE,
  get_marsdog_robot_cfg,
)
from mjlab.asset_zoo.robots.marsdog.marsdog_constants import MARSDOG_JOINT_NAMES
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def _add_marsdog_step(spec: mujoco.MjSpec) -> None:
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


def marsdog_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create marsdog flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()
  cfg.sim.nconmax = None
  cfg.sim.njmax = 2048
  cfg.sim.contact_sensor_maxmatch = 192
  cfg.scene.spec_fn = _add_marsdog_step
  cfg.scene.entities = {"robot": get_marsdog_robot_cfg()}
  # The reference motion uses a single fixed step at a world-space location, but
  # the step geom lives in the shared model and is not replicated per env. With a
  # nonzero env_spacing the grid layout offsets each env's robot/reference motion
  # away from that single step, so only one env actually lands on it. Collapse the
  # env grid to the origin (all envs overlap) so every env shares the one step.
  # mjwarp simulates each env in an isolated world, so overlapping robots never
  # collide across envs.
  cfg.scene.env_spacing = 0.0

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = MARSDOG_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "base_link"
  motion_cmd.body_names = (
    "base_link",  # anchor
    "waist_yaw_link",
    "waist_pitch_link",
    "neck_pitch_link",
    "head_pitch_link",
    "rl_thigh_link",
    "rl_calf_link",
    "rl_foot_link",
    "rr_thigh_link",
    "rr_calf_link",
    "rr_foot_link",
    "fl_hip_pitch_link",
    "fl_calf_link",
    "fl_foot_link",
    "fr_hip_pitch_link",
    "fr_calf_link",
    "fr_foot_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(fl|fr|rl|rr)_foot_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_yaw_link",)
  # marsdog is lightweight and sensitive to perturbations. Use conservative DR ranges.
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.008, 0.008),
    1: (-0.010, 0.010),
    2: (-0.006, 0.006),
  }
  cfg.events["encoder_bias"].params["bias_range"] = (-0.005, 0.005)
  cfg.events["foot_friction"].params["ranges"] = (0.45, 0.90)

  # Stiffness/damping randomization. The training model is a perfectly rigid
  # hinge that tracks joint targets exactly; the real robot's structural
  # compliance/backlash makes the achieved body attitude deflect more under load
  # (e.g. an extra few degrees of pitch on a step). Scaling kp DOWN makes the
  # simulated joint sag more under contact/gravity load, reproducing this
  # motion-correlated deflection so the policy learns it is normal and responds
  # smoothly instead of over-correcting. kd is jittered both ways. Conservative
  # ranges to start (only soften kp); widen kp_range toward 0.7 if it stays
  # stable.
  cfg.events["pd_gains"] = EventTermCfg(
    func=dr.pd_gains,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.8, 1.0),
      "kd_range": (0.8, 1.2),
      "operation": "scale",
    },
  )

  # Small, sparse base pushes for general transient-disturbance robustness.
  # Kept gentler than the base task (marsdog is lightweight and push-sensitive):
  # sparser interval and smaller velocity impulses. This is a complement to the
  # compliance DR above, not a substitute for it.
  cfg.events["push_robot"] = EventTermCfg(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(3.0, 6.0),
    params={
      "velocity_range": {
        "x": (-0.01, 0.01),
        "y": (-0.01, 0.01),
        "roll": (-0.01, 0.01),
        "pitch": (-0.01, 0.01),
      }
    },
  )

  # Persistent IMU bias DR (resampled every episode). The real servo robot has a
  # frequently-drifting IMU; a constant per-episode roll/pitch offset (~5 deg) plus
  # a gyro zero-bias forces the actor to be robust to this low-frequency error,
  # which the per-step observation noise alone cannot model.
  cfg.events["imu_bias"] = EventTermCfg(
    func=dr.imu_bias,
    mode="reset",
    params={
      "ori_range": (-0.09, 0.09),  # (-0.087, 0.087), rad (~5 deg) on roll and pitch.
      "gyro_range": (-0.05, 0.05),  # (-0.05, 0.05) rad/s per axis.
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
    actor_terms[_term].delay_max_lag = 1

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "fl_foot_link",
    "fr_foot_link",
    "rl_foot_link",
    "rr_foot_link",
  )
  cfg.rewards["joint_limit"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=MARSDOG_JOINT_NAMES
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
