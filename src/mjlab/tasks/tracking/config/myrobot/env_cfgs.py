"""myrobot flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  MYROBOT_ACTION_SCALE,
  get_myrobot_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def myrobot_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create myrobot flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_myrobot_robot_cfg()}

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

  # Relative joint position control: target = current_joint_pos + action * scale.
  # The per-step delta makes the command stream self-smoothing (an implicit
  # integrator), so the policy reacts to small attitude/IMU transients far less
  # aggressively than absolute targets. The `scale` is shrunk to ~0.3x of the
  # absolute-control scale because it now bounds a per-control-step (50 Hz)
  # increment rather than an offset from the default pose; `clip` caps the delta
  # as a safety rail against output spikes. NOTE: switching action semantics
  # requires training from scratch (old checkpoints are incompatible).
  rel_action_scale = {k: v * 0.3 for k, v in MYROBOT_ACTION_SCALE.items()}
  cfg.actions["joint_pos"] = RelativeJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=rel_action_scale,
    clip={".*": (-0.15, 0.15)},  # rad, per-step delta safety bound.
  )

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "base_link"
  motion_cmd.body_names = (
    "base_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "head_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(l|r)_foot_[12]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_yaw_link",)
  # Myrobot is lightweight and sensitive to perturbations. Use conservative DR ranges.
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
  # Kept gentler than the base task (myrobot is lightweight and push-sensitive):
  # sparser interval and smaller velocity impulses. This is a complement to the
  # compliance DR above, not a substitute for it.
  cfg.events["push_robot"] = EventTermCfg(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(3.0, 6.0),
    params={
      "velocity_range": {
        "x": (-0.2, 0.2),
        "y": (-0.2, 0.2),
        "roll": (-0.2, 0.2),
        "pitch": (-0.2, 0.2),
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
    actor_terms[_term].delay_max_lag = 2

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_link",
    "right_elbow_link",
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
    motion_cmd.sampling_mode = "start"

  return cfg
