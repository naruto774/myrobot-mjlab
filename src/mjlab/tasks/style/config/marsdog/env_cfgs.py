"""Marsdog flat style-imitation environment."""

from __future__ import annotations

from mjlab.asset_zoo.robots import MARSDOG_ACTION_SCALE, get_marsdog_robot_cfg
from mjlab.asset_zoo.robots.marsdog.marsdog_constants import MARSDOG_JOINT_NAMES
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.style.mdp.actions import DecapJointPositionActionCfg
from mjlab.tasks.style.mdp.commands import StyleVelocityCommandCfg
from mjlab.tasks.style.style_env_cfg import make_style_env_cfg

LEG_JOINT_NAMES = (
  r".*(fr|fl|rr|rl).*(hip|thigh|calf).*",
  r".*(fr|fl).*tarsus.*",
)
WAIST_JOINT_NAMES = (
  "waist_roll_joint",
  "waist_pitch_joint",
  "waist_yaw_joint",
)
HEAD_JOINT_NAMES = (
  "neck_pitch_joint",
  "head_roll_joint",
  "head_yaw_joint",
  "head_pitch_joint",
)
FOOT_SITE_NAMES = ("fr", "fl", "rr", "rl")
FOOT_GEOM_NAMES = tuple(f"{name}_foot_collision" for name in FOOT_SITE_NAMES)


def marsdog_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Marsdog flat style configuration."""
  cfg = make_style_env_cfg(joint_names=MARSDOG_JOINT_NAMES)
  cfg.scene.entities = {"robot": get_marsdog_robot_cfg()}

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
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
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, DecapJointPositionActionCfg)
  joint_pos_action.scale = MARSDOG_ACTION_SCALE
  joint_pos_action.actuator_names = (".*",)
  joint_pos_action.schedule = "exp"
  joint_pos_action.gamma = 0.99
  joint_pos_action.k = 500.0

  cfg.rewards["imitate_joint_legs"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=LEG_JOINT_NAMES
  )
  cfg.rewards["imitate_joint_waist"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=WAIST_JOINT_NAMES
  )
  cfg.rewards["imitate_joint_head"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=HEAD_JOINT_NAMES
  )
  cfg.rewards["feet_slip"].params["asset_cfg"] = SceneEntityCfg(
    "robot", site_names=FOOT_SITE_NAMES
  )
  cfg.rewards["ang_vel_xy"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("base_link",)
  )

  cfg.events["foot_friction_slide"].params["asset_cfg"] = SceneEntityCfg(
    "robot", geom_names=FOOT_GEOM_NAMES
  )

  cfg.viewer.body_name = "base_link"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    for term_name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
      term = cfg.observations["actor"].terms[term_name]
      term.delay_min_lag = 0
      term.delay_max_lag = 0
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
      cfg.events.pop(key, None)
    joint_pos_action.decap_enabled = False
    twist = cfg.commands["twist"]
    assert isinstance(twist, StyleVelocityCommandCfg)
    twist.play_mode = True
    twist.rel_standing_envs = 0.0
    twist.vx_noise = 0.0

  return cfg
