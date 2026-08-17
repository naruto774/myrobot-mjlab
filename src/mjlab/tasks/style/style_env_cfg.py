"""Style-imitation task configuration factory."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.style import mdp as style_mdp
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from .mdp.commands import StyleCommandCfg, StyleVelocityCommandCfg


def make_style_env_cfg(
  joint_names: tuple[str, ...] = (".*",),
) -> ManagerBasedRlEnvCfg:
  """Create the base style task (flat, 21-DoF command layout)."""
  joint_cfg = SceneEntityCfg("robot", joint_names=joint_names)
  base_cfg = SceneEntityCfg("robot", body_names=("base_link",))

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=style_mdp.base_ang_vel_biased,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      scale=0.25,
      delay_min_lag=0,
      delay_max_lag=1,
    ),
    "projected_gravity": ObservationTermCfg(
      func=style_mdp.projected_gravity_biased,
      noise=Unoise(n_min=-0.1, n_max=0.1),
      delay_min_lag=0,
      delay_max_lag=1,
    ),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
      scale=(2.0, 2.0, 0.25),
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": joint_cfg, "biased": True},
      noise=Unoise(n_min=-0.02, n_max=0.02),
      delay_min_lag=0,
      delay_max_lag=1,
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": joint_cfg},
      noise=Unoise(n_min=-1.5, n_max=1.5),
      scale=0.05,
      delay_min_lag=0,
      delay_max_lag=1,
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }

  critic_terms = {
    **{
      key: ObservationTermCfg(
        func=term.func,
        params=dict(term.params),
        scale=term.scale,
      )
      for key, term in actor_terms.items()
    },
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      scale=0.25,
    ),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": joint_cfg, "biased": False},
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": joint_cfg},
      scale=0.05,
    ),
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "gait_phase": ObservationTermCfg(
      func=style_mdp.gait_phase, params={"command_name": "style"}
    ),
    "style_joint_pos": ObservationTermCfg(
      func=style_mdp.style_joint_pos,
      params={"command_name": "style", "asset_cfg": joint_cfg},
    ),
    "style_feet_yaw_b": ObservationTermCfg(
      func=style_mdp.style_feet_yaw_b, params={"command_name": "style"}
    ),
    "style_g_xy": ObservationTermCfg(
      func=style_mdp.style_g_xy, params={"command_name": "style"}
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": style_mdp.DecapJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
      command_name="style",
      decap_enabled=True,
      schedule="exp",
      gamma=0.99,
      k=500.0,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "style": StyleCommandCfg(
      motion_file="",
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=False,
    ),
    "twist": StyleVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(5.0, 5.0),
      rel_standing_envs=0.1,
      rel_heading_envs=0.0,
      rel_forward_envs=0.0,
      heading_command=False,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.2, 0.8),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-1.5, 1.5),
      ),
      vx_noise=0.2,
      vx_clip=(0.2, 0.8),
      play_mode=False,
    ),
  }

  events = {
    "foot_friction_slide": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        "axes": [0],
        "ranges": (0.3, 1.25),
        "shared_random": True,
      },
    ),
    "base_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "operation": "add",
        "ranges": (-0.2, 0.2),
      },
    ),
    "link_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=r"^(?!base_link$).+"),
        "operation": "scale",
        "ranges": (0.9, 1.1),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "operation": "add",
        "ranges": {0: (-0.03, 0.03), 1: (-0.03, 0.03), 2: (-0.03, 0.03)},
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.035, 0.035)},
    ),
    "pd_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "imu_bias": EventTermCfg(
      mode="reset",
      func=dr.imu_bias,
      params={"ori_range": (-0.087, 0.087), "gyro_range": (-0.05, 0.05)},
    ),
    "push_robot": EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(4.0, 4.0),
      params={
        "velocity_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
          "yaw": (-0.4, 0.4),
        }
      },
    ),
  }

  # Style std is APEX σ in exp(-mean((x-x*)²) / σ), not a Gaussian σ² bandwidth.
  rewards = {
    "imitate_joint_legs": RewardTermCfg(
      func=style_mdp.imitate_joint_pos_exp,
      weight=3.5,
      group=0,
      params={"command_name": "style", "std": 0.05, "asset_cfg": joint_cfg},
    ),
    "imitate_joint_waist": RewardTermCfg(
      func=style_mdp.imitate_joint_pos_exp,
      weight=1.2,
      group=0,
      params={"command_name": "style", "std": 0.05, "asset_cfg": joint_cfg},
    ),
    "imitate_joint_head": RewardTermCfg(
      func=style_mdp.imitate_joint_pos_exp,
      weight=0.4,
      group=0,
      params={"command_name": "style", "std": 0.05, "asset_cfg": joint_cfg},
    ),
    "imitate_feet": RewardTermCfg(
      func=style_mdp.imitate_feet_yaw_exp,
      weight=2.5,
      group=0,
      params={"command_name": "style", "std": 0.08},
    ),
    "imitate_tilt": RewardTermCfg(
      func=style_mdp.imitate_tilt_exp,
      weight=0.5,
      group=0,
      params={"command_name": "style", "std": 0.1, "asset_cfg": base_cfg},
    ),
    "track_linear_velocity": RewardTermCfg(
      func=vel_mdp.track_linear_velocity,
      weight=2.0,
      group=1,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=vel_mdp.track_angular_velocity,
      weight=1.5,
      group=1,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "torques": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-1.0e-5, group=1),
    "dof_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7, group=1),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01, group=1),
    "collision": RewardTermCfg(
      func=vel_mdp.self_collision_cost,
      weight=-1.0,
      group=1,
      params={"sensor_name": "self_collision"},
    ),
    "feet_slip": RewardTermCfg(
      func=vel_mdp.feet_slip,
      weight=-0.04,
      group=1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),
      },
    ),
    "dof_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits, weight=-1.0, group=1
    ),
    "imitation_height_penalty": RewardTermCfg(
      func=style_mdp.imitation_height_penalty,
      weight=-10.0,
      group=1,
      params={"command_name": "style"},
    ),
    "ang_vel_xy": RewardTermCfg(
      func=vel_mdp.body_angular_velocity_penalty,
      weight=-0.05,
      group=1,
      params={"asset_cfg": base_cfg},
    ),
    "soft_landing": RewardTermCfg(
      func=vel_mdp.soft_landing,
      weight=-2.5e-3,
      group=1,
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
    ),
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane", terrain_generator=None),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=1.5,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=None,
      njmax=300,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
        cone="elliptic",
        impratio=10,
      ),
    ),
    decimation=4,
    episode_length_s=19.0,
  )
