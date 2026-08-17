"""Style command: cyclic expert pose plus Go2-style twist velocity."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
# 数学工具
from mjlab.utils.lab_api.math import (
  quat_apply_inverse, # 逆向应用四元数旋转
  quat_from_euler_xyz, # 从欧拉角转换为四元数
  quat_mul, # 四元数乘法
  sample_uniform, # 均匀采样
  yaw_quat, # 计算四元数的yaw角
)

if TYPE_CHECKING:
  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class StyleCommand(CommandTerm):
  """Cyclic style reference loaded from a tiled mjlab motion NPZ.

  Time index walks linearly through the clip (``index % N`` as a wrap
  guard). Critic phase uses ``cycle_len`` so the same pose always maps to
  the same gait clock after tiling. Standing (‖v_cmd,xy‖ < threshold)
  swaps q* / feet* / tilt* to the default pose.
  """

  cfg: StyleCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: StyleCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.body_indexes = torch.arange(
      len(self.robot.body_names), device=self.device, dtype=torch.long
    )
    self.motion = MotionLoader(
      self.cfg.motion_file,
      self.body_indexes,
      self.robot.joint_names,
      self.robot.body_names,
      device=self.device,
    )
    expected_fps = 1.0 / env.step_dt
    if not math.isclose(self.motion.fps, expected_fps, rel_tol=1.0e-5):
      raise ValueError(
        f"Motion FPS ({self.motion.fps:g}) must match the control frequency "
        f"({expected_fps:g} Hz)."
      )
    self.cmd_vx = self.motion.cmd_vx
    self.cycle_len = max(int(self.motion.cycle_len), 1)
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    self.base_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
    self.foot_body_ids = torch.tensor(
      [self.robot.body_names.index(name) for name in self.cfg.foot_body_names],
      dtype=torch.long,
      device=self.device,
    )

    base_pos = self.motion.body_pos_w[:, self.base_body_index]
    base_quat = self.motion.body_quat_w[:, self.base_body_index]
    foot_pos = self.motion.body_pos_w[:, self.foot_body_ids]
    rel = foot_pos - base_pos[:, None, :]
    yaw_q = yaw_quat(base_quat)
    n_feet = len(self.cfg.foot_body_names)
    self._feet_yaw_b = quat_apply_inverse(yaw_q[:, None, :].expand(-1, n_feet, -1), rel)
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(
      self.motion.time_step_total, 3
    )
    gravity_b = quat_apply_inverse(base_quat, gravity_w)
    self._g_xy_star = gravity_b[:, :2]

    default_joint = self.robot.data.default_joint_pos
    assert default_joint is not None
    self._standing_joint_pos = default_joint[0].clone()
    self._standing_feet_b = self._feet_yaw_b[0].clone()
    self._standing_g_xy = torch.zeros(2, device=self.device)

    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["gait_phase"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.gait_phase

  @property
  def gait_phase(self) -> torch.Tensor:
    phase = (self.time_steps % self.cycle_len).float() / float(self.cycle_len)
    return torch.stack(
      [torch.sin(2.0 * math.pi * phase), torch.cos(2.0 * math.pi * phase)], dim=-1
    )

  @property
  def is_standing(self) -> torch.Tensor:
    twist = cast(
      UniformVelocityCommand,
      self._env.command_manager.get_term(self.cfg.twist_command_name),
    )
    cmd = twist.command
    standing = torch.norm(cmd[:, :2], dim=-1) < self.cfg.standing_threshold
    return standing | twist.is_standing_env

  @property
  def style_joint_pos(self) -> torch.Tensor:
    q_star = self.motion.joint_pos[self.time_steps]
    standing = self.is_standing
    if torch.any(standing):
      q_star = q_star.clone()
      q_star[standing] = self._standing_joint_pos
    return q_star

  @property
  def style_feet_yaw_b(self) -> torch.Tensor:
    feet = self._feet_yaw_b[self.time_steps]
    standing = self.is_standing
    if torch.any(standing):
      feet = feet.clone()
      feet[standing] = self._standing_feet_b
    return feet

  @property
  def style_g_xy(self) -> torch.Tensor:
    g_xy = self._g_xy_star[self.time_steps]
    standing = self.is_standing
    if torch.any(standing):
      g_xy = g_xy.clone()
      g_xy[standing] = self._standing_g_xy
    return g_xy

  @property
  def style_root_height(self) -> torch.Tensor:
    height = self.motion.body_pos_w[self.time_steps, self.base_body_index, 2]
    standing = self.is_standing
    if torch.any(standing):
      default_root = self.robot.data.default_root_state
      assert default_root is not None
      height = height.clone()
      height[standing] = default_root[standing, 2]
    return height

  def _update_metrics(self) -> None:
    self.metrics["error_joint_pos"] = torch.norm(
      self.style_joint_pos - self.robot.data.joint_pos, dim=-1
    )
    phase = (self.time_steps % self.cycle_len).float() / float(self.cycle_len)
    self.metrics["gait_phase"] = phase

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.time_steps[env_ids] = torch.randint(
      0, self.motion.time_step_total, (len(env_ids),), device=self.device
    )
    self._apply_rsi(env_ids)

  def _apply_rsi(self, env_ids: torch.Tensor) -> None:
    """Random start: expert pose scaled by U(0.5, 1.5), height from clip."""
    q_star = self.motion.joint_pos[self.time_steps[env_ids]].clone()
    scale = sample_uniform(
      self.cfg.joint_scale_range[0],
      self.cfg.joint_scale_range[1],
      q_star.shape,
      device=self.device,
    )
    joint_pos = q_star * scale
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    joint_vel = self.motion.joint_vel[self.time_steps[env_ids]]

    root_pos = self.motion.body_pos_w[self.time_steps[env_ids], self.base_body_index]
    root_pos = root_pos.clone()
    root_pos[:, :2] = self._env.scene.env_origins[env_ids, :2]
    pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pose_range = torch.tensor(
      [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in pose_keys],
      device=self.device,
    )
    pose_samples = sample_uniform(
      pose_range[:, 0], pose_range[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[:, :2] += pose_samples[:, :2]
    root_ori = self.motion.body_quat_w[self.time_steps[env_ids], self.base_body_index]
    delta_ori = quat_from_euler_xyz(
      pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    root_ori = quat_mul(delta_ori, root_ori)

    vel_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    vel_range = torch.tensor(
      [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in vel_keys],
      device=self.device,
    )
    vel_samples = sample_uniform(
      vel_range[:, 0], vel_range[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel = (
      self.motion.body_lin_vel_w[self.time_steps[env_ids], self.base_body_index]
      + vel_samples[:, :3]
    )
    root_ang_vel = (
      self.motion.body_ang_vel_w[self.time_steps[env_ids], self.base_body_index]
      + vel_samples[:, 3:]
    )

    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _update_command(self) -> None:
    self.time_steps += 1
    self.time_steps %= self.motion.time_step_total


@dataclass(kw_only=True)
class StyleCommandCfg(CommandTermCfg):
  motion_file: str
  entity_name: str = "robot"
  anchor_body_name: str = "base_link"
  foot_body_names: tuple[str, ...] = (
    "fl_foot_link",
    "fr_foot_link",
    "rl_foot_link",
    "rr_foot_link",
  )
  twist_command_name: str = "twist"
  standing_threshold: float = 0.1
  joint_scale_range: tuple[float, float] = (0.5, 1.5)
  pose_range: dict[str, tuple[float, float]] = field(
    default_factory=lambda: {
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "yaw": (-3.14, 3.14),
    }
  )
  velocity_range: dict[str, tuple[float, float]] = field(
    default_factory=lambda: {
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "z": (-0.5, 0.5),
      "roll": (-0.5, 0.5),
      "pitch": (-0.5, 0.5),
      "yaw": (-0.5, 0.5),
    }
  )

  def build(self, env: ManagerBasedRlEnv) -> StyleCommand:
    return StyleCommand(self, env)


class StyleVelocityCommand(UniformVelocityCommand):
  """Go2-style twist: vx around expert cmd_vx, vy = 0, random ωz."""

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    cfg = cast(StyleVelocityCommandCfg, self.cfg)
    style = cast(
      StyleCommand, self._env.command_manager.get_term(cfg.style_command_name)
    )
    n = len(env_ids)
    r = torch.empty(n, device=self.device)
    if cfg.play_mode:
      vx = torch.full((n,), style.cmd_vx, device=self.device)
    else:
      noise = r.uniform_(-cfg.vx_noise, cfg.vx_noise)
      vx = (style.cmd_vx + noise).clamp(cfg.vx_clip[0], cfg.vx_clip[1])
    self.vel_command_b[env_ids, 0] = vx
    self.vel_command_b[env_ids, 1] = 0.0
    self.vel_command_b[env_ids, 2] = r.uniform_(*cfg.ranges.ang_vel_z)
    self.is_heading_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
    if cfg.play_mode:
      self.is_standing_env[env_ids] = False
    else:
      self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= cfg.rel_standing_envs

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    del on_change, request_action
    from viser import Icon

    wz_max = abs(self.cfg.ranges.ang_vel_z[1])
    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      standing = server.gui.add_checkbox("Stand", initial_value=False)
      wz = server.gui.add_slider(
        "ang_vel_z",
        min=-wz_max,
        max=wz_max,
        step=0.05,
        initial_value=0.0,
      )
      zero_btn = server.gui.add_button("Zero yaw", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        wz.value = 0.0

    self._joystick_enabled = enabled
    self._joystick_sliders = [wz]
    self._joystick_standing = standing
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    CommandTerm.compute(self, dt)
    if self._joystick_enabled is None or not self._joystick_enabled.value:
      return
    assert self._joystick_get_env_idx is not None
    idx = self._joystick_get_env_idx()
    style = cast(
      StyleCommand,
      self._env.command_manager.get_term(
        cast(StyleVelocityCommandCfg, self.cfg).style_command_name
      ),
    )
    standing = bool(
      getattr(self, "_joystick_standing", None) and self._joystick_standing.value
    )
    if standing:
      self.vel_command_b[idx] = 0.0
      self.is_standing_env[idx] = True
    else:
      self.vel_command_b[idx, 0] = style.cmd_vx
      self.vel_command_b[idx, 1] = 0.0
      self.vel_command_b[idx, 2] = self._joystick_sliders[0].value
      self.is_standing_env[idx] = False


@dataclass(kw_only=True)
class StyleVelocityCommandCfg(UniformVelocityCommandCfg):
  style_command_name: str = "style"
  vx_noise: float = 0.2
  vx_clip: tuple[float, float] = (0.2, 0.8)
  play_mode: bool = False

  def build(self, env: ManagerBasedRlEnv) -> StyleVelocityCommand:
    return StyleVelocityCommand(self, env)
