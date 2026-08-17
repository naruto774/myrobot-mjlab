from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
# 数学工具
from mjlab.utils.lab_api.math import (
  matrix_from_quat, # 从四元数计算旋转矩阵
  quat_apply, # 应用四元数旋转
  wrap_to_pi, # 将角度限制在-π到π之间
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)
    self.is_world_env = torch.zeros_like(self.is_heading_env)
    self.is_forward_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  # 命令属性
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  # 更新指标
  def _update_metrics(self) -> None: 
    # 计算最大命令时间步长
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    # 计算线性速度误差
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    # 计算角速度误差
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  # 重新采样命令
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # 均匀采样命令
    r = torch.empty(len(env_ids), device=self.device)
    # 采样线性速度
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    # 如果启用了heading命令
    if self.cfg.heading_command:
      # 确保heading范围不为空
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      # 计算heading误差
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
      # 计算站立误差
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    # Randomly assign world-frame envs.
    # 随机分配世界帧环境
    self.is_world_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_world_envs
    # 复制采样速度作为世界帧参考
    # Copy sampled velocities as world-frame reference for world envs.
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    # Forward-only envs: positive lin_vel_x, zero lateral and angular.
    # 正向帧环境: 正线性速度, 零侧向和角速度
    self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
    # 正向帧环境索引
    fwd_ids = env_ids[self.is_forward_env[env_ids]]
    if len(fwd_ids) > 0:
      # 设置正向帧环境线性速度
      self.vel_command_b[fwd_ids, 0] = (
        self.vel_command_b[fwd_ids, 0].abs().clamp(min=0.3)
      )
      # 设置正向帧环境侧向速度
      self.vel_command_b[fwd_ids, 1] = 0.0
      self.vel_command_b[fwd_ids, 2] = 0.0

    # 初始化速度掩码
    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    # 初始化速度环境索引
    init_vel_env_ids = env_ids[init_vel_mask]
    # 如果初始化速度环境索引不为空
    if len(init_vel_env_ids) > 0:
      # 获取初始化速度环境根位置
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      # 获取初始化速度环境根四元数
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      # 获取初始化速度环境根线性速度
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      # 设置初始化速度环境根线性速度
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  # 更新命令
  def _update_command(self) -> None:
    # 如果启用了heading命令
    if self.cfg.heading_command:
      # 计算heading误差
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      # 获取heading环境索引
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      # 设置heading环境角速度
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    # World-frame envs: rotate world-frame linear vel into body frame.
    # 如果世界帧环境不为空
    if self.is_world_env.any():
      # 获取世界帧环境索引
      w_ids = self.is_world_env.nonzero(as_tuple=False).flatten()
      # 获取世界帧环境heading
      heading = self.robot.data.heading_w[w_ids]
      # 计算cos和sin
      cos_h = torch.cos(heading)
      sin_h = torch.sin(heading)
      # 获取世界帧环境线性速度
      vx_w = self.vel_command_w[w_ids, 0]
      # 获取世界帧环境侧向速度
      vy_w = self.vel_command_w[w_ids, 1]
      # 设置世界帧环境线性速度
      self.vel_command_b[w_ids, 0] = cos_h * vx_w + sin_h * vy_w
      self.vel_command_b[w_ids, 1] = -sin_h * vx_w + cos_h * vy_w

    # 站立环境索引
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    # 设置站立环境线性速度
    self.vel_command_b[standing_env_ids, :] = 0.0
    self.vel_command_w[standing_env_ids, :] = 0.0

  # GUI.  

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon
    # 获取范围
    ranges = self.cfg.ranges

    # 创建轴
    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    # 创建滑块
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  # 计算命令
  def compute(self, dt: float) -> None:
    super().compute(dt)
    # 如果启用了joystick
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      # 确保joystick环境索引不为空
      assert self._joystick_get_env_idx is not None
      # 获取joystick环境索引
      idx = self._joystick_get_env_idx()
      # 设置joystick环境线性速度
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value

  # Visualization.

  # 调试可视化
  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    # 获取命令
    cmds = self.command.cpu().numpy()
    # 获取基座位置
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    # 获取基座四元数
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()
    # 获取缩放比例
    scale = self.cfg.viz.scale
    # 获取偏移量
    z_offset = self.cfg.viz.z_offset
    # 遍历环境索引

    for batch in env_indices:
      # 获取基座位置
      base_pos_w = base_pos_ws[batch]
      # 获取基座旋转矩阵
      base_mat_w = base_mat_ws[batch]
      # 获取命令
      cmd = cmds[batch]
      # 获取基座线性速度
      lin_vel_b = lin_vel_bs[batch]
      # 获取基座角速度
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      # 如果基座位置为零, 跳过
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Helper to transform local to world coordinates.
      # 辅助函数: 将局部坐标转换为世界坐标
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      # 命令线性速度箭头(蓝色)
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      # 添加箭头
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      # 命令角速度箭头(绿色)
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      # 添加箭头
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      # 实际线性速度箭头(青色)
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      # 添加箭头
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      # 实际角速度箭头(浅绿色)
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      # 添加箭头
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  # 实体名称
  entity_name: str
  # 是否启用heading命令
  heading_command: bool = False
  # heading控制刚度
  heading_control_stiffness: float = 1.0
  # 相对站立环境比例
  rel_standing_envs: float = 0.0
  # 相对heading环境比例
  rel_heading_envs: float = 1.0
  # 相对世界环境比例
  rel_world_envs: float = 0.0
  """Fraction of environments that use world-frame velocity commands.
  World-frame envs sample linear velocity in world frame and rotate to body
  frame each step, so the command direction stays fixed in the world."""
  # 相对正向环境比例
  rel_forward_envs: float = 0.0
  # 初始化速度概率
  """Fraction of environments that receive forward-only commands (positive
  lin_vel_x, zero lin_vel_y and ang_vel_z). Increases training coverage for
  straight-line walking, which is important for stair climbing."""
  # 初始化速度概率
  init_velocity_prob: float = 0.0

  @dataclass
  class Ranges:
    # 线性速度x范围
    lin_vel_x: tuple[float, float]
    # 线性速度y范围
    lin_vel_y: tuple[float, float]
    # 角速度z范围
    ang_vel_z: tuple[float, float]
    # heading范围
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    # z偏移量
    z_offset: float = 0.2
    # 缩放比例
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    # 构建统一速度命令
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
