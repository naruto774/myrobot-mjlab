"""Multi-critic PPO on top of rsl-rl-lib 5.2.

One actor and two independent critic MLPs. Each reward group has its own GAE;
normalized advantages are fused with ``α = [0.5, 0.5]`` before the PPO clip.
Value loss is the mean clipped MSE of both heads.

This is not a vendor of APEX's old ``rsl_rl``; it uses the 5.2 ``MLPModel`` /
``RolloutStorage`` APIs.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import chain

import torch
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage.rollout_storage import RolloutStorage
from rsl_rl.utils import (
  compile_model,
  resolve_callable,
  resolve_obs_groups,
  resolve_optimizer,
)
from tensordict import TensorDict


def fuse_group_advantages(
  advantages: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
  """Normalize each group then take a weighted sum.

  Args:
    advantages: Per-group GAE, shape ``(T, N, G)``.
    weights: Group weights, shape ``(G,)``.

  Returns:
    Fused policy advantage, shape ``(T, N, 1)``.
  """
  fused = torch.zeros(
    advantages.shape[:-1], device=advantages.device, dtype=advantages.dtype
  )
  for group in range(advantages.shape[-1]):
    adv = advantages[..., group]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    fused = fused + weights[group] * adv
  return fused.unsqueeze(-1)


class _ConcatCritic:
  """Forward two critic MLPs and concatenate their scalar values."""

  def __init__(self, critic_a: MLPModel, critic_b: MLPModel) -> None:
    self.critic_a = critic_a
    self.critic_b = critic_b

  def __call__(self, obs: TensorDict, **kwargs) -> torch.Tensor:
    return torch.cat(
      [self.critic_a(obs, **kwargs), self.critic_b(obs, **kwargs)], dim=-1
    )

  def reset(self, dones: torch.Tensor | None = None) -> None:
    self.critic_a.reset(dones)
    self.critic_b.reset(dones)

  def update_normalization(self, obs: TensorDict) -> None:
    self.critic_a.update_normalization(obs)
    self.critic_b.update_normalization(obs)

  def get_hidden_state(self):
    return self.critic_a.get_hidden_state()

  def train(self) -> None:
    self.critic_a.train()
    self.critic_b.train()

  def eval(self) -> None:
    self.critic_a.eval()
    self.critic_b.eval()

  @property
  def is_recurrent(self) -> bool:
    return bool(self.critic_a.is_recurrent or self.critic_b.is_recurrent)

  def parameters(self):
    return chain(self.critic_a.parameters(), self.critic_b.parameters())


class MultiCriticRolloutStorage(RolloutStorage):
  """Rollout buffer whose rewards / values / returns are ``(T, N, G)``."""

  def __init__(
    self,
    training_type: str,
    num_envs: int,
    num_transitions_per_env: int,
    obs: TensorDict,
    actions_shape: tuple[int, ...] | list[int],
    device: str = "cpu",
    num_reward_groups: int = 2,
  ) -> None:
    super().__init__(
      training_type,
      num_envs,
      num_transitions_per_env,
      obs,
      actions_shape,
      device,
    )
    self.num_reward_groups = num_reward_groups
    self.rewards = torch.zeros(
      num_transitions_per_env, num_envs, num_reward_groups, device=device
    )
    if training_type == "rl":
      self.values = torch.zeros(
        num_transitions_per_env, num_envs, num_reward_groups, device=device
      )
      self.returns = torch.zeros(
        num_transitions_per_env, num_envs, num_reward_groups, device=device
      )
      self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)

  def add_transition(self, transition: RolloutStorage.Transition) -> None:
    if self.step >= self.num_transitions_per_env:
      raise OverflowError(
        "Rollout buffer overflow! You should call clear() before adding new "
        "transitions."
      )
    assert transition.observations is not None
    self.observations[self.step].copy_(transition.observations)  # type: ignore[arg-type]
    self.actions[self.step].copy_(transition.actions)  # type: ignore[arg-type]
    assert transition.rewards is not None
    self.rewards[self.step].copy_(
      transition.rewards.reshape(self.num_envs, self.num_reward_groups)
    )
    assert transition.dones is not None
    self.dones[self.step].copy_(transition.dones.view(-1, 1))
    if self.training_type == "rl":
      self.values[self.step].copy_(transition.values)  # type: ignore[arg-type]
      assert transition.actions_log_prob is not None
      self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
      if self.distribution_params is None:
        self.distribution_params = tuple(
          torch.zeros(self.num_transitions_per_env, *p.shape, device=self.device)
          for p in transition.distribution_params  # type: ignore[union-attr]
        )
      for i, p in enumerate(transition.distribution_params):  # type: ignore[arg-type]
        self.distribution_params[i][self.step].copy_(p)
    self._save_hidden_states(transition.hidden_states)
    self.step += 1


class MultiCriticPPO(PPO):
  """PPO with two critic heads and fused advantages."""

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    storage: RolloutStorage,
    critic2: MLPModel,
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    clip_param: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    value_loss_coef: float = 1.0,
    entropy_coef: float = 0.01,
    learning_rate: float = 0.001,
    max_grad_norm: float = 1.0,
    optimizer: str = "adam",
    use_clipped_value_loss: bool = True,
    schedule: str = "adaptive",
    desired_kl: float = 0.01,
    normalize_advantage_per_mini_batch: bool = False,
    device: str = "cpu",
    rnd_cfg: dict | None = None,
    symmetry_cfg: dict | None = None,
    multi_gpu_cfg: dict | None = None,
    num_reward_groups: int = 2,
    advantage_weights: tuple[float, ...] = (0.5, 0.5),
    share_cnn_encoders: bool = False,
  ) -> None:
    del share_cnn_encoders
    super().__init__(
      actor=actor,
      critic=critic,
      storage=storage,
      num_learning_epochs=num_learning_epochs,
      num_mini_batches=num_mini_batches,
      clip_param=clip_param,
      gamma=gamma,
      lam=lam,
      value_loss_coef=value_loss_coef,
      entropy_coef=entropy_coef,
      learning_rate=learning_rate,
      max_grad_norm=max_grad_norm,
      optimizer=optimizer,
      use_clipped_value_loss=use_clipped_value_loss,
      schedule=schedule,
      desired_kl=desired_kl,
      normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
      device=device,
      rnd_cfg=rnd_cfg,
      symmetry_cfg=symmetry_cfg,
      multi_gpu_cfg=multi_gpu_cfg,
    )
    self.num_reward_groups = num_reward_groups
    self.advantage_weights = torch.tensor(
      advantage_weights, device=self.device, dtype=torch.float32
    )
    self.critic2 = critic2.to(self.device)
    self._raw_critic2 = self.critic2
    self.critic = _ConcatCritic(self.critic, self.critic2)  # type: ignore[assignment]
    optimizer_cls = resolve_optimizer(optimizer)
    self.optimizer = optimizer_cls(  # type: ignore[misc]
      chain(
        self.actor.parameters(),
        self._raw_critic.parameters(),
        self._raw_critic2.parameters(),
      ),
      lr=learning_rate,
    )

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)
    if self.rnd:
      self.rnd.update_normalization(obs)

    groups = extras.get("reward_groups")
    if groups is None:
      groups = rewards.unsqueeze(-1).expand(-1, self.num_reward_groups)
    self.transition.rewards = groups.to(self.device).clone()
    self.transition.dones = dones
    step_rewards = self.transition.rewards
    step_values = self.transition.values
    assert step_rewards is not None
    assert step_values is not None

    if self.rnd:
      self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
      step_rewards = step_rewards + self.intrinsic_rewards.unsqueeze(-1)

    if "time_outs" in extras:
      timeouts = extras["time_outs"].to(self.device).unsqueeze(-1)
      step_rewards = step_rewards + self.gamma * step_values * timeouts

    self.transition.rewards = step_rewards
    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.actor.reset(dones)
    self.critic.reset(dones)

  def compute_returns(self, obs: TensorDict) -> None:
    storage = self.storage
    last_values = self.critic(obs).detach()
    advantage = torch.zeros_like(last_values)
    for step in reversed(range(storage.num_transitions_per_env)):
      next_values = (
        last_values
        if step == storage.num_transitions_per_env - 1
        else storage.values[step + 1]
      )
      next_is_not_terminal = 1.0 - storage.dones[step].float()
      delta = (
        storage.rewards[step]
        + next_is_not_terminal * self.gamma * next_values
        - storage.values[step]
      )
      advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
      storage.returns[step] = advantage + storage.values[step]
    group_adv = storage.returns - storage.values
    if self.normalize_advantage_per_mini_batch:
      storage.advantages = group_adv.mean(dim=-1, keepdim=True)
    else:
      storage.advantages = fuse_group_advantages(group_adv, self.advantage_weights)

  def train_mode(self) -> None:
    self.actor.train()
    self.critic.train()
    if self.rnd:
      self.rnd.train()

  def eval_mode(self) -> None:
    self.actor.eval()
    self.critic.eval()
    if self.rnd:
      self.rnd.eval()

  def save(self) -> dict:
    saved_dict = super().save()
    saved_dict["critic2_state_dict"] = self._raw_critic2.state_dict()
    return saved_dict

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    if load_cfg is None or load_cfg.get("critic", True):
      if "critic2_state_dict" in loaded_dict:
        self._raw_critic2.load_state_dict(
          loaded_dict["critic2_state_dict"], strict=strict
        )
    return load_iteration

  def compile(self, mode: str | None = None) -> None:
    self.actor = compile_model(self._raw_actor, mode)  # type: ignore[assignment]
    critic_a = compile_model(self._raw_critic, mode)
    critic_b = compile_model(self._raw_critic2, mode)
    self.critic2 = critic_b
    self.critic = _ConcatCritic(critic_a, critic_b)  # type: ignore[assignment]

  def broadcast_parameters(self) -> None:
    model_params = [
      self._raw_actor.state_dict(),
      self._raw_critic.state_dict(),
      self._raw_critic2.state_dict(),
    ]
    if self.rnd:
      model_params.append(self.rnd.predictor.state_dict())
    torch.distributed.broadcast_object_list(model_params, src=0)  # type: ignore[attr-defined]
    self._raw_actor.load_state_dict(model_params[0])
    self._raw_critic.load_state_dict(model_params[1])
    self._raw_critic2.load_state_dict(model_params[2])
    if self.rnd:
      self.rnd.predictor.load_state_dict(model_params[3])

  def reduce_parameters(self) -> None:
    all_params = chain(
      self.actor.parameters(),
      self._raw_critic.parameters(),
      self._raw_critic2.parameters(),
    )
    if self.rnd:
      all_params = chain(all_params, self.rnd.parameters())
    all_params = list(all_params)
    grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
    all_grads = torch.cat(grads)
    torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)  # type: ignore[attr-defined]
    all_grads /= self.gpu_world_size
    offset = 0
    for param in all_params:
      if param.grad is not None:
        numel = param.numel()
        param.grad.data.copy_(
          all_grads[offset : offset + numel].view_as(param.grad.data)
        )
        offset += numel

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: VecEnv, cfg: dict, device: str
  ) -> MultiCriticPPO:
    alg_class: type[MultiCriticPPO] = resolve_callable(
      cfg["algorithm"].pop("class_name")
    )  # type: ignore[assignment]
    actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore[assignment]
    critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore[assignment]

    default_sets = ["actor", "critic"]
    if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
      default_sets.append("rnd_state")
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

    num_reward_groups = int(cfg["algorithm"].pop("num_reward_groups", 2))
    advantage_weights = tuple(cfg["algorithm"].pop("advantage_weights", (0.5, 0.5)))

    actor: MLPModel = actor_class(
      obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
    ).to(device)
    print(f"Actor Model: {actor}")
    if cfg["algorithm"].pop("share_cnn_encoders", None):
      cfg["critic"]["cnns"] = actor.cnns
    critic_kwargs = deepcopy(cfg["critic"])
    critic: MLPModel = critic_class(
      obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
    ).to(device)
    critic2: MLPModel = critic_class(
      obs, cfg["obs_groups"], "critic", 1, **critic_kwargs
    ).to(device)
    print(f"Critic Model: {critic}")
    print(f"Critic2 Model: {critic2}")

    storage = MultiCriticRolloutStorage(
      "rl",
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      [env.num_actions],
      device,
      num_reward_groups=num_reward_groups,
    )
    alg: MultiCriticPPO = alg_class(
      actor,
      critic,
      storage,
      critic2=critic2,
      device=device,
      num_reward_groups=num_reward_groups,
      advantage_weights=advantage_weights,
      **cfg["algorithm"],
      multi_gpu_cfg=cfg["multi_gpu"],
    )
    alg.compile(cfg.get("torch_compile_mode"))
    return alg
