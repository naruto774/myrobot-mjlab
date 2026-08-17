"""RL configuration for the Marsdog style task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlMultiCriticPpoAlgorithmCfg,
  RslRlOnPolicyRunnerCfg,
)


def marsdog_style_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Marsdog style imitation."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlMultiCriticPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      num_reward_groups=2,
      advantage_weights=(0.5, 0.5),
    ),
    experiment_name="marsdog_style",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
