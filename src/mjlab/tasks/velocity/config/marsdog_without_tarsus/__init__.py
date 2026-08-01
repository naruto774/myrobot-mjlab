from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  marsdog_without_tarsus_flat_env_cfg,
  marsdog_without_tarsus_rough_env_cfg,
)
from .rl_cfg import marsdog_without_tarsus_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Marsdog-Without-Tarsus",
  env_cfg=marsdog_without_tarsus_rough_env_cfg(),
  play_env_cfg=marsdog_without_tarsus_rough_env_cfg(play=True),
  rl_cfg=marsdog_without_tarsus_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Marsdog-Without-Tarsus",
  env_cfg=marsdog_without_tarsus_flat_env_cfg(),
  play_env_cfg=marsdog_without_tarsus_flat_env_cfg(play=True),
  rl_cfg=marsdog_without_tarsus_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
