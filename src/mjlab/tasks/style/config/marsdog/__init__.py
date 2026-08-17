from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.style.rl import StyleOnPolicyRunner

from .env_cfgs import marsdog_flat_env_cfg
from .rl_cfg import marsdog_style_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Style-Flat-Marsdog",
  env_cfg=marsdog_flat_env_cfg(),
  play_env_cfg=marsdog_flat_env_cfg(play=True),
  rl_cfg=marsdog_style_ppo_runner_cfg(),
  runner_cls=StyleOnPolicyRunner,
)
