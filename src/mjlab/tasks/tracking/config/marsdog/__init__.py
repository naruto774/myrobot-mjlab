from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import marsdog_flat_tracking_env_cfg
from .rl_cfg import marsdog_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-marsdog",
  env_cfg=marsdog_flat_tracking_env_cfg(),
  play_env_cfg=marsdog_flat_tracking_env_cfg(play=True),
  rl_cfg=marsdog_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-marsdog-No-State-Estimation",
  env_cfg=marsdog_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=marsdog_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=marsdog_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
