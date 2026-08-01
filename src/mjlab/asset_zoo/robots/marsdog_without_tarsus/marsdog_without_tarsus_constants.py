"""Marsdog-without-tarsus constants.

Front tarsus joints are removed from the MJCF (links remain welded as fixed
bodies). Rear tarsus stay as calf-mimic joints. Actuation comes from
BuiltinPositionActuatorCfg only — the MJCF has no XML <motor> block.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

MARSDOG_WITHOUT_TARSUS_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "marsdog_without_tarsus"
  / "xmls"
  / "assets"
  / "marsdog_without_tarsus.xml"
)
assert MARSDOG_WITHOUT_TARSUS_XML.exists()

# Actuated DOFs only (no front/rear tarsus).
MARSDOG_WITHOUT_TARSUS_JOINT_NAMES: tuple[str, ...] = (
  "rl_hip_joint",
  "rl_thigh_joint",
  "rl_calf_joint",
  "rr_hip_joint",
  "rr_thigh_joint",
  "rr_calf_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "waist_yaw_joint",
  "neck_pitch_joint",
  "head_roll_joint",
  "head_yaw_joint",
  "head_pitch_joint",
  "fl_hip_pitch_joint",
  "fl_thigh_roll_joint",
  "fl_calf_joint",
  "fr_hip_pitch_joint",
  "fr_thigh_roll_joint",
  "fr_calf_joint",
)


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(MARSDOG_WITHOUT_TARSUS_XML))


##
# Actuator config.
##
# 低速端等效惯量
RS00_INERTIA = 0.001
RS01_INERTIA = 0.0042
EL05_INERTIA = 0.00094
# 裸转子惯量
PA43_INERTIA = 0.001

RS00_ACTUATOR = ElectricActuator(
  reflected_inertia=RS00_INERTIA,
  velocity_limit=32.9867,
  effort_limit=14,
)
RS01_ACTUATOR = ElectricActuator(
  reflected_inertia=RS01_INERTIA,
  velocity_limit=32.9867,
  effort_limit=17,
)
EL05_ACTUATOR = ElectricActuator(
  reflected_inertia=EL05_INERTIA,
  velocity_limit=10.472,
  effort_limit=6,
)
PA43_ACTUATOR = ElectricActuator(
  reflected_inertia=PA43_INERTIA,
  velocity_limit=14.66087,
  effort_limit=18,
)

REAR_LEG_STIFFNESS = 12.0
REAR_LEG_DAMPING = 1.8

FRONT_LEG_STIFFNESS = 16.0
FRONT_LEG_DAMPING = 1.6
WAIST_STIFFNESS = 20.0
WAIST_DAMPING = 2.5
HEAD_TAIL_STIFFNESS = 4.0
HEAD_TAIL_DAMPING = 0.6

MARSDOG_REAR_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "rl_hip_joint",
    "rr_hip_joint",
  ),
  stiffness=REAR_LEG_STIFFNESS,
  damping=REAR_LEG_DAMPING,
  effort_limit=PA43_ACTUATOR.effort_limit,
  armature=PA43_ACTUATOR.reflected_inertia,
)
MARSDOG_REAR_LOWER_LEG_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "rl_thigh_joint",
    "rl_calf_joint",
    "rr_thigh_joint",
    "rr_calf_joint",
  ),
  stiffness=REAR_LEG_STIFFNESS,
  damping=REAR_LEG_DAMPING,
  effort_limit=RS00_ACTUATOR.effort_limit,
  armature=RS00_ACTUATOR.reflected_inertia,
)

MARSDOG_FRONT_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("fl_hip_pitch_joint", "fr_hip_pitch_joint"),
  stiffness=FRONT_LEG_STIFFNESS,
  damping=FRONT_LEG_DAMPING,
  effort_limit=RS01_ACTUATOR.effort_limit,
  armature=RS01_ACTUATOR.reflected_inertia,
)
MARSDOG_FRONT_LOWER_LEG_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "fl_thigh_roll_joint",
    "fr_thigh_roll_joint",
    "fl_calf_joint",
    "fr_calf_joint",
  ),
  stiffness=FRONT_LEG_STIFFNESS,
  damping=FRONT_LEG_DAMPING,
  effort_limit=EL05_ACTUATOR.effort_limit,
  armature=EL05_ACTUATOR.reflected_inertia,
)

MARSDOG_WAIST_ROLL_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_roll_joint",),
  stiffness=WAIST_STIFFNESS,
  damping=WAIST_DAMPING,
  effort_limit=RS01_ACTUATOR.effort_limit,
  armature=RS01_ACTUATOR.reflected_inertia,
)
MARSDOG_WAIST_PITCH_YAW_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_pitch_joint",
    "waist_yaw_joint",
  ),
  stiffness=WAIST_STIFFNESS,
  damping=WAIST_DAMPING,
  effort_limit=PA43_ACTUATOR.effort_limit,
  armature=PA43_ACTUATOR.reflected_inertia,
)
MARSDOG_NECK_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("neck_pitch_joint",),
  stiffness=HEAD_TAIL_STIFFNESS,
  damping=HEAD_TAIL_DAMPING,
  effort_limit=PA43_ACTUATOR.effort_limit,
  armature=PA43_ACTUATOR.reflected_inertia,
)
MARSDOG_HEAD_TAIL_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "head_roll_joint",
    "head_yaw_joint",
    "head_pitch_joint",
  ),
  stiffness=HEAD_TAIL_STIFFNESS,
  damping=HEAD_TAIL_DAMPING,
  effort_limit=EL05_ACTUATOR.effort_limit,
  armature=EL05_ACTUATOR.reflected_inertia,
)

##
# Keyframes.
##

# Front joints keep the user nominal pose; rear thigh/calf are matched so all
# four feet sit near z=0. Rear tarsus tracks calf via MJCF equality mimic.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.289),
  joint_pos={
    "rl_hip_joint": 0.0,
    "rl_thigh_joint": -0.01,
    "rl_calf_joint": 0.44,
    "rl_tarsus_joint": 0.44,
    "rr_hip_joint": 0.0,
    "rr_thigh_joint": -0.01,
    "rr_calf_joint": 0.44,
    "rr_tarsus_joint": 0.44,
    "fl_hip_pitch_joint": -0.6,
    "fl_thigh_roll_joint": 0.0,
    "fl_calf_joint": 0.8,
    "fr_hip_pitch_joint": -0.6,
    "fr_thigh_roll_joint": 0.0,
    "fr_calf_joint": 0.8,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^(fl|fr|rl|rr)_foot_collision$"

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  solref=(0.01, 1),
  condim={_foot_regex: 6, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (1, 5e-3, 5e-4)},
)

##
# Final config.
##

MARSDOG_WITHOUT_TARSUS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    MARSDOG_REAR_HIP_ACTUATOR_CFG,
    MARSDOG_REAR_LOWER_LEG_ACTUATOR_CFG,
    MARSDOG_FRONT_HIP_ACTUATOR_CFG,
    MARSDOG_FRONT_LOWER_LEG_ACTUATOR_CFG,
    MARSDOG_WAIST_ROLL_ACTUATOR_CFG,
    MARSDOG_WAIST_PITCH_YAW_ACTUATOR_CFG,
    MARSDOG_NECK_ACTUATOR_CFG,
    MARSDOG_HEAD_TAIL_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_marsdog_without_tarsus_robot_cfg() -> EntityCfg:
  """Get a fresh Marsdog-without-tarsus robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    # Full collision: feet get condim-6 friction; body geoms prevent floor clipping.
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=MARSDOG_WITHOUT_TARSUS_ARTICULATION,
  )


MARSDOG_WITHOUT_TARSUS_ACTION_SCALE: dict[str, float] = {}
for a in MARSDOG_WITHOUT_TARSUS_ARTICULATION.actuators:
  if not isinstance(a, BuiltinPositionActuatorCfg):
    continue
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    MARSDOG_WITHOUT_TARSUS_ACTION_SCALE[n] = 0.25 * e / s
MARSDOG_WITHOUT_TARSUS_ACTION_SCALE.update(
  {
    "fl_hip_pitch_joint": 0.25,
    "fr_hip_pitch_joint": 0.25,
    "fl_thigh_roll_joint": 0.25,
    "fr_thigh_roll_joint": 0.25,
    "fl_calf_joint": 0.25,
    "fr_calf_joint": 0.25,
    "waist_roll_joint": 0.25,
    "waist_pitch_joint": 0.25,
    "waist_yaw_joint": 0.25,
    "neck_pitch_joint": 0.25,
    "head_roll_joint": 0.25,
    "head_yaw_joint": 0.25,
    "head_pitch_joint": 0.25,
    "rl_hip_joint": 0.25,
    "rl_thigh_joint": 0.25,
    "rl_calf_joint": 0.25,
    "rr_hip_joint": 0.25,
    "rr_thigh_joint": 0.25,
    "rr_calf_joint": 0.25,
  }
)

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_marsdog_without_tarsus_robot_cfg())

  viewer.launch(robot.spec.compile())
