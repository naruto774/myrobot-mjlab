"""Marsdog constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

MARSDOG_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "marsdog"
  / "xmls"
  / "assets"
  / "marsdog.xml"
)
assert MARSDOG_XML.exists()
MARSDOG_JOINT_NAMES: tuple[str, ...] = (
  "tail1_pitch_joint",
  "tail1_yaw_joint",
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
  return mujoco.MjSpec.from_file(str(MARSDOG_XML))


##
# Actuator config.
##
# 低速端等效惯量
RS00_INERTIA = 0.001
RS01_INERTIA = 0.0042
EL05_INERTIA = 0.00094
# 裸转子惯量
PA43_INERTIA = 0.001

RS00_GEAR_RATIO = 10
RS01_GEAR_RATIO = 7.75
EL05_GEAR_RATIO = 9
PA43_GEAR_RATIO = 25

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

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_RS00 = RS00_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_RS00 = 2 * DAMPING_RATIO * RS00_ACTUATOR.reflected_inertia * NATURAL_FREQ

STIFFNESS_RS01 = RS01_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_RS01 = 2 * DAMPING_RATIO * RS01_ACTUATOR.reflected_inertia * NATURAL_FREQ

STIFFNESS_EL05 = EL05_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_EL05 = 2 * DAMPING_RATIO * EL05_ACTUATOR.reflected_inertia * NATURAL_FREQ

STIFFNESS_PA43 = PA43_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_PA43 = 2 * DAMPING_RATIO * PA43_ACTUATOR.reflected_inertia * NATURAL_FREQ

REAR_LEG_STIFFNESS = 12.0
REAR_LEG_DAMPING = 1.8
FRONT_LEG_STIFFNESS = 10.0
FRONT_LEG_DAMPING = 1.6
WAIST_STIFFNESS = 20.0
WAIST_DAMPING = 2.5
HEAD_TAIL_STIFFNESS = 4.0
HEAD_TAIL_DAMPING = 0.6

# Rear-leg tarsus joints are PASSIVE: a <equality> in the XML couples each tarsus
# to its calf (rl_tarsus = +rl_calf, rr_tarsus = -rr_calf), reproducing the real
# 4-bar linkage. They are therefore NOT actuated here, so the action space matches
# the hardware (no tarsus motor). The tarsus joints still exist as DOFs, so they
# remain in the proprioceptive observation and motion command but carry no action.
# Front-leg tarsus links are fixed (no joint in XML).
# Tail segments 2-12 mimic tail1 via <equality>; only tail1 is actuated.
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
    "tail1_pitch_joint",
    "tail1_yaw_joint",
  ),
  stiffness=HEAD_TAIL_STIFFNESS,
  damping=HEAD_TAIL_DAMPING,
  effort_limit=EL05_ACTUATOR.effort_limit,
  armature=EL05_ACTUATOR.reflected_inertia,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.35),
  joint_pos={
    "tail1_pitch_joint": 0.0,
    "tail1_yaw_joint": 0.0,
    "rl_hip_joint": 0.15,
    "rl_thigh_joint": 0.11,
    "rl_calf_joint": 0.0,
    # Passive: coupled to calf by the XML <equality> (rl_tarsus = +rl_calf).
    # Keep consistent with rl_calf so reset does not violate the constraint.
    "rl_tarsus_joint": 0.0,
    "rr_hip_joint": -0.09,
    "rr_thigh_joint": -0.47,
    "rr_calf_joint": 0.0,
    # Passive: coupled to calf by the XML <equality> (rr_tarsus = -rr_calf).
    "rr_tarsus_joint": 0.0,
    "waist_roll_joint": 0.07,
    "waist_pitch_joint": 0.0,
    "waist_yaw_joint": -0.10,
    "neck_pitch_joint": -0.58,
    "head_roll_joint": -0.04,
    "head_yaw_joint": -0.03,
    "head_pitch_joint": 0.50,
    "fl_hip_pitch_joint": 0.36,
    "fl_thigh_roll_joint": -0.18,
    "fl_calf_joint": -0.75,
    "fr_hip_pitch_joint": -0.42,
    "fr_thigh_roll_joint": 0.13,
    "fr_calf_joint": 0.70,
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

MARSDOG_ARTICULATION = EntityArticulationInfoCfg(
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


def get_marsdog_robot_cfg() -> EntityCfg:
  """Get a fresh Marsdog robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=MARSDOG_ARTICULATION,
  )


MARSDOG_ACTION_SCALE: dict[str, float] = {}
for a in MARSDOG_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    MARSDOG_ACTION_SCALE[n] =0.25* e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_marsdog_robot_cfg())

  viewer.launch(robot.spec.compile())
