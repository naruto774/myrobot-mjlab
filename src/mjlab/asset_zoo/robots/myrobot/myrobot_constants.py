"""myrobot constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

MYROBOT_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "myrobot" / "xmls" / "myrobot.xml"
)
assert MYROBOT_XML.exists()

# Joint order must match incline.csv columns 7-27.
MYROBOT_JOINT_NAMES: tuple[str, ...] = (
  "right_hip_pitch",
  "right_hip_roll",
  "right_hip_yaw",
  "right_knee",
  "right_ankle_pitch",
  "right_ankle_roll",
  "left_hip_pitch",
  "left_hip_roll",
  "left_hip_yaw",
  "left_knee",
  "left_ankle_pitch",
  "left_ankle_roll",
  "waist_pitch",
  "waist_yaw",
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_elbow",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_elbow",
  "head",
)


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(MYROBOT_XML))


##
# Actuator config.
# Mirrors the Isaac Lab reference grouping/gains in
# /home/elephant/go/robot_lab/source/robot_lab/robot_lab/assets/myrobot.py.
##

# Armatures are taken from MJCF joint defaults / reference asset.
LEG_ARMATURE = 0.00343
ANKLE_ARMATURE = 0.00262
WAIST_ARMATURE = 0.02262
ELBOW_HEAD_ARMATURE = 0.0009

MYROBOT_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch",
    ".*_hip_roll",
    ".*_hip_yaw",
    ".*_knee",
    ".*_shoulder_pitch",
  ),
  stiffness=8.0,
  damping=1.0,
  effort_limit=1.96,
  armature=LEG_ARMATURE,
)

MYROBOT_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch", ".*_ankle_roll"),
  stiffness=6.0,
  damping=1.0,
  effort_limit=2.942,
  armature=ANKLE_ARMATURE,
)

MYROBOT_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch", "waist_yaw", ".*_shoulder_roll"),
  stiffness=24.0,
  damping=1.0,
  effort_limit=2.94,
  armature=WAIST_ARMATURE,
)

MYROBOT_ACTUATOR_ELBOW_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow", "head"),
  stiffness=5.0,
  damping=1.0,
  effort_limit=0.441,
  armature=ELBOW_HEAD_ARMATURE,
)

##
# Keyframe config.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.22),
  joint_pos={
    "waist_.*": 0.0,
    ".*_hip_pitch": -0.3,
    "left_hip_roll": 0.05,
    "left_hip_yaw": -0.004,
    "left_ankle_roll": -0.06,
    "right_hip_roll": -0.05,
    "right_hip_yaw": 0.004,
    "right_ankle_roll": -0.06,
    ".*_knee": 0.5,
    ".*_ankle_pitch": -0.1,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FOOT_COLLISION = r"^(l|r)_foot_[12]_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={FOOT_COLLISION: 3, ".*_collision": 1},
  priority={FOOT_COLLISION: 1},
  friction={FOOT_COLLISION: (0.6,)},
)

##
# Final config.
##

MYROBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    MYROBOT_ACTUATOR_LEG,
    MYROBOT_ACTUATOR_ANKLE,
    MYROBOT_ACTUATOR_WAIST,
    MYROBOT_ACTUATOR_ELBOW_HEAD,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_myrobot_robot_cfg() -> EntityCfg:
  """Get a fresh myrobot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=MYROBOT_ARTICULATION,
  )


MYROBOT_ACTION_SCALE: dict[str, float] = {}
for a in MYROBOT_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    MYROBOT_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_myrobot_robot_cfg())
  viewer.launch(robot.spec.compile())
