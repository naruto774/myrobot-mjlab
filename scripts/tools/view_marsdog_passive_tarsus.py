"""Launch Marsdog in MuJoCo Viewer with mjlab actuators and debug terrain.

This tool is for checking the passive rear-leg tarsus coupling in the training
model, not the bare XML. It builds the robot through ``marsdog_constants.py`` so
mjlab injects the same actuators used by training, then adds a floor and step for
manual MuJoCo Viewer inspection.

Usage:
  uv run python scripts/tools/view_marsdog_passive_tarsus.py

In the viewer, use the Control/Actuator panel to move ``rl_calf_joint`` and
``rr_calf_joint``. The tarsus joints should follow through XML equality
constraints, while no tarsus actuator should be present.
"""

from __future__ import annotations

import argparse

import mujoco
import mujoco.viewer as viewer

from mjlab.asset_zoo.robots.marsdog.marsdog_constants import get_marsdog_robot_cfg
from mjlab.entity.entity import Entity


def _add_debug_floor_and_step(
  spec: mujoco.MjSpec,
  *,
  floor_z: float,
  step_y: float,
) -> None:
  """Add simple terrain so the robot does not fall in the viewer."""
  spec.add_material(name="ground_debug", rgba=(0.3, 0.35, 0.4, 1.0))
  spec.add_material(name="step_debug", rgba=(0.55, 0.55, 0.58, 1.0))

  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    pos=(0.0, 0.0, floor_z),
    size=(0.0, 0.0, 0.05),
    material="ground_debug",
  )

  spec.worldbody.add_body(name="step").add_geom(
    name="step1",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.8, 0.40, 0.3),
    pos=(0.0, step_y, 0.02),
    material="step_debug",
    friction=(0.8, 0.005, 0.0001),
  )


def _print_model_summary(model: mujoco.MjModel) -> None:
  """Print the actuator and rear tarsus equality summary."""
  actuators = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
  ]
  print(f"nu = {model.nu}")
  print(f"actuators = {actuators}")
  print(f"tarsus actuators = {[name for name in actuators if 'tarsus' in name]}")
  print(f"neq = {model.neq}")
  if model.neq >= 2:
    print(
      "rear tarsus equality polycoefs = "
      f"{model.eq_data[model.neq - 2].tolist()}, "
      f"{model.eq_data[model.neq - 1].tolist()}"
    )


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Open Marsdog with training actuators to inspect passive tarsus."
  )
  parser.add_argument(
    "--floor-z",
    type=float,
    default=-0.25,
    help="Debug floor Z position.",
  )
  parser.add_argument(
    "--step-y",
    type=float,
    default=-1.9,
    help="Debug step Y position.",
  )
  args = parser.parse_args()

  robot = Entity(get_marsdog_robot_cfg())
  spec = robot.spec
  _add_debug_floor_and_step(spec, floor_z=args.floor_z, step_y=args.step_y)

  model = spec.compile()
  data = mujoco.MjData(model)
  _print_model_summary(model)

  viewer.launch(model, data)


if __name__ == "__main__":
  main()
