"""Compare equality constraint enforcement on CPU MuJoCo vs mujoco_warp."""

from __future__ import annotations

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.asset_zoo.robots.marsdog.marsdog_constants import get_marsdog_robot_cfg
from mjlab.entity.entity import Entity
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg

MARSDOG_XML = "src/mjlab/asset_zoo/robots/marsdog/xmls/assets/marsdog.xml"
TARSUS_JOINT = "rr_tarsus_joint"
CALF_JOINT = "rr_calf_joint"
ERR_TOL = 1e-3


def _get_joint_qposadr(model: mujoco.MjModel) -> tuple[int, int]:
  q_tarsus = model.joint(TARSUS_JOINT).qposadr[0]
  q_calf = model.joint(CALF_JOINT).qposadr[0]
  return q_tarsus, q_calf


def _print_model_equality_status(model: mujoco.MjModel, label: str) -> None:
  print(f"[{label}] neq = {model.neq}")
  print(
    f"[{label}] equality disabled = "
    f"{bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EQUALITY)}"
  )


def test_cpu_vs_warp_bare_xml(device: str) -> None:
  """Compare bare marsdog.xml on CPU MuJoCo and mujoco_warp."""
  model = mujoco.MjModel.from_xml_path(MARSDOG_XML)
  q_tarsus, q_calf = _get_joint_qposadr(model)
  _print_model_equality_status(model, "bare xml")

  mj_data = mujoco.MjData(model)
  model.opt.gravity[:] = 0
  mj_data.qpos[q_calf] = 0.5
  mj_data.qpos[q_tarsus] = 0.0
  mujoco.mj_forward(model, mj_data)

  with wp.ScopedDevice(device):
    wp_model = mjwarp.put_model(model)
    wp_data = mjwarp.put_data(model, mj_data, nworld=1)

  print("\n=== bare xml: CPU vs warp ===")
  for i in range(100):
    mujoco.mj_step(model, mj_data)
    with wp.ScopedDevice(device):
      mjwarp.step(wp_model, wp_data)

    if i % 20 == 0:
      err_cpu = mj_data.qpos[q_tarsus] - mj_data.qpos[q_calf]
      err_warp = float(
        wp_data.qpos.numpy()[0, q_tarsus] - wp_data.qpos.numpy()[0, q_calf]
      )
      print(f"step {i:3d}: cpu={err_cpu:.6f}, warp={err_warp:.6f}")

  err_cpu = abs(mj_data.qpos[q_tarsus] - mj_data.qpos[q_calf])
  err_warp = abs(
    float(wp_data.qpos.numpy()[0, q_tarsus] - wp_data.qpos.numpy()[0, q_calf])
  )
  print(f"final |err| cpu={err_cpu:.6f}, warp={err_warp:.6f}")
  assert err_cpu < ERR_TOL, f"CPU equality failed: |err|={err_cpu}"
  assert err_warp < ERR_TOL, f"Warp equality failed: |err|={err_warp}"


def test_warp_training_model(device: str) -> None:
  """Test equality on the mjlab training build path (Entity + Simulation)."""
  robot = Entity(get_marsdog_robot_cfg())
  model = robot.compile()
  sim_cfg = SimulationCfg(mujoco=MujocoCfg(gravity=(0.0, 0.0, 0.0)))
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  robot.initialize(model, sim.model, sim.data, device)

  q_tarsus, q_calf = _get_joint_qposadr(sim.mj_model)
  _print_model_equality_status(sim.mj_model, "training model")

  sim.data.qpos[0, q_calf] = 0.5
  sim.data.qpos[0, q_tarsus] = 0.0
  sim.forward()

  print("\n=== training model: mjlab Simulation.step ===")
  for i in range(200):
    sim.step()
    if i % 20 == 0:
      err = float(sim.data.qpos[0, q_tarsus] - sim.data.qpos[0, q_calf])
      print(f"step {i:3d}: err={err:.6f}")

  final_err = abs(float(sim.data.qpos[0, q_tarsus] - sim.data.qpos[0, q_calf]))
  print(f"final |err| = {final_err:.6f}")
  assert final_err < ERR_TOL, f"Training backend equality failed: |err|={final_err}"


def main() -> None:
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"device = {device}")

  test_cpu_vs_warp_bare_xml(device)
  test_warp_training_model(device)
  print("\nPASS: equality works on CPU and warp training backend")


if __name__ == "__main__":
  main()
