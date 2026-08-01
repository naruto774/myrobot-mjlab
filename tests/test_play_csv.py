import csv
from pathlib import Path
from typing import Any

import torch

from mjlab.scripts.play import _wrap_policy_for_torque_csv


def test_torque_csv_records_named_joint_torques(tmp_path: Path) -> None:
  torques = torch.tensor([1.5, -2.0])

  class IdentityPolicy:
    def __call__(self, obs: Any) -> torch.Tensor:
      return obs

  policy, flush = _wrap_policy_for_torque_csv(
    IdentityPolicy(),
    joint_names=("hip_joint", "knee_joint"),
    read_torques=lambda: torques,
    step_dt=0.02,
  )

  policy(torch.zeros(1, 2))
  torques.copy_(torch.tensor([3.0, 4.0]))
  policy(torch.zeros(1, 2))

  output_path = tmp_path / "torques.csv"
  flush(str(output_path))

  with output_path.open(newline="") as f:
    rows = list(csv.reader(f))

  assert rows == [
    [
      "step",
      "time_s",
      "torque_0_hip_joint_Nm",
      "torque_1_knee_joint_Nm",
    ],
    ["0", "0.0", "1.5", "-2.0"],
    ["1", "0.02", "3.0", "4.0"],
  ]
