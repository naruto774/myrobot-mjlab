"""Extract absolute joint positions and velocities from a tracking rollout CSV.

The script supports the actor observation layouts currently used by this project:

* Marsdog: 23-D q/qdot (including two passive rear tarsus joints), 21-D action.
* Go2: 12-D q/qdot and action.

The policy observes relative angles. This script restores absolute joint angles:

  q_abs = q_rel + q_default

Usage:
  uv run scripts/tools/extract_marsdog_joint_states.py \
    --input-file rollout.csv \
    --output-file marsdog_joint_states.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotObservationLayout:
  """Joint metadata and actor-observation indices for one robot."""

  joint_names: tuple[str, ...]
  default_joint_pos: tuple[float, ...]
  joint_pos_start: int
  joint_vel_start: int


GO2_JOINT_NAMES = (
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
)
GO2_DEFAULT_JOINT_POS = (
  -0.2685233,
  -0.18719791,
  -0.8383993,
  0.45251113,
  0.6864915,
  -1.4168774,
  -0.1301512,
  0.8982419,
  -1.1499609,
  0.2014172,
  1.0143824,
  -1.3094097,
)

MARSDOG_JOINT_NAMES = (
  "rl_hip_joint",
  "rl_thigh_joint",
  "rl_calf_joint",
  "rl_tarsus_joint",
  "rr_hip_joint",
  "rr_thigh_joint",
  "rr_calf_joint",
  "rr_tarsus_joint",
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
  "fl_tarsus_joint",
  "fr_hip_pitch_joint",
  "fr_thigh_roll_joint",
  "fr_calf_joint",
  "fr_tarsus_joint",
)
MARSDOG_DEFAULT_JOINT_POS = (
  0.13826222717761993,
  -0.18853963911533356,
  0.02164936251938343,
  0.02164936251938343,
  0.04337790980935097,
  -0.4007984697818756,
  0.1781260371208191,
  -0.1781260371208191,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.24397215247154236,
  -0.08243412524461746,
  0.29261311888694763,
  0.01103648729622364,
  0.18704774975776672,
  -0.08918917179107666,
  -0.17999999225139618,
  0.35730117559432983,
)

GO2_LAYOUT = RobotObservationLayout(
  joint_names=GO2_JOINT_NAMES,
  default_joint_pos=GO2_DEFAULT_JOINT_POS,
  joint_pos_start=33,
  joint_vel_start=45,
)
MARSDOG_LAYOUT = RobotObservationLayout(
  joint_names=MARSDOG_JOINT_NAMES,
  default_joint_pos=MARSDOG_DEFAULT_JOINT_POS,
  joint_pos_start=55,
  joint_vel_start=78,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input-file",
    type=Path,
    required=True,
    help="Rollout CSV written with play --save-rollout-csv.",
  )
  parser.add_argument(
    "--output-file",
    type=Path,
    required=True,
    help="Destination CSV for measured joint states.",
  )
  parser.add_argument(
    "--control-dt",
    type=float,
    default=0.02,
    help="Control interval in seconds, used to write time_s (default: 0.02).",
  )
  return parser.parse_args()


def extract_joint_states(
  input_file: Path, output_file: Path, control_dt: float
) -> None:
  """Write absolute q [rad] and qd [rad/s] for every rollout step."""
  if control_dt <= 0:
    raise ValueError(f"--control-dt must be positive, got {control_dt}.")

  with input_file.open(newline="") as input_csv:
    reader = csv.DictReader(input_csv)
    if reader.fieldnames is None:
      raise ValueError(f"Input CSV has no header: {input_file}")

    layout = _detect_layout(list(reader.fieldnames))
    num_joints = len(layout.joint_names)
    required_columns = [
      f"obs_{index}"
      for index in range(layout.joint_pos_start, layout.joint_vel_start + num_joints)
    ]
    missing_columns = [
      name for name in required_columns if name not in reader.fieldnames
    ]
    if missing_columns:
      raise ValueError(
        "Input does not match the Marsdog no-state-estimation actor layout. "
        f"Missing columns: {', '.join(missing_columns)}"
      )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
      ["step", "time_s"]
      + [f"joint_pos_{name}_rad" for name in layout.joint_names]
      + [f"joint_vel_{name}_rad_s" for name in layout.joint_names]
    )
    with output_file.open("w", newline="") as output_csv:
      writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
      writer.writeheader()
      for step, row in enumerate(reader):
        output_row = {"step": step, "time_s": step * control_dt}
        for index, joint_name in enumerate(layout.joint_names):
          joint_pos_rel = float(row[f"obs_{layout.joint_pos_start + index}"])
          output_row[f"joint_pos_{joint_name}_rad"] = (
            joint_pos_rel + layout.default_joint_pos[index]
          )
          output_row[f"joint_vel_{joint_name}_rad_s"] = row[
            f"obs_{layout.joint_vel_start + index}"
          ]
        writer.writerow(output_row)


def _detect_layout(fieldnames: list[str]) -> RobotObservationLayout:
  """Infer the robot from the known actor-observation width."""
  obs_columns = [name for name in fieldnames if name.startswith("obs_")]
  if len(obs_columns) == 122:
    return MARSDOG_LAYOUT
  if len(obs_columns) == 69:
    return GO2_LAYOUT
  raise ValueError(
    "Unsupported actor observation width. Expected 122 for Marsdog or 69 for Go2, "
    f"got {len(obs_columns)}."
  )


def main() -> None:
  args = parse_args()
  extract_joint_states(args.input_file, args.output_file, args.control_dt)
  print(f"Saved Marsdog joint states to {args.output_file.resolve()}")


if __name__ == "__main__":
  main()
