"""Filter Marsdog retargeted CSVs down to the 21 actuated joints.

The full retargeted CSV layout is expected to be:
  [0:7]  root pose = x, y, z, qx, qy, qz, qw
  [7:52] 45 joint columns, including passive tail and tarsus joints

Training only actuates 21 joints. Passive joints are reconstructed by MuJoCo
equality constraints when ``csv_to_npz.py`` writes the active joints to the
simulation and calls ``sim.forward()``.

Usage:
  uv run python scripts/tools/filter_marsdog_active_dofs.py \
    --input-file marsdog_walk.csv \
    --output-file marsdog_walk_active.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

FULL_COLUMN_COUNT = 52
ACTIVE_COLUMN_COUNT = 28

# Keep root pose and the 21 actuated joints. Indices follow the retargeted CSV
# table: root columns 0-6, tail1 at 7-8, active robot joints at 31-33, 35-37,
# and 39-51. Passive tail2-tail12 and rear tarsus columns are dropped.
ACTIVE_COLUMN_INDICES: tuple[int, ...] = (
  0,
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  31,
  32,
  33,
  35,
  36,
  37,
  39,
  40,
  41,
  42,
  43,
  44,
  45,
  46,
  47,
  48,
  49,
  50,
  51,
)

ACTIVE_COLUMN_NAMES: tuple[str, ...] = (
  "root_x",
  "root_y",
  "root_z",
  "root_qx",
  "root_qy",
  "root_qz",
  "root_qw",
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


def _default_output_path(input_path: Path) -> Path:
  return input_path.with_name(f"{input_path.stem}_active{input_path.suffix}")


def filter_active_dofs(input_path: Path, output_path: Path) -> int:
  """Write a CSV with only root pose and the 21 actuated Marsdog joints."""
  rows_written = 0
  with (
    input_path.open("r", encoding="utf-8") as src,
    output_path.open("w", encoding="utf-8", newline="") as dst,
  ):
    for line_no, line in enumerate(src, start=1):
      stripped = line.rstrip("\n")
      if not stripped:
        dst.write(line)
        continue

      fields = stripped.split(",")
      if len(fields) == ACTIVE_COLUMN_COUNT:
        # Already filtered. Preserve the row so rerunning the tool is harmless.
        selected = fields
      elif len(fields) == FULL_COLUMN_COUNT:
        selected = [fields[idx] for idx in ACTIVE_COLUMN_INDICES]
      else:
        raise ValueError(
          f"Line {line_no} has {len(fields)} columns, expected "
          f"{FULL_COLUMN_COUNT} full columns or {ACTIVE_COLUMN_COUNT} active columns."
        )

      dst.write(",".join(selected) + "\n")
      rows_written += 1

  return rows_written


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Keep only root pose and 21 actuated joints in a Marsdog CSV."
  )
  parser.add_argument(
    "--input-file",
    type=Path,
    default=Path("marsdog_walk.csv"),
    help="Input Marsdog retargeted CSV with 52 columns.",
  )
  parser.add_argument(
    "--output-file",
    type=Path,
    default=None,
    help="Output CSV. Defaults to '<input>_active.csv'.",
  )
  parser.add_argument(
    "--in-place",
    action="store_true",
    help="Overwrite the input file instead of writing a separate output file.",
  )
  parser.add_argument(
    "--print-columns",
    action="store_true",
    help="Print the kept output column order and exit.",
  )
  args = parser.parse_args()

  if args.print_columns:
    for idx, name in enumerate(ACTIVE_COLUMN_NAMES):
      print(f"{idx}: {name}")
    return

  input_path = args.input_file
  if not input_path.exists():
    raise FileNotFoundError(input_path)

  if args.in_place:
    output_path = input_path.with_suffix(input_path.suffix + ".tmp")
  else:
    output_path = args.output_file or _default_output_path(input_path)

  rows_written = filter_active_dofs(input_path, output_path)

  if args.in_place:
    output_path.replace(input_path)
    output_path = input_path

  print(
    f"Wrote {rows_written} rows to {output_path} "
    f"({ACTIVE_COLUMN_COUNT} columns: root + 21 actuated joints)."
  )


if __name__ == "__main__":
  main()
