"""Filter Marsdog retargeted CSVs down to the current 21 command joints.

Supported input layouts:
  - 30 columns from dison.bvh:
    [0:7] root pose = x, y, z, qx, qy, qz, qw
    [7:30] MuJoCo hinge qpos, including passive rear tarsus joints.
  - 28 columns already filtered to:
    root pose + MARSDOG_JOINT_NAMES order.

The output is always 28 columns and can be passed directly to
``mjlab.scripts.csv_to_npz``.  The rear tarsus joints are dropped because they
are reconstructed by MuJoCo equality constraints; the front tarsus joints are
kept and moved to the end to match ``MARSDOG_JOINT_NAMES``.

Usage:
  uv run python scripts/tools/filter_marsdog_active_dofs.py \
    --input-file src/mjlab/csv/dison.csv \
    --output-file src/mjlab/csv/dison_active.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

DISON_COLUMN_COUNT = 30
ACTIVE_COLUMN_COUNT = 28

# dison.bvh CSV layout:
#   root: 0-6
#   qpos: rl hip/thigh/calf/tarsus, rr hip/thigh/calf/tarsus, waist, head,
#         fl hip/thigh/calf/tarsus, fr hip/thigh/calf/tarsus
#
# csv_to_npz.py writes these values into MARSDOG_JOINT_NAMES, whose current
# order keeps front tarsus joints at the end.  This is therefore not a plain
# deletion of columns 10 and 14; it is a deletion plus reorder.
DISON_TO_ACTIVE_COLUMN_INDICES: tuple[int, ...] = (
  0,
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  11,
  12,
  13,
  15,
  16,
  17,
  18,
  19,
  20,
  21,
  22,
  23,
  24,
  26,
  27,
  28,
  25,
  29,
)

ACTIVE_COLUMN_NAMES: tuple[str, ...] = (
  "root_x",
  "root_y",
  "root_z",
  "root_qx",
  "root_qy",
  "root_qz",
  "root_qw",
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
  "fl_tarsus_joint",
  "fr_tarsus_joint",
)

assert len(DISON_TO_ACTIVE_COLUMN_INDICES) == ACTIVE_COLUMN_COUNT
assert len(ACTIVE_COLUMN_NAMES) == ACTIVE_COLUMN_COUNT


def _default_output_path(input_path: Path) -> Path:
  return input_path.with_name(f"{input_path.stem}_active{input_path.suffix}")


def filter_active_dofs(input_path: Path, output_path: Path) -> int:
  """Write root pose plus joints in ``MARSDOG_JOINT_NAMES`` order."""
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
        # The expected order is printed by --print-columns.
        selected = fields
      elif len(fields) == DISON_COLUMN_COUNT:
        selected = [fields[idx] for idx in DISON_TO_ACTIVE_COLUMN_INDICES]
      else:
        raise ValueError(
          f"Line {line_no} has {len(fields)} columns, expected "
          f"{DISON_COLUMN_COUNT} dison columns or "
          f"{ACTIVE_COLUMN_COUNT} active columns."
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
    default=Path("src/mjlab/csv/dison.csv"),
    help="Input Marsdog CSV with 30 dison columns or 28 filtered columns.",
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
