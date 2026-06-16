"""Offset the root-position Z column in a Marsdog motion CSV.

The Marsdog CSV layout is:
  [0:3] root_pos = X, Y, Z
  [3:7] root_rot = qx, qy, qz, qw
  [7:]  joint positions

By default this script subtracts 0.03 m from column 2 and writes a new CSV,
leaving all other columns exactly as they appear in the input file.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path


def _default_output_path(input_path: Path) -> Path:
  return input_path.with_name(f"{input_path.stem}_z_minus_0p03{input_path.suffix}")


def offset_root_z(
  input_path: Path,
  output_path: Path,
  z_offset: Decimal,
) -> int:
  """Subtract ``z_offset`` from the root-position Z column."""
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
      if len(fields) < 3:
        raise ValueError(
          f"Line {line_no} has {len(fields)} columns, expected at least 3."
        )

      fields[2] = format(Decimal(fields[2]) - z_offset, ".18e")
      dst.write(",".join(fields) + "\n")
      rows_written += 1

  return rows_written


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Subtract an offset from Marsdog motion CSV root_pos Z."
  )
  parser.add_argument(
    "--input-file",
    type=Path,
    default=Path("marsdog_motion.csv"),
    help="Input Marsdog motion CSV.",
  )
  parser.add_argument(
    "--output-file",
    type=Path,
    default=None,
    help="Output CSV. Defaults to '<input>_z_minus_0p03.csv'.",
  )
  parser.add_argument(
    "--offset",
    type=Decimal,
    default=Decimal("-0.03"),
    help="Value to subtract from root_pos Z, in meters.",
  )
  parser.add_argument(
    "--in-place",
    action="store_true",
    help="Overwrite the input file instead of writing a separate output file.",
  )
  args = parser.parse_args()

  input_path = args.input_file
  if not input_path.exists():
    raise FileNotFoundError(input_path)

  if args.in_place:
    output_path = input_path.with_suffix(input_path.suffix + ".tmp")
  else:
    output_path = args.output_file or _default_output_path(input_path)

  rows_written = offset_root_z(input_path, output_path, args.offset)

  if args.in_place:
    output_path.replace(input_path)
    output_path = input_path

  print(f"Wrote {rows_written} rows to {output_path} (root_pos Z -= {args.offset} m).")


if __name__ == "__main__":
  main()
