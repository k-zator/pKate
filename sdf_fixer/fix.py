"""
fix — CLI entry-point for the interactive SDF protonation fixer.

Usage examples::

    python -m sdf_fixer.fix  data/raw/INCORRECT_literature_compilation.sdf
    python -m sdf_fixer.fix  data/raw/INCORRECT_literature_compilation.sdf --ph 7.4 -n 50
    python -m sdf_fixer.fix  data/raw/INCORRECT_literature_compilation.sdf --auto-only
"""

import argparse
import sys

from sdf_fixer.interactive_fixer import fix_sdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix protonation states in an SDF file interactively."
    )
    parser.add_argument("sdf", help="Path to the input SDF file.")
    parser.add_argument(
        "-o", "--out",
        default=None,
        help="Output SDF path (default: FIXED_<input> in same directory).",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="CSV log path (default: FIXLOG_<input>.csv in same directory).",
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="Reference pH for Henderson-Hasselbalch (default: 7.4).",
    )
    parser.add_argument(
        "-n", "--max-records",
        type=int,
        default=0,
        help="Process only the first N records (0 = all).",
    )
    parser.add_argument(
        "--auto-only",
        action="store_true",
        help="Only auto-fix obvious cases; skip interactive prompts.",
    )

    args = parser.parse_args()

    fix_sdf(
        sdf_path=args.sdf,
        out_path=args.out,
        log_path=args.log,
        ph=args.ph,
        max_records=args.max_records,
        auto_only=args.auto_only,
    )


if __name__ == "__main__":
    main()
