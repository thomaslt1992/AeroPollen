import argparse
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .ps_methods.methods import calculate_ps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aeropollen",
        description="Calculate pollen-season characteristics from daily CSV data.",
    )
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument(
        "--method", choices=("percentage", "clinical", "moving", "grains"),
        default="percentage",
    )
    parser.add_argument(
        "--species", nargs="+", metavar="COLUMN",
        help="Pollen column names to analyze (default: all non-date columns).",
    )
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--clinical-pollen-type")
    parser.add_argument("--th-day", type=float, default=100)
    parser.add_argument("--perc", type=float, default=95)
    parser.add_argument("--man", type=int, default=11)
    parser.add_argument("--th-ma", type=float, default=5)
    parser.add_argument("--window-grains", type=int, default=5)
    parser.add_argument("--th-pollen", type=float, default=10)
    parser.add_argument("--n-clinical", type=int, default=5)
    parser.add_argument("--window-clinical", type=int, default=7)
    parser.add_argument("--th-sum", type=float, default=100)
    parser.add_argument("--interpolation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--int-method", default="linear")
    parser.add_argument("--maxdays", type=int, default=30)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        data = pd.read_csv(args.input)
        names = {str(column).lower(): column for column in data.columns}
        date_column = names.get(args.date_column.lower())
        if date_column is None:
            parser.error(f"Date column '{args.date_column}' not found.")
        if args.species:
            missing = [name for name in args.species if name not in data.columns]
            if missing:
                parser.error(f"Missing pollen columns: {', '.join(missing)}")
            selected = list(dict.fromkeys(args.species))
        else:
            selected = [col for col in data.columns if col != date_column]
        if not selected:
            parser.error("No pollen columns found.")
        if date_column != "date":
            data = data.rename(columns={date_column: "date"})
        columns = ["date", *selected]
        result = calculate_ps(
            data[columns],
            method=args.method,
            th_day=args.th_day,
            perc=args.perc,
            man=args.man,
            th_ma=args.th_ma,
            window_grains=args.window_grains,
            th_pollen=args.th_pollen,
            n_clinical=args.n_clinical,
            window_clinical=args.window_clinical,
            th_sum=args.th_sum,
            clinical_pollen_type=args.clinical_pollen_type,
            interpolation=args.interpolation,
            int_method=args.int_method,
            maxdays=args.maxdays,
        )
        if args.output is None:
            result.to_csv(sys.stdout, index=False)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.output, index=False)
            print(f"Saved {len(result)} rows to {args.output}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"aeropollen: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
