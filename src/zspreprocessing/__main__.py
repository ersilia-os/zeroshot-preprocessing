"""
python -m zspreprocessing <csv_file> --target <column> [--task classification|regression]

Print the PreprocessingProfile for a CSV dataset.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="python -m zspreprocessing",
        description="Profile a CSV dataset and print its preprocessing characteristics.",
    )
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("--target", required=True, help="Name of the target column")
    parser.add_argument(
        "--task",
        default="classification",
        choices=["classification", "regression"],
        help="Task type (default: classification)",
    )
    args = parser.parse_args()

    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        print(f"Error: {e}. Install pandas to use the CLI.", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"Error reading {args.csv!r}: {e}", file=sys.stderr)
        sys.exit(1)

    if args.target not in df.columns:
        print(f"Error: column {args.target!r} not found. Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    y = df[args.target].to_numpy()
    X = df.drop(columns=[args.target]).to_numpy()

    from zspreprocessing import inspect, select_reducer, select_scaler

    profile = inspect(X, y, task=args.task)

    print(repr(profile))
    print()
    print(f"  → scaler:  {select_scaler(profile)}")
    print(f"  → reducer: {select_reducer(profile)}")


if __name__ == "__main__":
    main()
