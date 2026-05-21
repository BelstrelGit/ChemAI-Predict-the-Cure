from __future__ import annotations

import argparse

import pandas as pd


TARGET_COLUMNS = ["IC50", "CC50", "SI"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clip submission target tails.")
    parser.add_argument("--input-submission", required=True)
    parser.add_argument("--output-submission", required=True)
    parser.add_argument("--ic50-cap", type=float, default=None)
    parser.add_argument("--cc50-cap", type=float, default=None)
    parser.add_argument("--si-cap", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission = pd.read_csv(args.input_submission)
    expected_columns = ["index", *TARGET_COLUMNS]
    if list(submission.columns) != expected_columns:
        raise ValueError(f"Expected columns {expected_columns}, got {list(submission.columns)}")

    caps = {
        "IC50": args.ic50_cap,
        "CC50": args.cc50_cap,
        "SI": args.si_cap,
    }
    for column, cap in caps.items():
        submission[column] = submission[column].clip(lower=0)
        if cap is not None:
            submission[column] = submission[column].clip(upper=cap)

    submission.to_csv(args.output_submission, index=False)
    print(f"output={args.output_submission}")


if __name__ == "__main__":
    main()

