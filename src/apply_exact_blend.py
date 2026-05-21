from __future__ import annotations

import argparse

import pandas as pd

from chemai.pipeline import (
    SUBMISSION_TARGET_COLS,
    apply_exact_match_blend,
    create_submission,
    load_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend a submission with exact train descriptor matches.")
    parser.add_argument("--train-path", default="train.csv")
    parser.add_argument("--test-path", default="test.csv")
    parser.add_argument("--sample-submission-path", default="sample_submission.csv")
    parser.add_argument("--input-submission", required=True)
    parser.add_argument("--output-submission", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.train_path, args.test_path, args.sample_submission_path)
    submission = pd.read_csv(args.input_submission)
    predictions = submission[SUBMISSION_TARGET_COLS].to_numpy(dtype=float)
    blended, matched = apply_exact_match_blend(dataset, predictions, alpha=args.alpha)
    create_submission(dataset, blended, args.output_submission)
    print(f"matched_rows={matched}")
    print(f"output={args.output_submission}")


if __name__ == "__main__":
    main()

