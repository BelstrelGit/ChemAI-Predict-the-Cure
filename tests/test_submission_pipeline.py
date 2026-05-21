from __future__ import annotations

import numpy as np
import pandas as pd

from apply_clipping import main as apply_clipping_main
from chemai.pipeline import (
    SUBMISSION_TARGET_COLS,
    create_submission,
    load_dataset,
    run,
)
from conftest import PROJECT_ROOT, require_competition_data


def test_create_submission_preserves_required_format(tmp_path) -> None:
    require_competition_data()
    dataset = load_dataset(
        PROJECT_ROOT / "train.csv",
        PROJECT_ROOT / "test.csv",
        PROJECT_ROOT / "sample_submission.csv",
    )
    predictions = np.column_stack(
        [
            np.full(len(dataset.test), 1.0),
            np.full(len(dataset.test), 2.0),
            np.full(len(dataset.test), 3.0),
        ]
    )
    submission_path = tmp_path / "submission.csv"

    submission = create_submission(dataset, predictions, submission_path)
    saved = pd.read_csv(submission_path)

    assert list(submission.columns) == ["index", *SUBMISSION_TARGET_COLS]
    assert list(saved.columns) == ["index", *SUBMISSION_TARGET_COLS]
    assert len(saved) == len(dataset.test)
    assert saved["index"].equals(dataset.test["index"])
    assert saved[SUBMISSION_TARGET_COLS].notna().all().all()
    assert (saved[SUBMISSION_TARGET_COLS] >= 0).all().all()


def test_ridge_smoke_pipeline_creates_valid_submission(tmp_path) -> None:
    require_competition_data()
    submission_path = tmp_path / "smoke_submission.csv"
    report_path = tmp_path / "smoke_report.md"
    output_dir = tmp_path / "outputs"

    run(
        [
            "--train-path",
            str(PROJECT_ROOT / "train.csv"),
            "--test-path",
            str(PROJECT_ROOT / "test.csv"),
            "--sample-submission-path",
            str(PROJECT_ROOT / "sample_submission.csv"),
            "--models",
            "ridge",
            "--preset",
            "fast",
            "--cv",
            "group",
            "--n-splits",
            "3",
            "--target-transform",
            "log1p",
            "--submission-path",
            str(submission_path),
            "--report-path",
            str(report_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    submission = pd.read_csv(submission_path)

    assert list(submission.columns) == ["index", *SUBMISSION_TARGET_COLS]
    assert len(submission) == 250
    assert submission["index"].is_unique
    assert submission[SUBMISSION_TARGET_COLS].notna().all().all()
    assert (submission[SUBMISSION_TARGET_COLS] >= 0).all().all()
    assert report_path.exists()
    assert (output_dir / "cv_results.csv").exists()
    assert (output_dir / "metadata.json").exists()


def test_apply_clipping_cli_clips_target_tails(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame(
        {
            "index": [0, 1],
            "IC50": [10.0, 3000.0],
            "CC50": [20.0, 4000.0],
            "SI": [30.0, 800.0],
        }
    ).to_csv(input_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "apply_clipping.py",
            "--input-submission",
            str(input_path),
            "--output-submission",
            str(output_path),
            "--ic50-cap",
            "2000",
            "--cc50-cap",
            "3200",
            "--si-cap",
            "500",
        ],
    )

    apply_clipping_main()
    clipped = pd.read_csv(output_path)

    assert clipped["IC50"].tolist() == [10.0, 2000.0]
    assert clipped["CC50"].tolist() == [20.0, 3200.0]
    assert clipped["SI"].tolist() == [30.0, 500.0]
