from __future__ import annotations

import numpy as np

from chemai.pipeline import TRAIN_TARGET_COLS, load_dataset
from conftest import PROJECT_ROOT, require_competition_data


def test_dataset_contract_matches_competition_files() -> None:
    require_competition_data()
    dataset = load_dataset(
        PROJECT_ROOT / "train.csv",
        PROJECT_ROOT / "test.csv",
        PROJECT_ROOT / "sample_submission.csv",
    )

    assert dataset.train.shape[0] == 751
    assert dataset.test.shape[0] == 250
    assert len(dataset.base_feature_cols) == 210
    assert len(dataset.feature_cols) == 192
    assert len(dataset.dropped_constant_cols) == 18

    assert dataset.train["index"].is_unique
    assert dataset.test["index"].is_unique
    assert dataset.sample_submission["index"].equals(dataset.test["index"])
    assert list(dataset.sample_submission.columns) == ["index", "IC50", "CC50", "SI"]

    for target in TRAIN_TARGET_COLS:
        assert target in dataset.train.columns
        assert target not in dataset.feature_cols


def test_si_is_ratio_of_cc50_to_ic50_in_train() -> None:
    require_competition_data()
    dataset = load_dataset(
        PROJECT_ROOT / "train.csv",
        PROJECT_ROOT / "test.csv",
        PROJECT_ROOT / "sample_submission.csv",
    )

    ratio = dataset.train["CC50, mM"] / dataset.train["IC50, mM"]
    assert np.allclose(ratio, dataset.train["SI"], rtol=1e-5, atol=1e-5)


def test_no_target_leakage_in_test_features() -> None:
    require_competition_data()
    dataset = load_dataset(
        PROJECT_ROOT / "train.csv",
        PROJECT_ROOT / "test.csv",
        PROJECT_ROOT / "sample_submission.csv",
    )

    test_feature_cols = [column for column in dataset.test.columns if column != "index"]
    assert test_feature_cols == dataset.base_feature_cols
    assert not set(TRAIN_TARGET_COLS).intersection(dataset.test.columns)
