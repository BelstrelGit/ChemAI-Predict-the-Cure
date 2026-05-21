from __future__ import annotations

import numpy as np

from chemai.pipeline import (
    apply_ratio_si,
    inverse_targets,
    kaggle_rmse,
    target_transform_caps,
    transform_targets,
)


def test_kaggle_rmse_matches_manual_average() -> None:
    y_true = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 6.0, 9.0],
        ]
    )
    y_pred = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 3.0],
        ]
    )

    score, details = kaggle_rmse(y_true, y_pred)
    expected_per_target = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

    assert np.isclose(score, expected_per_target.mean())
    assert np.isclose(details["IC50"], expected_per_target[0])
    assert np.isclose(details["CC50"], expected_per_target[1])
    assert np.isclose(details["SI"], expected_per_target[2])


def test_target_transform_roundtrip_for_supported_methods() -> None:
    y = np.array(
        [
            [0.0, 1.0, 2.0],
            [10.0, 100.0, 1000.0],
        ]
    )

    for method in ["raw", "sqrt", "log1p"]:
        transformed = transform_targets(y, method)
        restored = inverse_targets(transformed, method)
        assert np.allclose(restored, y)


def test_inverse_targets_clips_to_training_caps() -> None:
    y_train = np.array(
        [
            [10.0, 100.0, 1000.0],
            [20.0, 200.0, 2000.0],
        ]
    )
    caps = target_transform_caps(y_train, "raw", multiplier=1.0)
    predictions = np.array([[50.0, 500.0, 5000.0]])

    restored = inverse_targets(predictions, "raw", caps)

    assert restored.tolist() == [[20.0, 200.0, 2000.0]]


def test_apply_ratio_si_recomputes_and_caps_si() -> None:
    predictions = np.array(
        [
            [10.0, 50.0, 999.0],
            [0.0, 100.0, 999.0],
        ]
    )

    adjusted = apply_ratio_si(predictions, si_cap=20.0)

    assert adjusted[0, 2] == 5.0
    assert adjusted[1, 2] == 20.0
    assert adjusted[0, 0] == predictions[0, 0]
    assert adjusted[0, 1] == predictions[0, 1]

