from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib")))
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
)

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from sklearn.base import RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR


TRAIN_TARGET_COLS = ["IC50, mM", "CC50, mM", "SI"]
SUBMISSION_TARGET_COLS = ["IC50", "CC50", "SI"]


@dataclass(frozen=True)
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame
    base_feature_cols: list[str]
    feature_cols: list[str]
    dropped_constant_cols: list[str]
    groups: np.ndarray


@dataclass(frozen=True)
class ModelRun:
    name: str
    oof_log: np.ndarray
    test_log: np.ndarray
    direct_score: float
    direct_rmse: dict[str, float]
    ratio_score: float
    ratio_rmse: dict[str, float]
    fit_seconds: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ChemAI models and create a Kaggle submission.")
    parser.add_argument("--train-path", default="train.csv")
    parser.add_argument("--test-path", default="test.csv")
    parser.add_argument("--sample-submission-path", default="sample_submission.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--submission-path", default="submissions/submission.csv")
    parser.add_argument("--report-path", default="reports/experiment_summary.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--cv", choices=["group", "kfold"], default="group")
    parser.add_argument(
        "--models",
        default="auto",
        help="Comma-separated model names, or auto. Examples: auto,ridge,extra_trees,catboost",
    )
    parser.add_argument(
        "--preset",
        choices=["fast", "full"],
        default="fast",
        help="fast keeps runtime modest; full uses larger ensembles where available.",
    )
    parser.add_argument(
        "--target-transform",
        choices=["log1p", "sqrt", "raw"],
        default="log1p",
        help="Transform applied to targets during training.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="How many best OOF models to average.")
    parser.add_argument(
        "--exact-match-alpha",
        type=float,
        default=0.0,
        help="Blend final test predictions with train target means for exact descriptor matches.",
    )
    parser.add_argument(
        "--ratio-si",
        choices=["auto", "always", "never"],
        default="auto",
        help="Whether to set SI = CC50 / IC50 after prediction.",
    )
    parser.add_argument(
        "--refit-full",
        action="store_true",
        help="After CV model selection, refit selected model(s) on all train rows for test prediction.",
    )
    return parser.parse_args(argv)


def load_dataset(
    train_path: str | Path,
    test_path: str | Path,
    sample_submission_path: str | Path,
) -> Dataset:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_submission_path)

    missing_targets = [col for col in TRAIN_TARGET_COLS if col not in train.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in train: {missing_targets}")

    base_feature_cols = [col for col in train.columns if col not in ["index", *TRAIN_TARGET_COLS]]
    test_feature_cols = [col for col in test.columns if col != "index"]
    if base_feature_cols != test_feature_cols:
        raise ValueError("Train and test feature columns do not match in the same order.")

    dropped_constant_cols = [
        col for col in base_feature_cols if train[col].nunique(dropna=False) <= 1
    ]
    feature_cols = [col for col in base_feature_cols if col not in dropped_constant_cols]
    groups = row_hash(train[base_feature_cols]).to_numpy()

    return Dataset(
        train=train,
        test=test,
        sample_submission=sample_submission,
        base_feature_cols=base_feature_cols,
        feature_cols=feature_cols,
        dropped_constant_cols=dropped_constant_cols,
        groups=groups,
    )


def row_hash(df: pd.DataFrame) -> pd.Series:
    return hash_pandas_object(df, index=False).astype("uint64").astype(str)


def kaggle_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    per_target = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    details = {
        col: float(value)
        for col, value in zip(SUBMISSION_TARGET_COLS, per_target, strict=True)
    }
    return float(np.mean(per_target)), details


def transform_targets(y: pd.DataFrame | np.ndarray, method: str) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    values = np.clip(values, 0.0, None)
    if method == "log1p":
        return np.log1p(values)
    if method == "sqrt":
        return np.sqrt(values)
    if method == "raw":
        return values
    raise ValueError(f"Unknown target transform: {method}")


def target_transform_caps(
    y: pd.DataFrame | np.ndarray,
    method: str,
    multiplier: float = 1.1,
) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    caps = np.max(values, axis=0) * multiplier
    return transform_targets(caps.reshape(1, -1), method).reshape(-1)


def inverse_targets(
    y_transformed: np.ndarray,
    method: str,
    caps: np.ndarray | None = None,
) -> np.ndarray:
    y_transformed = np.asarray(y_transformed, dtype=float)
    if caps is not None:
        y_transformed = np.clip(y_transformed, 0.0, np.asarray(caps, dtype=float))
    else:
        y_transformed = np.clip(y_transformed, 0.0, None)
    if method == "log1p":
        y = np.expm1(y_transformed)
    elif method == "sqrt":
        y = y_transformed ** 2
    elif method == "raw":
        y = y_transformed
    else:
        raise ValueError(f"Unknown target transform: {method}")
    y = np.nan_to_num(y, nan=0.0, neginf=0.0, posinf=1.0e12)
    return np.clip(y, 0.0, None)


def log_targets(y: pd.DataFrame | np.ndarray) -> np.ndarray:
    return transform_targets(y, "log1p")


def target_log_caps(y: pd.DataFrame | np.ndarray, multiplier: float = 1.1) -> np.ndarray:
    return target_transform_caps(y, "log1p", multiplier)


def inverse_log_targets(y_log: np.ndarray, log_caps: np.ndarray | None = None) -> np.ndarray:
    return inverse_targets(y_log, "log1p", log_caps)


def apply_ratio_si(y_pred: np.ndarray, si_cap: float | None = None) -> np.ndarray:
    adjusted = np.asarray(y_pred, dtype=float).copy()
    denom = np.clip(adjusted[:, 0], 1.0e-6, None)
    adjusted[:, 2] = adjusted[:, 1] / denom
    if si_cap is not None and math.isfinite(si_cap):
        adjusted[:, 2] = np.clip(adjusted[:, 2], 0.0, si_cap)
    return adjusted


def choose_splits(dataset: Dataset, n_splits: int, cv: str, seed: int):
    x = dataset.train[dataset.feature_cols]
    if cv == "group":
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(x, groups=dataset.groups))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(x))


def make_common_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median", add_indicator=True)


def make_model_factories(seed: int, preset: str) -> dict[str, Callable[[], RegressorMixin]]:
    n_estimators_tree = 450 if preset == "fast" else 900
    max_iter_hgb = 350 if preset == "fast" else 900

    factories: dict[str, Callable[[], RegressorMixin]] = {
        "ridge": lambda: Pipeline(
            steps=[
                ("imputer", make_common_imputer()),
                ("scaler", RobustScaler()),
                ("model", Ridge(alpha=20.0, random_state=seed)),
            ]
        ),
        "elasticnet": lambda: MultiOutputRegressor(
            Pipeline(
                steps=[
                    ("imputer", make_common_imputer()),
                    ("scaler", RobustScaler()),
                    (
                        "model",
                        ElasticNet(
                            alpha=0.002,
                            l1_ratio=0.08,
                            max_iter=50000,
                            random_state=seed,
                        ),
                    ),
                ]
            )
        ),
        "knn": lambda: Pipeline(
            steps=[
                ("imputer", make_common_imputer()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    KNeighborsRegressor(
                        n_neighbors=18,
                        weights="distance",
                        p=2,
                    ),
                ),
            ]
        ),
        "kernel_ridge": lambda: MultiOutputRegressor(
            Pipeline(
                steps=[
                    ("imputer", make_common_imputer()),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        KernelRidge(
                            alpha=0.35,
                            kernel="rbf",
                            gamma=0.004,
                        ),
                    ),
                ]
            )
        ),
        "svr": lambda: MultiOutputRegressor(
            Pipeline(
                steps=[
                    ("imputer", make_common_imputer()),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        SVR(
                            C=7.0,
                            epsilon=0.03,
                            gamma="scale",
                            kernel="rbf",
                        ),
                    ),
                ]
            )
        ),
        "extra_trees": lambda: Pipeline(
            steps=[
                ("imputer", make_common_imputer()),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=n_estimators_tree,
                        max_features=0.75,
                        min_samples_leaf=1,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient": lambda: MultiOutputRegressor(
            Pipeline(
                steps=[
                    ("imputer", make_common_imputer()),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            learning_rate=0.035,
                            max_iter=max_iter_hgb,
                            max_leaf_nodes=15,
                            l2_regularization=0.05,
                            random_state=seed,
                        ),
                    ),
                ]
            )
        ),
        "gradient_boosting": lambda: MultiOutputRegressor(
            Pipeline(
                steps=[
                    ("imputer", make_common_imputer()),
                    (
                        "model",
                        GradientBoostingRegressor(
                            n_estimators=650 if preset == "fast" else 1300,
                            learning_rate=0.025,
                            max_depth=2,
                            min_samples_leaf=6,
                            subsample=0.85,
                            random_state=seed,
                        ),
                    ),
                ]
            )
        ),
    }

    if preset == "full":
        factories["random_forest"] = lambda: Pipeline(
            steps=[
                ("imputer", make_common_imputer()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=900,
                        max_features=0.75,
                        min_samples_leaf=1,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    try:
        from catboost import CatBoostRegressor

        iterations = 1200 if preset == "fast" else 2500
        factories["catboost"] = lambda: Pipeline(
            steps=[
                ("imputer", make_common_imputer()),
                (
                    "model",
                    CatBoostRegressor(
                        loss_function="MultiRMSE",
                        iterations=iterations,
                        learning_rate=0.03,
                        depth=5,
                        l2_leaf_reg=8.0,
                        random_seed=seed,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
        factories["catboost_single"] = lambda: MultiOutputRegressor(
            Pipeline(
                steps=[
                    ("imputer", make_common_imputer()),
                    (
                        "model",
                        CatBoostRegressor(
                            loss_function="RMSE",
                            iterations=iterations,
                            learning_rate=0.03,
                            depth=5,
                            l2_leaf_reg=8.0,
                            random_seed=seed,
                            verbose=False,
                            allow_writing_files=False,
                        ),
                    ),
                ]
            )
        )
        factories["catboost_native"] = lambda: CatBoostRegressor(
            loss_function="MultiRMSE",
            iterations=iterations,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=8.0,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
        factories["catboost_single_native"] = lambda: MultiOutputRegressor(
            CatBoostRegressor(
                loss_function="RMSE",
                iterations=iterations,
                learning_rate=0.03,
                depth=5,
                l2_leaf_reg=8.0,
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            )
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor

        n_estimators = 1000 if preset == "fast" else 2500
        factories["lightgbm"] = lambda: MultiOutputRegressor(
            Pipeline(
                steps=[
                    ("imputer", make_common_imputer()),
                    (
                        "model",
                        LGBMRegressor(
                            objective="regression",
                            n_estimators=n_estimators,
                            learning_rate=0.025,
                            num_leaves=15,
                            min_child_samples=8,
                            subsample=0.85,
                            colsample_bytree=0.75,
                            reg_lambda=3.0,
                            random_state=seed,
                            n_jobs=-1,
                            verbosity=-1,
                        ),
                    ),
                ]
            )
        )
    except Exception:
        pass

    try:
        from xgboost import XGBRegressor

        n_estimators = 900 if preset == "fast" else 2200
        factories["xgboost"] = lambda: MultiOutputRegressor(
            Pipeline(
                steps=[
                    ("imputer", make_common_imputer()),
                    (
                        "model",
                        XGBRegressor(
                            objective="reg:squarederror",
                            n_estimators=n_estimators,
                            learning_rate=0.025,
                            max_depth=3,
                            min_child_weight=2.0,
                            subsample=0.85,
                            colsample_bytree=0.75,
                            reg_lambda=5.0,
                            random_state=seed,
                            n_jobs=-1,
                            tree_method="hist",
                        ),
                    ),
                ]
            )
        )
    except Exception:
        pass

    return factories


def select_model_names(requested: str, factories: dict[str, Callable[[], RegressorMixin]]) -> list[str]:
    if requested == "auto":
        preferred = [
            "catboost",
            "lightgbm",
            "xgboost",
            "extra_trees",
            "hist_gradient",
            "gradient_boosting",
            "knn",
            "kernel_ridge",
            "svr",
            "ridge",
        ]
        names = [name for name in preferred if name in factories]
    else:
        names = [name.strip() for name in requested.split(",") if name.strip()]
        unknown = [name for name in names if name not in factories]
        if unknown:
            available = ", ".join(sorted(factories))
            raise ValueError(f"Unknown or unavailable models: {unknown}. Available: {available}")
    if not names:
        raise ValueError("No models selected.")
    return names


def fit_predict_cv(
    dataset: Dataset,
    model_name: str,
    model_factory: Callable[[], RegressorMixin],
    splits: list[tuple[np.ndarray, np.ndarray]],
    target_transform: str,
) -> ModelRun:
    start = time.time()
    x = dataset.train[dataset.feature_cols]
    x_test = dataset.test[dataset.feature_cols]
    y = dataset.train[TRAIN_TARGET_COLS]
    y_values = y.to_numpy(dtype=float)
    y_fit = transform_targets(y, target_transform)
    transform_caps = target_transform_caps(y_values, target_transform)

    oof_log = np.zeros((len(dataset.train), len(TRAIN_TARGET_COLS)), dtype=float)
    test_log_folds: list[np.ndarray] = []

    print(f"\n[{model_name}]")
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = model_factory()
        model.fit(x.iloc[train_idx], y_fit[train_idx])
        valid_pred_trans = np.asarray(model.predict(x.iloc[valid_idx]), dtype=float)
        test_pred_trans = np.asarray(model.predict(x_test), dtype=float)
        oof_log[valid_idx] = valid_pred_trans
        test_log_folds.append(test_pred_trans)

        valid_pred = inverse_targets(valid_pred_trans, target_transform, transform_caps)
        fold_score, fold_rmse = kaggle_rmse(y_values[valid_idx], valid_pred)
        print(f"  fold {fold}: score={fold_score:.5f} rmse={format_rmse(fold_rmse)}")

    test_log = np.mean(test_log_folds, axis=0)
    direct_pred = inverse_targets(oof_log, target_transform, transform_caps)
    direct_score, direct_rmse = kaggle_rmse(y_values, direct_pred)

    ratio_pred = apply_ratio_si(direct_pred, si_cap=float(y["SI"].max() * 1.1))
    ratio_score, ratio_rmse = kaggle_rmse(y_values, ratio_pred)
    fit_seconds = time.time() - start
    print(
        f"  oof direct={direct_score:.5f} ratio_si={ratio_score:.5f} "
        f"time={fit_seconds:.1f}s"
    )

    return ModelRun(
        name=model_name,
        oof_log=oof_log,
        test_log=test_log,
        direct_score=direct_score,
        direct_rmse=direct_rmse,
        ratio_score=ratio_score,
        ratio_rmse=ratio_rmse,
        fit_seconds=fit_seconds,
    )


def format_rmse(values: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.4f}" for key, value in values.items())


def build_ensemble(
    runs: list[ModelRun],
    y_true: np.ndarray,
    top_k: int,
    ratio_mode: str,
    target_transform: str,
) -> tuple[list[str], np.ndarray, np.ndarray, bool, float, dict[str, float]]:
    ranked = sorted(runs, key=lambda run: min(run.direct_score, run.ratio_score))
    max_k = max(1, min(top_k, len(ranked)))
    transform_caps = target_transform_caps(y_true, target_transform)
    best: tuple[list[str], np.ndarray, np.ndarray, bool, float, dict[str, float]] | None = None

    for k in range(1, max_k + 1):
        selected = ranked[:k]
        selected_names = [run.name for run in selected]
        oof_log = np.mean([run.oof_log for run in selected], axis=0)
        test_log = np.mean([run.test_log for run in selected], axis=0)
        oof_pred = inverse_targets(oof_log, target_transform, transform_caps)
        direct_score, direct_rmse = kaggle_rmse(y_true, oof_pred)

        ratio_pred = apply_ratio_si(oof_pred, si_cap=float(y_true[:, 2].max() * 1.1))
        ratio_score, ratio_rmse = kaggle_rmse(y_true, ratio_pred)

        if ratio_mode == "always":
            use_ratio = True
        elif ratio_mode == "never":
            use_ratio = False
        else:
            use_ratio = ratio_score < direct_score

        score = ratio_score if use_ratio else direct_score
        rmse = ratio_rmse if use_ratio else direct_rmse
        candidate = (selected_names, oof_log, test_log, use_ratio, score, rmse)
        if best is None or candidate[4] < best[4]:
            best = candidate

    if best is None:
        raise ValueError("No ensemble candidate was built.")
    return best


def refit_selected_models(
    dataset: Dataset,
    selected_names: list[str],
    factories: dict[str, Callable[[], RegressorMixin]],
    target_transform: str,
) -> np.ndarray:
    x = dataset.train[dataset.feature_cols]
    x_test = dataset.test[dataset.feature_cols]
    y_fit = transform_targets(dataset.train[TRAIN_TARGET_COLS], target_transform)
    test_predictions = []
    for name in selected_names:
        print(f"  refit full: {name}")
        model = factories[name]()
        model.fit(x, y_fit)
        test_predictions.append(np.asarray(model.predict(x_test), dtype=float))
    return np.mean(test_predictions, axis=0)


def exact_match_means(dataset: Dataset) -> pd.DataFrame:
    keyed = dataset.train[["index", *TRAIN_TARGET_COLS]].copy()
    keyed["_key"] = row_hash(dataset.train[dataset.base_feature_cols]).to_numpy()
    return keyed.groupby("_key", sort=False)[TRAIN_TARGET_COLS].mean()


def apply_exact_match_blend(dataset: Dataset, predictions: np.ndarray, alpha: float) -> tuple[np.ndarray, int]:
    if alpha <= 0:
        return predictions, 0
    means = exact_match_means(dataset)
    test_keys = row_hash(dataset.test[dataset.base_feature_cols]).to_numpy()
    blended = predictions.copy()
    matched = 0
    for row_idx, key in enumerate(test_keys):
        if key not in means.index:
            continue
        target_mean = means.loc[key, TRAIN_TARGET_COLS].to_numpy(dtype=float)
        blended[row_idx] = (1.0 - alpha) * blended[row_idx] + alpha * target_mean
        matched += 1
    return blended, matched


def create_submission(dataset: Dataset, predictions: np.ndarray, path: str | Path) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission = dataset.sample_submission.copy()
    submission["index"] = dataset.test["index"].to_numpy()
    for train_col, sub_col in zip(TRAIN_TARGET_COLS, SUBMISSION_TARGET_COLS, strict=True):
        idx = TRAIN_TARGET_COLS.index(train_col)
        submission[sub_col] = predictions[:, idx]
    submission.to_csv(path, index=False)
    return submission


def write_outputs(
    dataset: Dataset,
    runs: list[ModelRun],
    selected_names: list[str],
    use_ratio: bool,
    ensemble_score: float,
    ensemble_rmse: dict[str, float],
    exact_matches: int,
    args: argparse.Namespace,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)

    cv_rows = []
    for run in runs:
        row = {
            "model": run.name,
            "direct_score": run.direct_score,
            "ratio_score": run.ratio_score,
            "fit_seconds": run.fit_seconds,
        }
        row.update({f"direct_rmse_{k}": v for k, v in run.direct_rmse.items()})
        row.update({f"ratio_rmse_{k}": v for k, v in run.ratio_rmse.items()})
        cv_rows.append(row)
    cv_rows.append(
        {
            "model": "ensemble",
            "direct_score": ensemble_score if not use_ratio else np.nan,
            "ratio_score": ensemble_score if use_ratio else np.nan,
            "fit_seconds": np.nan,
            **{f"direct_rmse_{k}": v for k, v in ensemble_rmse.items() if not use_ratio},
            **{f"ratio_rmse_{k}": v for k, v in ensemble_rmse.items() if use_ratio},
        }
    )
    pd.DataFrame(cv_rows).to_csv(output_dir / "cv_results.csv", index=False)

    metadata = {
        "seed": args.seed,
        "cv": args.cv,
        "n_splits": args.n_splits,
        "preset": args.preset,
        "target_transform": args.target_transform,
        "models": [run.name for run in runs],
        "selected_ensemble_models": selected_names,
        "use_ratio_si": use_ratio,
        "ensemble_score": ensemble_score,
        "ensemble_rmse": ensemble_rmse,
        "dropped_constant_cols": dataset.dropped_constant_cols,
        "exact_match_alpha": args.exact_match_alpha,
        "exact_match_test_rows": exact_matches,
        "refit_full": args.refit_full,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report_lines = [
        "# ChemAI experiment summary",
        "",
        f"- CV: {args.cv}, n_splits={args.n_splits}, seed={args.seed}",
        f"- Preset: {args.preset}",
        f"- Target transform: {args.target_transform}",
        f"- Feature columns: {len(dataset.feature_cols)} used, {len(dataset.dropped_constant_cols)} constants dropped",
        f"- Selected ensemble: {', '.join(selected_names)}",
        f"- SI postprocess: {'ratio CC50/IC50' if use_ratio else 'direct model prediction'}",
        f"- Exact-match blend alpha: {args.exact_match_alpha}; matched test rows: {exact_matches}",
        f"- Refit full for test prediction: {args.refit_full}",
        f"- Ensemble OOF score: {ensemble_score:.6f}",
        f"- Ensemble OOF RMSE: {format_rmse(ensemble_rmse)}",
        "",
        "## Model CV",
        "",
        "| model | direct score | ratio-SI score | seconds |",
        "|---|---:|---:|---:|",
    ]
    for run in sorted(runs, key=lambda item: min(item.direct_score, item.ratio_score)):
        report_lines.append(
            f"| {run.name} | {run.direct_score:.6f} | {run.ratio_score:.6f} | {run.fit_seconds:.1f} |"
        )
    Path(args.report_path).write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    np.random.seed(args.seed)

    dataset = load_dataset(args.train_path, args.test_path, args.sample_submission_path)
    print("Dataset")
    print(f"  train={dataset.train.shape} test={dataset.test.shape}")
    print(f"  features={len(dataset.feature_cols)} dropped_constants={len(dataset.dropped_constant_cols)}")
    print(f"  missing_train={int(dataset.train.isna().sum().sum())} missing_test={int(dataset.test.isna().sum().sum())}")
    print(f"  target_transform={args.target_transform}")

    factories = make_model_factories(args.seed, args.preset)
    model_names = select_model_names(args.models, factories)
    print(f"Models: {', '.join(model_names)}")

    splits = choose_splits(dataset, args.n_splits, args.cv, args.seed)
    runs = [
        fit_predict_cv(dataset, name, factories[name], splits, args.target_transform)
        for name in model_names
    ]

    y_true = dataset.train[TRAIN_TARGET_COLS].to_numpy(dtype=float)
    selected_names, ensemble_oof_log, ensemble_test_log, use_ratio, ensemble_score, ensemble_rmse = (
        build_ensemble(runs, y_true, args.top_k, args.ratio_si, args.target_transform)
    )
    if args.refit_full:
        print("\nRefit selected model(s) on all train rows")
        ensemble_test_log = refit_selected_models(
            dataset,
            selected_names,
            factories,
            args.target_transform,
        )

    final_predictions = inverse_targets(
        ensemble_test_log,
        args.target_transform,
        target_transform_caps(y_true, args.target_transform),
    )
    if use_ratio:
        final_predictions = apply_ratio_si(final_predictions, si_cap=float(y_true[:, 2].max() * 1.1))
    final_predictions, exact_matches = apply_exact_match_blend(
        dataset,
        final_predictions,
        alpha=args.exact_match_alpha,
    )

    submission = create_submission(dataset, final_predictions, args.submission_path)
    write_outputs(
        dataset=dataset,
        runs=runs,
        selected_names=selected_names,
        use_ratio=use_ratio,
        ensemble_score=ensemble_score,
        ensemble_rmse=ensemble_rmse,
        exact_matches=exact_matches,
        args=args,
    )

    print("\nFinal")
    print(f"  selected={', '.join(selected_names)}")
    print(f"  use_ratio_si={use_ratio}")
    print(f"  oof_score={ensemble_score:.6f} rmse={format_rmse(ensemble_rmse)}")
    print(f"  exact_match_blended_rows={exact_matches}")
    print(f"  submission={args.submission_path} rows={len(submission)}")


if __name__ == "__main__":
    run()
