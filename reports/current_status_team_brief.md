# ChemAI current status brief

## Current best result

Best public Kaggle score so far:

- `submission_diverse_raw_group_pubshape_clip.csv` -> `277.59118`

This is a strong improvement over the first good CatBoost submission:

- `submission_catboost_full_group.csv` -> `303.61057`

And over the first boosting ensemble:

- `submission_full_group_boosting.csv` -> `299.48276`

## Task and metric

We solve a tabular regression problem on molecular descriptors.

Targets:

- `IC50`
- `CC50`
- `SI`

Competition metric:

```text
score = (RMSE(IC50) + RMSE(CC50) + RMSE(SI)) / 3
```

The metric is calculated on the original target scale, not on log-transformed
values.

## Data observations

Important findings:

- Train has 751 rows, test has 250 rows.
- There are 210 numeric molecular descriptor features.
- Some descriptor columns contain missing values.
- 18 features are constant and can be removed.
- Targets are highly skewed, especially `SI`.
- In train, `SI` is almost exactly `CC50 / IC50`.
- Some rows have identical descriptors, but their targets can be very different.

The last point matters: exact descriptor matches do not guarantee identical
biological activity.

## Pipeline

The pipeline:

1. Loads `train.csv`, `test.csv`, and `sample_submission.csv`.
2. Drops constant features.
3. Handles missing values with median imputation and missing-value indicators.
4. Trains models with cross-validation.
5. Computes the Kaggle-like metric locally.
6. Averages fold predictions.
7. Applies optional post-processing.
8. Saves a submission CSV.

Main code:

- `src/chemai/pipeline.py`
- `src/train.py`

## Validation

We used `GroupKFold` by exact descriptor hashes.

Reason:

- There are duplicate descriptor rows.
- We wanted to avoid the same descriptor vector appearing in both train and
  validation folds.

This validation is conservative. Public Kaggle score is much better than local
GroupKFold score, so public test appears easier than this strict validation.

## Models tried

Models that were tried:

- `CatBoost`
- `LightGBM`
- `XGBoost`
- `ExtraTreesRegressor`
- `RandomForestRegressor`
- `HistGradientBoostingRegressor`
- `GradientBoostingRegressor`
- `KernelRidge`
- `SVR`
- `KNN`
- `Ridge`
- `ElasticNet`

Best current modeling core:

- `KernelRidge`
- `ExtraTreesRegressor`

## Current best model idea

The current best submission is based on:

```text
features
  -> KernelRidge trained on raw targets
  -> ExtraTreesRegressor trained on raw targets
  -> average predictions
  -> clip high tails
  -> submission
```

Both models predict the original target values directly:

```text
raw target = IC50 / CC50 / SI as given in train
```

No `log1p` or `sqrt` target transform is used in the current best approach.

## Why KernelRidge and ExtraTrees

`KernelRidge`:

- A regularized regression model with a kernel function.
- Can model nonlinear relationships in a smooth way.
- Gave strong raw-scale predictions.

`ExtraTreesRegressor`:

- An ensemble of many randomized decision trees.
- Captures nonlinear feature interactions.
- Has different error patterns from KernelRidge.

They work well together because their errors are not identical. Averaging them
reduced the overall error.

## What averaging predictions means

Each model predicts `IC50`, `CC50`, and `SI`.

Example:

```text
KernelRidge:
IC50 = 100, CC50 = 500, SI = 20

ExtraTrees:
IC50 = 140, CC50 = 460, SI = 30
```

Final averaged prediction:

```text
IC50 = (100 + 140) / 2 = 120
CC50 = (500 + 460) / 2 = 480
SI   = (20 + 30) / 2 = 25
```

This is an ensemble. The goal is to reduce individual model errors.

## Target transforms tried

We tested three ways to train on targets:

### log1p

```text
model target = log(1 + target)
```

This compresses large values strongly.

### sqrt

```text
model target = sqrt(target)
```

This compresses large values, but less aggressively than `log1p`.

### raw

```text
model target = target
```

The model learns directly on the same scale as Kaggle RMSE.

Result:

- `log1p` was stable but too conservative.
- `sqrt` improved results.
- `raw` was best after clipping extreme predictions.

## Clipping high tails

Raw models gave better predictions, but sometimes produced very large values.
RMSE heavily penalizes large mistakes, so we clipped extreme predictions.

Current best clipping:

```text
IC50 <= 2000
CC50 <= 3200
SI   <= 500
```

This means:

```text
if IC50 > 2000, set IC50 = 2000
if CC50 > 3200, set CC50 = 3200
if SI > 500, set SI = 500
```

Only rare high predictions are affected.

Why this helped:

- `submission_diverse_raw_group_si500.csv` scored `281.88337`.
- `submission_diverse_raw_group_pubshape_clip.csv` scored `277.59118`.

So clipping `IC50`/`CC50` tails, not just `SI`, helped public score.

## Strategies that did not work

### exact-match blending

We found test rows whose descriptors exactly matched train rows. We tried
blending predictions with the mean target of matching train rows.

Formula:

```text
new_prediction = (1 - alpha) * model_prediction + alpha * train_match_mean
```

Examples:

- `exact050` means `alpha = 0.50`.
- `exact100` means `alpha = 1.00`.

This worsened public score. Conclusion: exact descriptor matches are not
reliable target matches.

### refit-full

After CV, we tried retraining models on all train rows before predicting test.

This also worsened public score. Fold-averaged predictions were more stable.

### SI = CC50 / IC50

Since train has `SI ~= CC50 / IC50`, we tried computing `SI` from predicted
`CC50` and `IC50`.

This usually worsened results because small errors in `IC50` can create very
large errors in `SI`.

### separate CatBoost per target

We trained separate CatBoost models:

- one for `IC50`
- one for `CC50`
- one for `SI`

This was slower and did not improve public score.

## Current interpretation

The main insight is:

```text
Kaggle public score prefers raw-scale modeling, but raw predictions need tail control.
```

The best current recipe is:

```text
raw targets + KernelRidge/ExtraTrees ensemble + clipping high predictions
```

## Next experiments

Prepared next submissions focus on controlled changes around the current best:

- tune `IC50` cap around `1500-2200`;
- tune `SI` cap around `350-650`;
- tune `CC50` cap;
- change weights between `ExtraTrees` and `KernelRidge`;
- test multi-seed ExtraTrees averaging.

Detailed next submission plan:

- `reports/next_submission_plan.md`

