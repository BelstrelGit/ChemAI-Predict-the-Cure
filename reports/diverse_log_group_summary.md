# ChemAI experiment summary

- CV: group, n_splits=5, seed=42
- Preset: full
- Target transform: log1p
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: catboost, svr
- SI postprocess: direct model prediction
- Exact-match blend alpha: 0.0; matched test rows: 0
- Refit full for test prediction: False
- Ensemble OOF score: 543.068633
- Ensemble OOF RMSE: IC50=344.6073, CC50=494.3406, SI=790.2581

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| catboost | 546.975967 | 547.787494 | 38.5 |
| svr | 549.711927 | 621.385507 | 0.7 |
| gradient_boosting | 549.818481 | 550.671687 | 75.4 |
| xgboost | 549.905476 | 552.547241 | 23.9 |
| extra_trees | 550.907085 | 551.296918 | 6.2 |
| lightgbm | 552.883449 | 624.715109 | 52.5 |
| hist_gradient | 556.034981 | 689.415641 | 40.1 |
| kernel_ridge | 567.704879 | 568.523865 | 0.2 |
| knn | 568.487957 | 569.454618 | 0.1 |
