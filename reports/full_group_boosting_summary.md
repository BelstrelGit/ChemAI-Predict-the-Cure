# ChemAI experiment summary

- CV: group, n_splits=5, seed=42
- Preset: full
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: catboost, xgboost, extra_trees, lightgbm
- SI postprocess: direct model prediction
- Exact-match blend alpha: 0.0; matched test rows: 0
- Ensemble OOF score: 548.227031
- Ensemble OOF RMSE: IC50=351.6207, CC50=503.5897, SI=789.4707

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| catboost | 546.975967 | 547.787494 | 38.4 |
| xgboost | 549.905476 | 552.547241 | 24.0 |
| extra_trees | 550.907085 | 551.296918 | 7.5 |
| lightgbm | 552.883449 | 624.715109 | 52.6 |
| hist_gradient | 556.034981 | 689.415641 | 44.9 |
| ridge | 602.543845 | 594.426972 | 0.1 |
