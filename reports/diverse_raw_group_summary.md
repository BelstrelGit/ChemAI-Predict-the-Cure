# ChemAI experiment summary

- CV: group, n_splits=5, seed=42
- Preset: full
- Target transform: raw
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: kernel_ridge, extra_trees
- SI postprocess: direct model prediction
- Exact-match blend alpha: 0.0; matched test rows: 0
- Refit full for test prediction: False
- Ensemble OOF score: 510.090417
- Ensemble OOF RMSE: IC50=330.5181, CC50=450.0608, SI=749.6924

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| kernel_ridge | 514.462652 | 1544.375843 | 0.3 |
| extra_trees | 520.691134 | 530.134763 | 9.0 |
| catboost | 523.449410 | 1071.556241 | 40.0 |
| knn | 529.882559 | 543.416309 | 0.1 |
| gradient_boosting | 534.523802 | 1164.847458 | 75.5 |
| lightgbm | 536.896518 | 1340.040630 | 58.2 |
| hist_gradient | 538.750409 | 1769.356600 | 44.8 |
| xgboost | 541.907348 | 1418.745788 | 32.4 |
| svr | 615.027110 | 615.181220 | 0.6 |
