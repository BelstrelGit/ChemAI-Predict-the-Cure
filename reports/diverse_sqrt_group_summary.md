# ChemAI experiment summary

- CV: group, n_splits=5, seed=42
- Preset: full
- Target transform: sqrt
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: catboost, extra_trees, kernel_ridge
- SI postprocess: direct model prediction
- Exact-match blend alpha: 0.0; matched test rows: 0
- Refit full for test prediction: False
- Ensemble OOF score: 525.599341
- Ensemble OOF RMSE: IC50=332.9061, CC50=465.1116, SI=778.7803

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| catboost | 527.896340 | 532.498942 | 38.2 |
| extra_trees | 528.512905 | 531.585730 | 8.2 |
| kernel_ridge | 532.691688 | 672.277085 | 0.2 |
| gradient_boosting | 533.969998 | 539.135081 | 77.0 |
| xgboost | 536.864185 | 613.047836 | 23.5 |
| hist_gradient | 539.875603 | 772.037914 | 60.4 |
| lightgbm | 540.148016 | 615.402338 | 50.8 |
| knn | 540.346671 | 548.601913 | 0.1 |
| svr | 555.015062 | 628.328933 | 0.6 |
