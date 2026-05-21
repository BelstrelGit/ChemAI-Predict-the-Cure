# ChemAI experiment summary

- CV: kfold, n_splits=5, seed=42
- Preset: fast
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: hist_gradient, lightgbm, xgboost, catboost
- SI postprocess: direct model prediction
- Exact-match blend alpha: 0.0; matched test rows: 0
- Refit full for test prediction: True
- Ensemble OOF score: 536.437425
- Ensemble OOF RMSE: IC50=339.5506, CC50=490.8850, SI=778.8766

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| hist_gradient | 538.134148 | 539.578382 | 18.8 |
| lightgbm | 538.259040 | 539.521163 | 38.7 |
| xgboost | 538.804704 | 540.765109 | 10.3 |
| catboost | 540.806254 | 542.422468 | 35.9 |
| extra_trees | 543.229427 | 544.373131 | 3.1 |
| ridge | 592.918765 | 725.650426 | 0.1 |
