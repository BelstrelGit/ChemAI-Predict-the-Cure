# ChemAI experiment summary

- CV: group, n_splits=5, seed=42
- Preset: full
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: catboost_single_native, catboost_single, catboost
- SI postprocess: direct model prediction
- Exact-match blend alpha: 0.0; matched test rows: 0
- Refit full for test prediction: True
- Ensemble OOF score: 545.751496
- Ensemble OOF RMSE: IC50=351.1994, CC50=496.5627, SI=789.4924

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| catboost_single_native | 546.616071 | 547.225429 | 375.7 |
| catboost_single | 546.783250 | 547.160170 | 977.9 |
| catboost | 546.975967 | 547.787494 | 38.7 |
| catboost_native | 547.524541 | 548.325382 | 823.0 |
