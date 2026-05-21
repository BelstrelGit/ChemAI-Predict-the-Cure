# ChemAI experiment summary

- CV: group, n_splits=5, seed=42
- Preset: fast
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: xgboost, catboost, extra_trees, lightgbm
- SI postprocess: direct model prediction
- Exact-match blend alpha: 0.0; matched test rows: 0
- Ensemble OOF score: 548.809840
- Ensemble OOF RMSE: IC50=351.8538, CC50=505.0660, SI=789.5097

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| xgboost | 548.425319 | 550.845729 | 9.8 |
| catboost | 549.313117 | 550.177117 | 17.7 |
| extra_trees | 550.694640 | 551.092626 | 3.0 |
| lightgbm | 552.756483 | 624.593440 | 21.6 |
| hist_gradient | 553.777883 | 556.947551 | 15.3 |
| ridge | 602.543845 | 594.426972 | 0.1 |
