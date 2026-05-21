# ChemAI experiment summary

- CV: group, n_splits=5, seed=42
- Preset: fast
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: catboost, extra_trees, hist_gradient
- SI postprocess: direct model prediction
- Exact-match blend alpha: 0.0; matched test rows: 0
- Ensemble OOF score: 549.360461
- Ensemble OOF RMSE: IC50=351.3728, CC50=507.8365, SI=788.8721

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| catboost | 549.313117 | 550.177117 | 16.6 |
| extra_trees | 550.694640 | 551.092626 | 3.1 |
| hist_gradient | 553.777883 | 556.947551 | 14.8 |
| ridge | 602.543845 | 594.426972 | 0.1 |
