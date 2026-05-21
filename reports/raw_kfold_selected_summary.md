# ChemAI experiment summary

- CV: kfold, n_splits=5, seed=42
- Preset: full
- Target transform: raw
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: catboost, kernel_ridge
- SI postprocess: direct model prediction
- Exact-match blend alpha: 0.0; matched test rows: 0
- Refit full for test prediction: False
- Ensemble OOF score: 516.480921
- Ensemble OOF RMSE: IC50=316.6471, CC50=442.6789, SI=790.1168

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| catboost | 520.924981 | 1065.590839 | 37.7 |
| kernel_ridge | 522.462536 | 1572.570478 | 0.2 |
| extra_trees | 532.533608 | 525.667562 | 6.4 |
| knn | 541.235938 | 532.670034 | 0.1 |
| gradient_boosting | 535.155706 | 1289.702597 | 76.0 |
