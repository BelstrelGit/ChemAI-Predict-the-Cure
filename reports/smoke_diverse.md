# ChemAI experiment summary

- CV: group, n_splits=3, seed=42
- Preset: fast
- Target transform: log1p
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: svr, knn
- SI postprocess: direct model prediction
- Exact-match blend alpha: 0.0; matched test rows: 0
- Refit full for test prediction: False
- Ensemble OOF score: 554.611824
- Ensemble OOF RMSE: IC50=360.0506, CC50=514.3135, SI=789.4713

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| svr | 555.549400 | 622.344449 | 0.4 |
| knn | 569.096125 | 570.066070 | 0.2 |
| kernel_ridge | 573.791742 | 647.295084 | 0.1 |
