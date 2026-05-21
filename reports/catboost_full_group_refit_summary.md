# ChemAI experiment summary

- CV: group, n_splits=5, seed=42
- Preset: full
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: catboost
- SI postprocess: direct model prediction
- Exact-match blend alpha: 0.0; matched test rows: 0
- Refit full for test prediction: True
- Ensemble OOF score: 546.975967
- Ensemble OOF RMSE: IC50=347.2684, CC50=504.3614, SI=789.2981

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| catboost | 546.975967 | 547.787494 | 57.7 |
