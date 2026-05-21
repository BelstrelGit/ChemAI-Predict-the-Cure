# ChemAI experiment summary

- CV: group, n_splits=5, seed=42
- Preset: full
- Target transform: raw
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: catboost
- SI postprocess: direct model prediction
- Exact-match blend alpha: 0.0; matched test rows: 0
- Refit full for test prediction: False
- Ensemble OOF score: 523.449410
- Ensemble OOF RMSE: IC50=354.9550, CC50=460.4258, SI=754.9674

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| catboost | 523.449410 | 1071.556241 | 43.7 |
