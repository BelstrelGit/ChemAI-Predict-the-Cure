# ChemAI experiment summary

- CV: group, n_splits=5, seed=42
- Preset: full
- Target transform: raw
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: kernel_ridge
- SI postprocess: direct model prediction
- Exact-match blend alpha: 0.0; matched test rows: 0
- Refit full for test prediction: False
- Ensemble OOF score: 514.462652
- Ensemble OOF RMSE: IC50=330.5480, CC50=471.2209, SI=741.6191

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| kernel_ridge | 514.462652 | 1544.375843 | 0.4 |
