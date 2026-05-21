# ChemAI experiment summary

- CV: group, n_splits=5, seed=777
- Preset: full
- Target transform: raw
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: extra_trees
- SI postprocess: direct model prediction
- Exact-match blend alpha: 0.0; matched test rows: 0
- Refit full for test prediction: False
- Ensemble OOF score: 522.968404
- Ensemble OOF RMSE: IC50=345.2051, CC50=458.7469, SI=764.9532

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| extra_trees | 522.968404 | 532.283264 | 16.2 |
