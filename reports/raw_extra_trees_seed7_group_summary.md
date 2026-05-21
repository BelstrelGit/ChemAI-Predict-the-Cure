# ChemAI experiment summary

- CV: group, n_splits=5, seed=7
- Preset: full
- Target transform: raw
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: extra_trees
- SI postprocess: direct model prediction
- Exact-match blend alpha: 0.0; matched test rows: 0
- Refit full for test prediction: False
- Ensemble OOF score: 521.777933
- Ensemble OOF RMSE: IC50=342.9369, CC50=456.9198, SI=765.4771

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| extra_trees | 521.777933 | 530.921657 | 16.0 |
