# ChemAI experiment summary

- CV: group, n_splits=5, seed=42
- Preset: full
- Target transform: raw
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: extra_trees
- SI postprocess: direct model prediction
- Exact-match blend alpha: 0.0; matched test rows: 0
- Refit full for test prediction: False
- Ensemble OOF score: 520.691134
- Ensemble OOF RMSE: IC50=341.5972, CC50=455.9047, SI=764.5715

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| extra_trees | 520.691134 | 530.134763 | 8.1 |
