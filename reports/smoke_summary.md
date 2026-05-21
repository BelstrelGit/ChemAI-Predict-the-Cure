# ChemAI experiment summary

- CV: group, n_splits=3, seed=42
- Preset: fast
- Feature columns: 192 used, 18 constants dropped
- Selected ensemble: ridge
- SI postprocess: ratio CC50/IC50
- Exact-match blend alpha: 0.0; matched test rows: 0
- Ensemble OOF score: 728.960260
- Ensemble OOF RMSE: IC50=385.0227, CC50=612.8447, SI=1189.0133

## Model CV

| model | direct score | ratio-SI score | seconds |
|---|---:|---:|---:|
| ridge | 776.654460 | 728.960260 | 0.1 |
