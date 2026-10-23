# Subject-dependent affect semantics

## Protocol

Paired-valid ASCERTAIN trials were analyzed with fold-local preprocessing. Population models used other subjects only. Few-shot P1/P2/P3/P4 use K target-subject support ratings and are evaluated on the same remaining target trials; they are not zero-shot results. P3 is a strongly regularized residual ridge head on a training-only PCA projection; P4 shrinks P3 toward P1.

## Phase 1: rating variation

| Task | Clip fixed-effect R² | Subject fixed-effect R² | Combined fixed-effect R² |
|---|---:|---:|---:|
| valence | 0.605 | 0.084 | 0.671 |
| arousal | 0.156 | 0.135 | 0.294 |

These are descriptive fixed-effect decompositions, not identifiable subject×clip interaction estimates. They show stable subject mean/rating-scale differences after accounting only descriptively for clip effects, but cannot establish why subjects differ.

## Phase 2/4: zero-shot controls

| Task | Modality | Population feature MAE | Clip-mean MAE | Global-mean MAE |
|---|---|---:|---:|---:|
| valence | eeg | 1.572 | 0.936 | 1.515 |
| valence | face | 1.581 | 0.936 | 1.515 |
| valence | fusion | 1.632 | 0.936 | 1.515 |
| arousal | eeg | 1.091 | 0.988 | 1.039 |
| arousal | face | 1.151 | 0.988 | 1.039 |
| arousal | fusion | 1.177 | 0.988 | 1.039 |

## Phase 3: matched fusion adaptation

| Task | K | P0 MAE | P1 bias MAE | P2 scale MAE | P3 mapping MAE | P4 shrunk MAE |
|---|---:|---:|---:|---:|---:|---:|
| valence | 0 | 1.632 | nan | nan | nan | nan |
| valence | 1 | 1.645 | 2.220 | 2.220 | 2.220 | 2.220 |
| valence | 2 | 1.639 | 1.973 | 6.000 | 1.958 | 1.940 |
| valence | 4 | 1.627 | 1.832 | 2.069 | 1.868 | 1.803 |
| valence | 8 | 1.578 | 1.739 | 1.697 | 1.778 | 1.720 |
| arousal | 0 | 1.177 | nan | nan | nan | nan |
| arousal | 1 | 1.188 | 1.591 | 1.591 | 1.591 | 1.591 |
| arousal | 2 | 1.208 | 1.366 | 8.985 | 1.432 | 1.388 |
| arousal | 4 | 1.197 | 1.247 | 1.378 | 1.290 | 1.253 |
| arousal | 8 | 1.127 | 1.115 | 1.089 | 1.173 | 1.115 |

## Identity diagnostic

| Modality | Subjects | Top-1 ID accuracy |
|---|---:|---:|
| eeg | 47 | 0.228 |
| face | 47 | 0.683 |
| fusion | 47 | 0.664 |

## Conclusion

Clip effects dominate zero-shot prediction: clip means outperform every EEG, face, and fusion population feature model in this protocol. Although subject fixed effects are descriptively nonzero and identity is decodable (especially from facial features), the regularized P3/P4 mapping heads do not consistently outperform P1/P2 or even P0 on independent target-subject queries. Thus H1 is supported only as stable rating variation; H2-H4 are not supported by these static features and matched few-shot tests. Phase 5 is not justified. These findings do not establish that affect is inherently subject-dependent or that identity is nuisance; they show that this dataset/protocol supplies no reliable incremental predictive evidence for a complex subject-specific feature mapping beyond stimulus controls.
