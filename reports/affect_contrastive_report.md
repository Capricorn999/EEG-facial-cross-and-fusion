# Affect-aware cross-subject contrastive report

## Locked scope

This follow-up preserves the original manifest, quality filter, fixed labels and LOSO evaluation. It evaluates lightweight 88→64→32 EEG and 72→64→32 face encoders: M0 exact-pair neural contrastive control, M1 with cross-subject same-rating affect positives, and M2 with additional same-subject different-affect hard negatives.

## Diagnostic findings

- Paired support is substantially uneven: min 0, Q1 5, median 23, mean 20.3, Q3 36, max 36 trials per subject. 38, 32, and 26 subjects meet ≥15, ≥20, and ≥25 support respectively.
- Clip-consensus subsets were set by the lowest fold-local training-subject rating SD; they are sensitivity analyses, not performance-selected subsets.

## Main all-eligible LOSO results

| Task | M0 UAR | M1 UAR | M2 UAR | M0 ID top-1 | M1 ID top-1 | M2 ID top-1 |
|---|---:|---:|---:|---:|---:|---:|
| valence | 0.540 | 0.517 | 0.517 | 0.271 | 0.246 | 0.258 |
| arousal | 0.544 | 0.544 | 0.544 | 0.301 | 0.237 | 0.258 |

M1/M2 reduce frozen-probe identity relative to M0, but none approaches chance (~2.17% for 46 probe subjects).

## Geometry result

| Task / model | A exact | B same-subject different-affect | C different-subject same-affect | D different-subject different-affect |
|---|---:|---:|---:|---:|
| arousal M0 | 0.354 | 0.341 | 0.256 | 0.252 |
| arousal M1 | 0.264 | 0.259 | 0.208 | 0.210 |
| arousal M2 | 0.744 | 0.735 | 0.659 | 0.661 |
| valence M0 | 0.124 | 0.101 | 0.024 | 0.024 |
| valence M1 | 0.310 | 0.300 | 0.222 | 0.230 |
| valence M2 | 0.088 | 0.087 | 0.081 | 0.075 |

Across all six trained task/model combinations, C was not significantly greater than D; C also remained below B except for the weak/non-significant valence M2 contrast. Thus the requested affect-over-subject geometry was not achieved.

## Paired statistical tests

| Task | Comparison | n | ΔUAR | Wilcoxon p | rank-biserial |
|---|---:|---:|---:|---:|---:|
| valence | M1 vs M0 | 47 | -0.023 | 0.0346 | -0.553 |
| valence | M2 vs M0 | 47 | -0.023 | 0.0346 | -0.553 |
| valence | M2 vs M1 | 47 | 0.000 | 1 | 0.000 |
| valence | M1 vs EarlyFusion | 47 | -0.005 | 0.164 | -0.233 |
| valence | M2 vs EarlyFusion | 47 | -0.005 | 0.164 | -0.233 |
| arousal | M1 vs M0 | 47 | 0.000 | 1 | 0.000 |
| arousal | M2 vs M0 | 47 | 0.000 | 1 | 0.000 |
| arousal | M2 vs M1 | 47 | 0.000 | 1 | 0.000 |
| arousal | M1 vs EarlyFusion | 47 | 0.013 | 0.755 | 0.055 |
| arousal | M2 vs EarlyFusion | 47 | 0.013 | 0.755 | 0.055 |

## Decision

**Not supported; stop before adversarial or more complex modeling.** Cross-subject affective positives reduced subject-probe accuracy modestly but did not improve cross-subject affect geometry (C>D) or yield a statistically supported LOSO benefit. M2 did not improve M1, consistent with an ineffective hard-negative term at the locked margin.

A future route should be a targeted loss-design diagnosis—not GRL—such as stronger/multiple hard negatives or continuous affect-distance weights, first validated by an in-training pair-loss audit.
