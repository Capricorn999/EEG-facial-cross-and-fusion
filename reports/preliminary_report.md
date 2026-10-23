# ASCERTAIN preliminary experiment report

## Scope

This is a deliberately small, falsifiable pilot: quality-screened extracted EEG (88-D) and facial-motion (72-D) features, fixed rating-scale thresholds, fold-fitted preprocessing, LOSO logistic/linear-SVM baselines, and a 32-D ridge-CCA alignment comparator. No raw signals, personality inputs, Transformers, or graph models were used.

## Dataset and protocol

- 2,088 possible trials; 1,765 quality-screened EEG trials; 1,463 face trials; 1,177 paired trials with both primary labels. See `dataset_audit.md`.
- Primary labels: valence > 0 on the documented −3…3 scale; arousal > 3 on the 0…6 scale.
- Every emotion score is an LOSO aggregate. Imputation, zero-variance filtering and scaling were fit on the other subjects only.
- Fair fusion/alignment comparisons use the same 47 held-out subjects having paired test trials.

## Main results (mean LOSO UAR)

### Valence

| Representation / model | UAR | ROC-AUC |
|---|---:|---:|
| EEG (paired support) | 0.535 | 0.579 |
| Face (paired support) | 0.514 | 0.560 |
| Early fusion | 0.522 | 0.559 |
| CCA shared | 0.536 | 0.573 |
| CCA concat | 0.520 | 0.560 |

Paired Wilcoxon, CCA shared vs. early fusion: n=47, p=0.722.
Paired Wilcoxon, CCA concat vs. early fusion: n=47, p=0.735.

### Arousal

| Representation / model | UAR | ROC-AUC |
|---|---:|---:|
| EEG (paired support) | 0.517 | 0.504 |
| Face (paired support) | 0.522 | 0.453 |
| Early fusion | 0.530 | 0.479 |
| CCA shared | 0.501 | 0.495 |
| CCA concat | 0.547 | 0.540 |

Paired Wilcoxon, CCA shared vs. early fusion: n=47, p=0.252.
Paired Wilcoxon, CCA concat vs. early fusion: n=47, p=0.245.

## Subject-information diagnostic

| Representation | Top-1 subject ID | Top-5 | Caveat |
|---|---:|---:|---|
| raw_eeg | 0.224 | 0.558 | 57 analyzable subjects |
| raw_face | 0.662 | 0.884 | 47 analyzable subjects |
| early_fusion | 0.619 | 0.886 | 46 analyzable subjects |
| CCA shared latent | 0.411 | 0.754 | exploratory global unsupervised transform, 46 subjects |
| CCA concat latent | 0.445 | 0.771 | exploratory global unsupervised transform, 46 subjects |

Raw face features are highly subject-identifying (0.662 top-1, versus 0.021 chance for its 47 active subjects). Both aligned variants remain highly subject-identifying (shared 0.411; concat 0.445; both versus 0.022 chance for 46 subjects). These are lower than raw face but the probes are exploratory, not formal adversarial-disentanglement results.

## Cross-modal pairing diagnostic

See `pair_similarity_analysis.csv`. Same-subject/same-clip similarity is 0.318, versus 0.093 for same-subject/different-clip and approximately zero for different-subject pairs. In this linear alignment, synchronized pairing dominates similarity; different-subject, similar-affect pairs do not yet form a distinct positive-similarity category.

## Decision

**Stop the pilot before adding adversarial or raw-temporal complexity.** The strong-evidence criterion is not met: although concatenated CCA features numerically improve arousal UAR, the paired test is not significant and both aligned representations retain substantial subject-ID predictability. The moderate-evidence criterion is also not met because identity remains far above chance.

A follow-up is justified only after diagnosing the data-quality/support imbalance and testing a controlled contrastive model with affect-matched cross-subject positives against this locked LOSO protocol. Do not claim that cross-modal agreement currently isolates affect from subject nuisance.

## Deliberate exclusions

The requested small neural MLP and adversarial ablation are not reported as completed results. This is intentional under the supplied stopping rule: the linear aligned comparator did not provide adequate evidence, so adding higher-capacity/disentanglement machinery would confound diagnosis with architecture search. Their absence is not evidence of failure or success.
