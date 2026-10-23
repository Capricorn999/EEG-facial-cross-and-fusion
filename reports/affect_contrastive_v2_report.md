# Corrected affect-aware EEG–facial contrastive evaluation (v2)

## Scope and correction

This campaign re-evaluated the original affect-aware contrastive idea without
changing the ASCERTAIN manifest, quality screen, subject-level LOSO protocol,
encoder widths (EEG 88→64→32; face 72→64→32), optimizer, or full-batch
training behavior.  Results are new and do not overwrite the original
campaign.

The original runner passed `valence_fixed` / `arousal_fixed` binary labels to
the pair sampler while asking for `abs(label_i-label_j) >= 3` negative pairs.
Such pairs are impossible for labels in {0,1}, so both cross-subject negatives
and M2 hard negatives were empty.  M1 and M2 therefore optimized identical
effective objectives.

The v2 runner separates roles:

- `*_fixed` remains binary classification supervision and the held-out metric
  label.
- Original integer `valence` / `arousal` ratings form pairs: cross-subject
  positives use distance 0; cross-subject negatives and same-subject hard
  negatives use distance ≥3.

`pair_sampler_unit_test.json` verifies every constraint on a synthetic example
and proves that both corrected repulsive losses activate (cross-negative loss
0.0482; hard-negative loss 0.0614).  Across the actual folds every sampled
pair passed its declared constraint.  The smallest training fold still had 963
eligible cross-subject positive and negative anchors; the minimum hard-negative
support was 937 for valence and 690 for arousal.  Thus M2 is genuinely distinct
from M1 in v2.

## Training and diagnostic checks

The primary comparison used seeds 3407, 2027, and 911, 47 eligible held-out
subjects, and models CE-only, M0 exact-pair InfoNCE, M1 corrected
affect-positive/cross-negative, and M2 with the additional hard-negative term.
Each fold fitted imputation, variance filtering, scaling, validation-subject
selection, threshold selection, and pair construction from training subjects
only.  The documented batch size of 128 was not implemented by v1; v2 retains
its actual full-batch procedure for comparability and records this explicitly.

The saved loss traces show active corrected terms.  Mean first-update gradient
norms (reported in `learning_diagnostics.csv`) were about 0.175 for
cross-subject attraction and 0.024–0.026 for cross-subject repulsion; M2's
hard-negative gradient norm was 0.025–0.027.  A smoke-fold gradient-direction
check found exact-pair versus cross-positive cosines of 0.12–0.23, rather than
systematic opposition.  Exact-pair versus repulsive terms ranged from about
-0.10 to 0.13.  This is mixed but not evidence that the exact-pair and
cross-subject positive objectives are decisively incompatible, so a
multi-positive replacement was not introduced.

## Classification diagnosis

Primary fixed-threshold results are averages over subject-folds and seeds;
single-class held-out subjects are explicitly retained but excluded from the
usual two-class UAR calculation (3 valence and 12 arousal subject-folds per
seed).

| Task | Model | Mean fold UAR | Accuracy | F1 | AUC |
|---|---|---:|---:|---:|---:|
| Valence | CE | 0.526 | 0.552 | 0.231 | 0.571 |
| Valence | M0 | 0.506 | 0.563 | 0.108 | 0.525 |
| Valence | M1-corrected | 0.501 | 0.549 | 0.085 | 0.522 |
| Valence | M2-corrected | 0.506 | 0.542 | 0.111 | 0.536 |
| Arousal | CE | 0.504 | 0.655 | 0.774 | 0.478 |
| Arousal | M0 | 0.500 | 0.650 | 0.771 | 0.487 |
| Arousal | M1-corrected | 0.500 | 0.648 | 0.771 | 0.512 |
| Arousal | M2-corrected | 0.502 | 0.654 | 0.774 | 0.498 |

The earlier UAR=0.500 pattern is explained by majority-class behavior rather
than missing metrics: mean positive prediction fractions were 0.99 for arousal
across neural models and 0.10–0.21 for valence.  Per-fold probabilities,
confusion matrices, recalls, class counts, selected validation-only thresholds,
and convergence traces are saved in `fold_metrics.csv`, `oof_predictions.csv`,
and `learning_diagnostics.csv`.

As an exploratory diagnosis, class-weighted CE-only (same folds; weights
derived only from each training fold) reduced the collapse but did not solve
recognition: valence mean fold UAR was 0.540 and arousal 0.504.  It is not a
replacement primary comparison.

The v1 early-fusion logistic baseline remained higher numerically (valence
0.522; arousal 0.530).  Against it, seed-3407 M1 had valence ΔUAR=-0.043
(95% subject-bootstrap CI -0.085 to -0.002), but the Holm-adjusted paired
Wilcoxon p=0.148; all other neural-versus-early-fusion comparisons were also
not significant after correction.  Corrected M1/M2 versus M0 comparisons were
not significant after Holm correction for any task/seed.

## Completely unseen-subject geometry

To avoid mixing representations from different LOSO encoders, geometry used
six multi-subject held-out groups.  Each group was encoded by a model trained
without all evaluated subjects.  Categories use original ratings: A exact
EEG–face trial; B same subject/different affect; C different subject/same
affect; D different subject/different affect.  Subject-clustered bootstrap CIs
were computed from per-subject similarities, with same-clip and different-clip
slices saved separately.

Across all seeds, tasks, and M0/M1/M2 models, every all-clip C−D interval
included zero.  Examples: valence M1 C−D was -0.0044 [-0.0211, 0.0117] (seed
3407), -0.0024 [-0.0167, 0.0107] (2027), and -0.0026 [-0.0213, 0.0255] (911).
For arousal M1 it was 0.0112 [-0.0057, 0.0293], -0.0009 [-0.0377, 0.0302], and
0.0065 [-0.0112, 0.0261].  C−B was generally negative, so affect similarity did
not overcome same-subject similarity.  Full category means/medians and clip
strata are in `geometry/unseen_subject_geometry_summary.csv`.

## Subject information

Frozen probes were fitted on representations of training subjects, distinct
from the unseen-subject emotion test.  Across held-out geometry fits, subject
top-1 decoding remained substantial: 0.341/0.332/0.341 for arousal M0/M1/M2
and 0.349/0.333/0.341 for valence M0/M1/M2 (chance is roughly 1/47).  M1 lowers
identity decodability slightly but this does not establish affect
disentanglement, particularly because unseen-subject affect geometry and LOSO
recognition did not improve.

## Decision and limitations

**Outcome D — stop this contrastive-objective branch.**  The sampler bug was
real and is repaired; negative and hard-negative terms now have eligible pairs,
nonzero losses, and nonzero gradients.  Yet corrected M1/M2 do not provide
reproducible LOSO recognition improvement over M0 or early fusion, nor the
primary unseen-subject C>D geometry result.  These results do not prove
equivalence; their CIs limit what this experiment can support.

The immediate limitation appears to be weak subject-independent emotion signal
in these static extracted features combined with severe fold-level class
imbalance and heterogeneous self-reported ratings, not an untested hard-pair
implementation.  A next effort should prioritize label/response
personalization, reliability-aware labels, richer temporal features, or a
dataset with stronger cross-subject affect support—not GRL, larger networks,
or Transformers on this branch.

## Reproduction

Run from the repository root after placing ASCERTAIN feature files at
`../data/ASCERTAIN_processed/`:

```bash
python src/evaluation/run_affect_contrastive_v2.py \
  --data-dir ../data/ASCERTAIN_processed \
  --manifest data/manifests/ascertain_trial_manifest.csv \
  --out results/affect_contrastive_v2 --seeds 3407,2027,911
python src/evaluation/analyze_affect_contrastive_v2.py \
  --data-dir ../data/ASCERTAIN_processed \
  --manifest data/manifests/ascertain_trial_manifest.csv \
  --out results/affect_contrastive_v2/geometry --groups 6 --seeds 3407,2027,911
python src/evaluation/summarize_affect_contrastive_v2.py --root .
```
