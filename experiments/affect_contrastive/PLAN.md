# Affect-aware cross-subject contrastive campaign

## Parent result and question

The original LOSO pilot found that generic CCA alignment did not establish the
desired performance/invariance trade-off. This campaign tests whether
cross-subject pairs with similar affect, together with same-subject
different-affect hard negatives, change that geometry.

## Locked comparability contract

- Dataset, quality screen, fixed labels, manifest, and LOSO folds are reused
  unchanged from the original pilot.
- Main comparison uses all quality-screened paired trials and exactly the
  previously eligible paired test subjects.
- M0 is the neural generic-pair control; M1 adds cross-subject similar-affect
  positives; M2 additionally adds same-subject different-affect hard negatives.
- Each task (valence / arousal) is trained independently.
- Fold preprocessing and pair construction use training subjects only.

## Fixed pre-performance choices

- 32-D encoders: input -> 64 -> 32, ReLU, dropout 0.20, LayerNorm.
- AdamW (1e-3, 1e-4 weight decay), 40 epochs, batch size 128, seed 3407.
- Binary label distance thresholds: delta_pos=0 (same discrete rating) and
  delta_neg=3 for valence; delta_pos=0 and delta_neg=3 for arousal.
- M1/M2 pair loss is a margin ranking loss: positives attraction, negatives
  repulsion at cosine margin 0.20. Exact pairs get weight 1.0; affective
  cross-subject positives get 0.5; hard negatives get 1.0.

## Slices

1. Data diagnostics: paired support and clip rating consensus.
2. Claim-carrying all-eligible LOSO M0/M1/M2, valence and arousal.
3. Sensitivity slices: >=15 paired support, high-consensus half, and their
   intersection. These are diagnostic only, not a replacement main result.

## Stop condition

Stop if M1/M2 fail to improve cross-subject affect geometry or if subject
leakage stays high without meaningful LOSO benefit. Do not add GRL or more
complex architectures in this campaign.

## Execution envelope

RTX 4090 available (about 7 GB occupied at campaign start); the small models
and 58 LOSO folds fit comfortably. No external data or dependency is needed.
