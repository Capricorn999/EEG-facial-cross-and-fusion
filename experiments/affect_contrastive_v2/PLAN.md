# Corrected affect-aware contrastive campaign (v2)

## Question and locked protocol

This campaign corrects the original sampler defect without modifying the
manifest, quality screen, LOSO eligibility rule, encoder architecture, or
optimizer.  Fixed binary labels are used only for classification; original
ASCERTAIN ratings are used only to form affect-distance pairs.

The primary LOSO comparison is run with seeds `3407`, `2027`, and `911`:

| Model | CE | exact-pair InfoNCE | cross-subject same-rating attraction | cross-subject far-rating repulsion | same-subject far-rating repulsion |
|---|---|---|---|---|---|
| CE | yes | no | no | no | no |
| M0 | yes | yes | no | no | no |
| M1-corrected | yes | yes | yes | yes | no |
| M2-corrected | yes | yes | yes | yes | yes |

`same-rating` means absolute original-rating difference 0; `far-rating` means
difference at least 3.  All pair pools are constructed only from training
subjects.  The primary training mode remains full-batch, matching the original
implementation; the discrepancy with the legacy plan's documented batch size
of 128 is reported rather than silently changed.

## Gates

1. Unit tests establish that every pair has the declared subject/rating
   relationship and that far-pair losses activate.
2. A single-fold smoke run must emit nonzero eligible negative and hard-negative
   support and distinct M1/M2 loss components.
3. The main three-seed LOSO campaign saves predictions, embeddings, training
   diagnostics and fold metrics.
4. A held-out multi-subject protocol evaluates A/B/C/D geometry without mixing
   embeddings from separately trained LOSO encoders.  Subject-clustered
   bootstrap intervals are calculated from per-subject pair summaries.

## Decision rule

The branch can continue only if corrected affect-aware models show a
reproducible positive unseen-subject C-D geometry difference *and* improve
LOSO recognition relative to M0 and early fusion.  Otherwise the report states
the evidence boundary and stops before larger models or adversarial learning.
