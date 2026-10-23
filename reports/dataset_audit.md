# ASCERTAIN dataset audit

## Sources and tensor contract

```json
{
  "EEGFeatures_58": [
    58,
    36,
    88
  ],
  "EMOFeatures_58": [
    58,
    36,
    72
  ],
  "Ratings": [
    5,
    58,
    36
  ],
  "Personality": [
    58,
    5
  ],
  "PermutationList": [
    58,
    36
  ]
}
```

Total possible trials: **2088** (58 subjects × 36 clips).
Valid EEG trials (quality <=4, no explicit failure, some finite values): **1765**.
Valid face trials (same criterion): **1463**.
Valid paired EEG–face trials with valence/arousal present: **1177**.

## Exclusion and missing-data policy

- Quality score 1=perfect through 6=missing; analytic candidate requires score <=4.
- Explicit modality failure IDs exclude that modality for that trial.
- Entirely non-finite modality vectors exclude that modality. Partially non-finite vectors remain in the manifest and must be imputed within each LOSO training fold only.
- Main paired modeling uses `eeg_valid & emo_valid` and non-missing task label; no trial is deleted from the manifest.

## Missingness

- EEG explicit failures: **31**; face explicit failures: **454**.
- EEG non-finite feature entries: **7288** / 183744; face: **44856** / 150336.
- Valid EEG trials per subject: mean=30.431, std=9.661, min=1.000, max=36.000.
- Valid face trials per subject: mean=25.224, std=14.975, min=0.000, max=36.000.

## Labels

- Valence ranges from −3 to 3; primary fixed threshold is `valence > 0` (neutral is non-positive).
- Arousal ranges from 0 to 6; primary fixed threshold is `arousal > 3`.
- Paired-data medians are also written to the manifest for sensitivity analysis, but are not the primary labels.
- Pearson correlation (all complete ratings): **0.028**.

## Class balance on valid paired trials

- `valence_fixed`: {0: 651, 1: 526}
- `arousal_fixed`: {0: 418, 1: 759}
- `valence_paired_median`: {0: 651, 1: 526}
- `arousal_paired_median`: {0: 816, 1: 361}

## Reproducibility

Python: 3.12.8; NumPy: 1.26.4; SciPy: 1.15.3; pandas: 2.3.0; platform: Linux-6.8.0-64-generic-x86_64-with-glibc2.35.
