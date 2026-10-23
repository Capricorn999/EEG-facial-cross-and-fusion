# ASCERTAIN EEG–facial cross-modal pilot

This repository contains reproducible code, quality-screened trial manifests,
metrics, figures, and reports for two leakage-safe preliminary experiments on
ASCERTAIN extracted EEG and facial-motion features.

The source ASCERTAIN feature files are intentionally **not** included. Obtain
them under `../data/ASCERTAIN_processed/` relative to this directory before
running the scripts. The included trial manifest is sufficient to inspect the
published analysis protocol and results.

## Findings

- Generic CCA alignment did not provide a statistically supported improvement
  over early fusion while retaining substantial subject information.
- Affect-aware cross-subject contrastive positives modestly reduced subject-ID
  probe accuracy, but did not achieve the required affect geometry or improve
  LOSO recognition. The project therefore stops before adversarial learning.

See `reports/preliminary_report.md` and
`reports/affect_contrastive_report.md` for complete evidence-bound conclusions.

## Reproduce

Install `requirements.txt`, make the ASCERTAIN feature directory available,
then run the dataset audit and the evaluation scripts in `src/evaluation/`.
The affect-aware follow-up plan, locked hyperparameters, and stop rule are in
`experiments/affect_contrastive/PLAN.md`.

All LOSO preprocessing, validation-subject selection, and pair construction
are fold-local. Original pilot outputs remain immutable; follow-up outputs live
in `results/affect_contrastive/`.
