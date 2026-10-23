#!/usr/bin/env python3
"""Leakage-safe LOSO baselines for ASCERTAIN extracted features.

All transformations (median imputation, zero-variance removal and scaling) are
fit only on training subjects inside each fold.  The optional MLP has a
subject-held-out validation set, never a random trial validation split.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)


def stack_cells(path: Path, key: str) -> np.ndarray:
    raw = loadmat(path)[key]
    return np.stack([np.asarray(x, dtype=np.float64) for x in raw.flat])


def score(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None) -> dict[str, float]:
    result = {"accuracy": accuracy_score(y, pred), "uar": balanced_accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0)}
    result["auc"] = roc_auc_score(y, prob) if prob is not None and np.unique(y).size == 2 else np.nan
    return result


def validation_subjects(train_subjects: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_subjects)
    n_val = max(1, round(len(shuffled) * .1))
    return shuffled[n_val:], shuffled[:n_val]


def make_linear(kind: str, seed: int) -> Pipeline:
    # With <160 features and ~2k trials, the primal linear SVM and liblinear
    # logistic solver are deterministic, fast LOSO comparators.
    model = (LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", C=1.0, random_state=seed)
             if kind == "logreg" else LinearSVC(class_weight="balanced", C=1.0, dual=False, max_iter=1000, random_state=seed))
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("variance", VarianceThreshold(0.0)),
                     ("scaler", StandardScaler()), ("model", model)])


def run_one(model_name: str, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, seed: int) -> dict[str, float]:
    pipe = make_linear(model_name, seed); pipe.fit(Xtr, ytr)
    if model_name == "logreg": prob = pipe.predict_proba(Xte)[:, 1]
    else:
        dec = pipe.decision_function(Xte); prob = 1 / (1 + np.exp(-np.clip(dec, -50, 50)))
    return score(yte, (prob >= .5).astype(int), prob)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--label-scheme", choices=["fixed", "median"], default="fixed")
    args = ap.parse_args(); set_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mf = pd.read_csv(args.manifest)
    eeg = stack_cells(args.data_dir / "Dt_EEGFeatures.mat", "EEGFeatures_58")
    emo = stack_cells(args.data_dir / "Dt_EMOFeatures.mat", "EMOFeatures_58")
    # Every manifest row maps to the original subject x clip tensors by ID.
    idx_s, idx_c = mf.subject_id.to_numpy() - 1, mf.clip_id.to_numpy() - 1
    # SimpleImputer handles NaN but not ±Inf.  Converting non-finite values to
    # NaN here is value-neutral; each fold's training-only imputer supplies the
    # replacement statistic.
    X_eeg = np.where(np.isfinite(eeg[idx_s, idx_c]), eeg[idx_s, idx_c], np.nan)
    X_face = np.where(np.isfinite(emo[idx_s, idx_c]), emo[idx_s, idx_c], np.nan)
    paired_valid = (mf.eeg_valid & mf.emo_valid).to_numpy(bool)
    modalities = {"eeg": (X_eeg, mf.eeg_valid.to_numpy(bool)), "face": (X_face, mf.emo_valid.to_numpy(bool)),
                  # These matched-support rows are the only fair direct
                  # comparator for early fusion and aligned-latent models.
                  "eeg_paired": (X_eeg, paired_valid), "face_paired": (X_face, paired_valid),
                  "early_fusion": (np.c_[X_eeg, X_face], paired_valid)}
    rows = []
    for task in ["valence", "arousal"]:
        label = f"{task}_{'fixed' if args.label_scheme == 'fixed' else 'paired_median'}"
        y_all = mf[label].to_numpy(int)
        for modality, (X, valid) in modalities.items():
            models = ["logreg"] + (["linearsvm"] if modality not in {"early_fusion"} else [])
            for held_out in range(1, 59):
                test = valid & (mf.subject_id.to_numpy() == held_out)
                train = valid & (mf.subject_id.to_numpy() != held_out)
                if test.sum() == 0 or np.unique(y_all[train]).size < 2:
                    continue
                # This deterministic subject-only validation partition is persisted
                # for later MLP / latent model tuning; linear defaults are untouched.
                train_subj, val_subj = validation_subjects(np.setdiff1d(np.arange(1, 59), held_out), args.seed + held_out)
                for model in models:
                    metrics = run_one(model, X[train], y_all[train], X[test], y_all[test], args.seed + held_out)
                    rows.append({"task": task, "label_scheme": args.label_scheme, "model": model, "modality": modality,
                                 "fold_subject": held_out, "n_train": int(train.sum()), "n_test": int(test.sum()),
                                 "validation_subjects": ";".join(map(str, val_subj)), **metrics})
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)
    summary = result.groupby(["task", "model", "modality"])[["accuracy", "uar", "f1", "auc"]].agg(["mean", "std", "count"])
    summary.to_csv(args.output.with_name(args.output.stem + "_summary.csv"))
    print(summary.to_string())
    print(json.dumps({"rows": len(result), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
