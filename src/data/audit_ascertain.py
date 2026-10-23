#!/usr/bin/env python3
"""Create an auditable ASCERTAIN trial manifest and descriptive report.

No trial is silently removed: the manifest retains all 58 x 36 trials and
separates quality/failure flags from the stricter analytic-validity flags.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def _col_number(ref: str) -> int:
    col = re.match(r"([A-Z]+)", ref).group(1)
    n = 0
    for char in col:
        n = n * 26 + ord(char) - 64
    return n


def _xlsx_sheets(path: Path) -> dict[str, dict[tuple[int, int], float | str]]:
    """Minimal XLSX reader sufficient for the supplied quality workbook."""
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in elem.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")) for elem in root]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
        out = {}
        for sheet in workbook.find("m:sheets", NS):
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = targets[rid].lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            root = ET.fromstring(archive.read(target))
            cells: dict[tuple[int, int], float | str] = {}
            for cell in root.findall(".//m:sheetData/m:row/m:c", NS):
                value = cell.find("m:v", NS)
                if value is None:
                    continue
                raw = value.text or ""
                if cell.attrib.get("t") == "s":
                    val: float | str = shared[int(raw)]
                else:
                    try:
                        val = float(raw)
                    except ValueError:
                        val = raw
                row = int(re.search(r"(\d+)$", cell.attrib["r"]).group(1))
                cells[(row, _col_number(cell.attrib["r"]))] = val
            out[sheet.attrib["name"]] = cells
    return out


def _quality_matrix(cells: dict[tuple[int, int], float | str], data_start_row: int) -> np.ndarray:
    """Read supplied matrix sheets: rows=video ids, columns=subject ids."""
    result = np.full((58, 36), np.nan)
    # Quality sheets use the first 36 data rows and 58 data columns.
    for video in range(1, 37):
        for subject in range(1, 59):
            value = cells.get((data_start_row + video - 1, 1 + subject))
            try:
                result[subject - 1, video - 1] = float(value)
            except (TypeError, ValueError):
                pass
    return result


def _stack_cells(mat: dict, variable: str) -> np.ndarray:
    raw = mat[variable]
    values = [np.asarray(raw.flat[i], dtype=float) for i in range(raw.size)]
    return np.stack(values, axis=0)


def _failure_sets(mat: dict, variable: str) -> list[set[int]]:
    out = []
    for obj in mat[variable].flat:
        ids = np.asarray(obj).astype(int).ravel().tolist()
        out.append({v for v in ids if 1 <= v <= 36})
    return out


def _metric_lines(series: pd.Series) -> str:
    return f"mean={series.mean():.3f}, std={series.std(ddof=1):.3f}, min={series.min():.3f}, max={series.max():.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data, out = args.data_dir, args.output_dir
    manifests, plots = out / "data/manifests", out / "results/plots"
    manifests.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    required = ["Dt_EEGFeatures.mat", "Dt_EMOFeatures.mat", "Dt_SelfReports.mat", "Dt_Personality.mat", "Dt_Order_Movie.mat"]
    missing = [name for name in required if not (data / name).exists()]
    if missing:
        raise FileNotFoundError(f"Required files missing: {missing}")
    eeg_mat, emo_mat = loadmat(data / "Dt_EEGFeatures.mat"), loadmat(data / "Dt_EMOFeatures.mat")
    self_mat, personality_mat, order_mat = (loadmat(data / name) for name in required[2:])
    eeg, emo = _stack_cells(eeg_mat, "EEGFeatures_58"), _stack_cells(emo_mat, "EMOFeatures_58")
    ratings = np.asarray(self_mat["Ratings"], dtype=float)
    clips = np.asarray(self_mat["ClipNumbers"], dtype=int)
    personality = np.asarray(personality_mat["Personality"], dtype=float)
    order = np.asarray(order_mat["PermutationList"], dtype=int)
    assert eeg.shape == (58, 36, 88), eeg.shape
    assert emo.shape == (58, 36, 72), emo.shape
    assert ratings.shape == (5, 58, 36), ratings.shape
    assert clips.shape == order.shape == (58, 36)
    eeg_failures = _failure_sets(eeg_mat, "EEGFailures_58")
    emo_failures = _failure_sets(emo_mat, "EMOFailures_58")

    quality_file = data / "Data_Quality_Evaluation.xlsx"
    eeg_quality = emo_quality = np.full((58, 36), np.nan)
    quality_note = "Quality workbook unavailable."
    if quality_file.exists():
        sheets = _xlsx_sheets(quality_file)
        eeg_quality = _quality_matrix(sheets["raw_EEG"], 4)
        emo_quality = _quality_matrix(sheets["EMO"], 3)
        quality_note = "Quality score 1=perfect through 6=missing; analytic candidate requires score <=4."

    rows = []
    names = ["arousal", "valence", "engagement", "liking", "familiarity"]
    for s in range(58):
        for c in range(36):
            clip = int(clips[s, c])
            e_quality, m_quality = eeg_quality[s, c], emo_quality[s, c]
            e_failure, m_failure = clip in eeg_failures[s], clip in emo_failures[s]
            # Individual non-finite feature values are repaired within each training fold,
            # but an entirely non-finite trial has no usable modality information.
            eeg_has_signal = np.isfinite(eeg[s, c]).any()
            emo_has_signal = np.isfinite(emo[s, c]).any()
            e_valid = bool(not e_failure and eeg_has_signal and (np.isnan(e_quality) or e_quality <= 4))
            m_valid = bool(not m_failure and emo_has_signal and (np.isnan(m_quality) or m_quality <= 4))
            record = {"subject_id": s + 1, "clip_id": clip, "presentation_position": int(order[s, c])}
            record.update({name: float(ratings[i, s, c]) for i, name in enumerate(names)})
            record.update({"eeg_valid": e_valid, "emo_valid": m_valid, "eeg_failure": e_failure, "emo_failure": m_failure,
                           "eeg_quality": e_quality, "emo_quality": m_quality,
                           "eeg_nonfinite_features": int((~np.isfinite(eeg[s, c])).sum()),
                           "emo_nonfinite_features": int((~np.isfinite(emo[s, c])).sum())})
            rows.append(record)
    manifest = pd.DataFrame(rows).sort_values(["subject_id", "clip_id"])
    manifest.to_csv(manifests / "ascertain_trial_manifest.csv", index=False)

    valid_labels = manifest[["valence", "arousal"]].notna().all(axis=1)
    paired = manifest.eeg_valid & manifest.emo_valid & valid_labels
    manifest["valence_fixed"] = (manifest.valence > 0).astype("Int64")
    manifest["arousal_fixed"] = (manifest.arousal > 3).astype("Int64")
    for target in ["valence", "arousal"]:
        median = manifest.loc[paired, target].median()
        manifest[f"{target}_paired_median"] = (manifest[target] > median).astype("Int64")
    manifest.to_csv(manifests / "ascertain_trial_manifest.csv", index=False)

    # Descriptive plots retain all labelled trials, irrespective of modality quality.
    for target, midpoint in [("valence", 0), ("arousal", 3)]:
        fig, ax = plt.subplots(figsize=(6, 4))
        values = manifest[target].dropna()
        bins = np.arange(values.min() - .5, values.max() + 1, 1)
        ax.hist(values, bins=bins, color="#3969ac", edgecolor="white")
        ax.axvline(midpoint, color="#e68310", lw=2, label=f"fixed midpoint ({midpoint})")
        ax.set(xlabel=target.capitalize(), ylabel="Trials", title=f"ASCERTAIN {target} ratings")
        ax.legend()
        fig.tight_layout(); fig.savefig(plots / f"{target}_histogram.png", dpi=180); plt.close(fig)
    per_subject = manifest.groupby("subject_id")[["valence", "arousal"]].agg(["mean", "std"])
    per_clip = manifest.groupby("clip_id")[["valence", "arousal"]].agg(["mean", "std"])
    per_subject.to_csv(out / "results/subject_rating_summary.csv")
    per_clip.to_csv(out / "results/clip_rating_summary.csv")

    shape_rows = {"EEGFeatures_58": list(eeg.shape), "EMOFeatures_58": list(emo.shape), "Ratings": list(ratings.shape),
                  "Personality": list(personality.shape), "PermutationList": list(order.shape)}
    audit = ["# ASCERTAIN dataset audit", "", "## Sources and tensor contract", "", "```json", json.dumps(shape_rows, indent=2), "```", "",
             f"Total possible trials: **{len(manifest)}** (58 subjects × 36 clips).",
             f"Valid EEG trials (quality <=4, no explicit failure, some finite values): **{int(manifest.eeg_valid.sum())}**.",
             f"Valid face trials (same criterion): **{int(manifest.emo_valid.sum())}**.",
             f"Valid paired EEG–face trials with valence/arousal present: **{int(paired.sum())}**.", "",
             "## Exclusion and missing-data policy", "", f"- {quality_note}",
             "- Explicit modality failure IDs exclude that modality for that trial.",
             "- Entirely non-finite modality vectors exclude that modality. Partially non-finite vectors remain in the manifest and must be imputed within each LOSO training fold only.",
             "- Main paired modeling uses `eeg_valid & emo_valid` and non-missing task label; no trial is deleted from the manifest.", "",
             "## Missingness", "", f"- EEG explicit failures: **{sum(map(len, eeg_failures))}**; face explicit failures: **{sum(map(len, emo_failures))}**.",
             f"- EEG non-finite feature entries: **{int((~np.isfinite(eeg)).sum())}** / {eeg.size}; face: **{int((~np.isfinite(emo)).sum())}** / {emo.size}.",
             f"- Valid EEG trials per subject: {_metric_lines(manifest.groupby('subject_id').eeg_valid.sum())}.",
             f"- Valid face trials per subject: {_metric_lines(manifest.groupby('subject_id').emo_valid.sum())}.", "",
             "## Labels", "", "- Valence ranges from −3 to 3; primary fixed threshold is `valence > 0` (neutral is non-positive).",
             "- Arousal ranges from 0 to 6; primary fixed threshold is `arousal > 3`.",
             "- Paired-data medians are also written to the manifest for sensitivity analysis, but are not the primary labels.",
             f"- Pearson correlation (all complete ratings): **{manifest[['valence', 'arousal']].corr().iloc[0,1]:.3f}**.", "",
             "## Class balance on valid paired trials", ""]
    for column in ["valence_fixed", "arousal_fixed", "valence_paired_median", "arousal_paired_median"]:
        counts = manifest.loc[paired, column].value_counts().sort_index().to_dict()
        audit.append(f"- `{column}`: {counts}")
    audit += ["", "## Reproducibility", "", f"Python: {sys.version.split()[0]}; NumPy: {np.__version__}; SciPy: {scipy.__version__}; pandas: {pd.__version__}; platform: {platform.platform()}."]
    (out / "reports").mkdir(exist_ok=True)
    (out / "reports/dataset_audit.md").write_text("\n".join(audit) + "\n")
    print(json.dumps({"manifest": str(manifests / "ascertain_trial_manifest.csv"), "paired_trials": int(paired.sum()), "eeg_valid": int(manifest.eeg_valid.sum()), "emo_valid": int(manifest.emo_valid.sum())}, indent=2))


if __name__ == "__main__":
    main()
