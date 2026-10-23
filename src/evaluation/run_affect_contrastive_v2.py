#!/usr/bin/env python3
"""Corrected, leakage-safe ASCERTAIN affect-aware contrastive evaluation.

Classification labels and pair ratings are deliberately separate: binary fixed
labels drive CE/metrics, whereas original integer ratings define pair distance.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEEDS = (3407, 2027, 911)
MODELS = ("CE", "M0", "M1", "M2")


def set_seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)


def stack(path: Path, key: str) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float32) for v in loadmat(path)[key].flat])


def preprocessing() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("var", VarianceThreshold()), ("scale", StandardScaler())])


class Encoder(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, 64), torch.nn.LayerNorm(64),
                                       torch.nn.ReLU(), torch.nn.Dropout(.2), torch.nn.Linear(64, 32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.net(x), dim=1)


class Net(torch.nn.Module):
    def __init__(self, eeg_dim: int, face_dim: int):
        super().__init__(); self.eeg = Encoder(eeg_dim); self.face = Encoder(face_dim); self.head = torch.nn.Linear(32, 2)

    def forward(self, eeg: torch.Tensor, face: torch.Tensor):
        ze, zf = self.eeg(eeg), self.face(face)
        return ze, zf, self.head((ze + zf) / 2)


def pair_indices(subject: np.ndarray, rating: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one valid partner per anchor under the preregistered constraints."""
    n = len(subject); pos = np.full(n, -1, dtype=int); neg = pos.copy(); hard = pos.copy()
    for i in range(n):
        delta = np.abs(rating - rating[i])
        cross = subject != subject[i]
        same = ~cross
        pools = (np.flatnonzero(cross & (delta == 0)),
                 np.flatnonzero(cross & (delta >= 3)),
                 np.flatnonzero(same & (np.arange(n) != i) & (delta >= 3)))
        for target, pool in zip((pos, neg, hard), pools):
            if pool.size: target[i] = rng.choice(pool)
    return pos, neg, hard


def pair_audit(subject: np.ndarray, rating: np.ndarray, indices: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
    names = ("positive", "cross_negative", "hard_negative"); out = {}
    for name, ii in zip(names, indices):
        mask = ii >= 0; anchors = np.flatnonzero(mask)
        if anchors.size:
            delta = np.abs(rating[anchors] - rating[ii[anchors]])
            same_subject = subject[anchors] == subject[ii[anchors]]
            required_same = name == "hard_negative"
            valid = ((same_subject == required_same) & ((delta == 0) if name == "positive" else (delta >= 3))).all()
            out.update({f"{name}_anchors": int(anchors.size), f"{name}_valid": bool(valid),
                        f"{name}_distance_mean": float(delta.mean()), f"{name}_distance_min": int(delta.min()),
                        f"{name}_distance_max": int(delta.max())})
        else:
            out.update({f"{name}_anchors": 0, f"{name}_valid": True,
                        f"{name}_distance_mean": np.nan, f"{name}_distance_min": np.nan, f"{name}_distance_max": np.nan})
    out["anchors_without_positive"] = float(np.mean(indices[0] < 0))
    out["anchors_without_cross_negative"] = float(np.mean(indices[1] < 0))
    out["anchors_without_hard_negative"] = float(np.mean(indices[2] < 0))
    return out


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0


def loss_terms(ze, zf, logits, y, mode, indices, class_weight=None):
    ce = torch.nn.functional.cross_entropy(logits, y, weight=class_weight)
    exact = torch.nn.functional.cross_entropy(ze @ zf.T / .15, torch.arange(len(y), device=y.device)) if mode != "CE" else _zero(ze)
    pos, neg, hard = indices
    def attraction(ii):
        mask = ii >= 0
        return (1 - (ze[mask] * zf[ii[mask]]).sum(1)).mean() if mask.any() else _zero(ze)
    def repulsion(ii):
        mask = ii >= 0
        return torch.relu((ze[mask] * zf[ii[mask]]).sum(1) - .2).mean() if mask.any() else _zero(ze)
    cross_pos = attraction(pos) if mode in ("M1", "M2") else _zero(ze)
    cross_neg = repulsion(neg) if mode in ("M1", "M2") else _zero(ze)
    hard_neg = repulsion(hard) if mode == "M2" else _zero(ze)
    weighted = {"ce": ce, "exact": .25 * exact, "cross_pos": .15 * cross_pos,
                "cross_neg": .15 * cross_neg, "hard_neg": .15 * hard_neg}
    return weighted, sum(weighted.values())


def gradient_norm(term: torch.Tensor, parameters) -> float:
    gradients = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((g.detach() ** 2).sum() for g in gradients if g is not None)).cpu())


def gradient_cosine(left: torch.Tensor, right: torch.Tensor, parameters) -> float:
    a = torch.autograd.grad(left, parameters, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(right, parameters, retain_graph=True, allow_unused=True)
    dot = sum((x.detach() * y.detach()).sum() for x, y in zip(a, b) if x is not None and y is not None)
    na = torch.sqrt(sum((x.detach() ** 2).sum() for x in a if x is not None)); nb = torch.sqrt(sum((y.detach() ** 2).sum() for y in b if y is not None))
    return float((dot / (na * nb + 1e-12)).cpu())


def prediction_metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int); classes = np.unique(y)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    recalls = np.divide(np.diag(cm), cm.sum(1), out=np.full(2, np.nan), where=cm.sum(1) > 0)
    return {"accuracy": accuracy_score(y, pred), "uar": balanced_accuracy_score(y, pred) if len(classes) == 2 else np.nan,
            "f1": f1_score(y, pred, zero_division=0), "auc": roc_auc_score(y, prob) if len(classes) == 2 else np.nan,
            "recall_0": recalls[0], "recall_1": recalls[1], "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]), "predicted_positive_fraction": float(pred.mean()),
            "single_class_test": bool(len(classes) < 2)}


def choose_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y)) < 2: return .5
    candidates = np.unique(np.r_[0., 1., prob])
    scores = [balanced_accuracy_score(y, prob >= v) for v in candidates]
    return float(candidates[int(np.argmax(scores))])


def fit_fold(xe, xf, meta, task, mode, train, val, test, device, seed_value, weighted_ce=False, return_fitted=False):
    pe, pf = preprocessing(), preprocessing()
    et = torch.tensor(pe.fit_transform(xe[train]), dtype=torch.float32, device=device)
    ft = torch.tensor(pf.fit_transform(xf[train]), dtype=torch.float32, device=device)
    vt = (torch.tensor(pe.transform(xe[val]), dtype=torch.float32, device=device), torch.tensor(pf.transform(xf[val]), dtype=torch.float32, device=device))
    tt = (torch.tensor(pe.transform(xe[test]), dtype=torch.float32, device=device), torch.tensor(pf.transform(xf[test]), dtype=torch.float32, device=device))
    y = meta[f"{task}_fixed"].to_numpy(int); rating = meta[task].to_numpy(int)
    yt = torch.tensor(y[train], dtype=torch.long, device=device); yv = y[val]; rng = np.random.default_rng(seed_value)
    raw_indices = pair_indices(meta.subject_id.to_numpy()[train], rating[train], rng)
    pair_diag = pair_audit(meta.subject_id.to_numpy()[train], rating[train], raw_indices)
    indices = tuple(torch.tensor(x, dtype=torch.long, device=device) for x in raw_indices)
    counts = np.bincount(y[train], minlength=2)
    class_weight = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device) if weighted_ce else None
    net = Net(et.shape[1], ft.shape[1]).to(device); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    diagnostic = []; best = None; stale = 0
    for epoch in range(40):
        net.train(); ze, zf, logits = net(et, ft); terms, total = loss_terms(ze, zf, logits, yt, mode, indices, class_weight)
        if epoch == 0:
            params = tuple(net.parameters())
            grads = {f"grad_{k}": gradient_norm(v, params) for k, v in terms.items()}
            grads["gradcos_exact_cross_pos"] = gradient_cosine(terms["exact"], terms["cross_pos"], params) if mode in ("M1", "M2") else np.nan
            grads["gradcos_exact_cross_neg"] = gradient_cosine(terms["exact"], terms["cross_neg"], params) if mode in ("M1", "M2") else np.nan
            grads["gradcos_exact_hard_neg"] = gradient_cosine(terms["exact"], terms["hard_neg"], params) if mode == "M2" else np.nan
        opt.zero_grad(); total.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vz, vf, vl = net(*vt); val_ce = torch.nn.functional.cross_entropy(vl, torch.tensor(yv, dtype=torch.long, device=device)).item()
        diagnostic.append({"epoch": epoch + 1, **{f"loss_{k}": float(v.detach().cpu()) for k, v in terms.items()}, "loss_total": float(total.detach().cpu()), "val_ce": val_ce})
        if best is None or val_ce < best[0] - 1e-4:
            best = (val_ce, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, epoch + 1); stale = 0
        else: stale += 1
        if stale >= 8: break
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad():
        vprob = torch.softmax(net(*vt)[2], 1)[:, 1].cpu().numpy()
        ze, zf, logits = net(*tt); prob = torch.softmax(logits, 1)[:, 1].cpu().numpy()
    threshold = choose_threshold(yv, vprob)
    metrics = {f"fixed_{k}": v for k, v in prediction_metrics(y[test], prob, .5).items()}
    metrics.update({f"val_threshold_{k}": v for k, v in prediction_metrics(y[test], prob, threshold).items()})
    diagnostic[-1].update(grads); diagnostic[-1]["best_epoch"] = best[2]; diagnostic[-1]["selected_threshold"] = threshold
    reps = (ze + zf).cpu().numpy() / 2
    if len(reps) < 2:
        rank = np.nan
    else:
        covariance = np.cov(reps, rowvar=False); eigen = np.linalg.eigvalsh(covariance); eigen = np.maximum(eigen, 0)
        rank = float((eigen.sum() ** 2) / np.maximum((eigen ** 2).sum(), 1e-12))
    payload = (metrics, prob, ze.cpu().numpy(), zf.cpu().numpy(), reps, pair_diag, diagnostic, best[2], threshold, counts, rank)
    return payload + (net, pe, pf) if return_fitted else payload


def data_arrays(data_dir: Path, manifest: pd.DataFrame):
    eeg = stack(data_dir / "Dt_EEGFeatures.mat", "EEGFeatures_58"); face = stack(data_dir / "Dt_EMOFeatures.mat", "EMOFeatures_58")
    s, c = manifest.subject_id.to_numpy() - 1, manifest.clip_id.to_numpy() - 1
    return np.where(np.isfinite(eeg[s, c]), eeg[s, c], np.nan), np.where(np.isfinite(face[s, c]), face[s, c], np.nan)


def valid_mask(m): return (m.eeg_valid & m.emo_valid & m.valence.notna() & m.arousal.notna()).to_numpy(bool)


def fold_masks(m, held, seed_value):
    valid = valid_mask(m); subjects = np.unique(m.subject_id[valid & (m.subject_id != held)])
    rng = np.random.default_rng(seed_value + held); val_subjects = rng.choice(subjects, size=max(1, round(.1 * len(subjects))), replace=False)
    test = valid & (m.subject_id.to_numpy() == held); val = valid & m.subject_id.isin(val_subjects).to_numpy(); train = valid & ~test & ~val
    return train, val, test, ";".join(map(str, sorted(val_subjects)))


def run_loso(xe, xf, m, out, seed_value, models, weighted_ce=False, smoke_subject=None):
    rows = []; predictions = []; embeddings = []; pair_rows = []; learning = []
    held_subjects = [smoke_subject] if smoke_subject else sorted(m.subject_id.unique())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for task in ("valence", "arousal"):
        for held in held_subjects:
            train, val, test, val_subjects = fold_masks(m, held, seed_value)
            if train.sum() < 20 or val.sum() < 2 or test.sum() < 1 or np.unique(m.loc[train, f"{task}_fixed"]).size < 2: continue
            for mode in models:
                result = fit_fold(xe, xf, m, task, mode, train, val, test, device, seed_value + held, weighted_ce)
                metrics, prob, ze, zf, reps, pair_diag, trace, best_epoch, threshold, counts, rank = result
                base = {"seed": seed_value, "task": task, "model": mode, "fold_subject": int(held), "n_train": int(train.sum()), "n_val": int(val.sum()), "n_test": int(test.sum()), "validation_subjects": val_subjects, "train_class_0": int(counts[0]), "train_class_1": int(counts[1]), "test_class_0": int((m.loc[test, f'{task}_fixed'] == 0).sum()), "test_class_1": int((m.loc[test, f'{task}_fixed'] == 1).sum()), "best_epoch": best_epoch, "selected_threshold": threshold, "embedding_effective_rank": rank, **metrics}
                rows.append(base); pair_rows.append({**base, **pair_diag})
                for item in trace: learning.append({"seed": seed_value, "task": task, "model": mode, "fold_subject": int(held), **item})
                for local, index in enumerate(np.flatnonzero(test)):
                    predictions.append({"seed": seed_value, "task": task, "model": mode, "fold_subject": int(held), "subject_id": int(m.subject_id.iloc[index]), "clip_id": int(m.clip_id.iloc[index]), "rating": int(m[task].iloc[index]), "label": int(m[f'{task}_fixed'].iloc[index]), "probability": float(prob[local]), "threshold": threshold})
                    embeddings.append({"seed": seed_value, "task": task, "model": mode, "fold_subject": int(held), "subject_id": int(m.subject_id.iloc[index]), "clip_id": int(m.clip_id.iloc[index]), "rating": int(m[task].iloc[index]), "label": int(m[f'{task}_fixed'].iloc[index]), "ze": ze[local].tolist(), "zf": zf[local].tolist(), "z": reps[local].tolist()})
    return pd.DataFrame(rows), pd.DataFrame(predictions), embeddings, pd.DataFrame(pair_rows), pd.DataFrame(learning)


def test_pair_sampler():
    subject = np.array([1, 1, 1, 2, 2, 2, 3, 3]); rating = np.array([-3, 0, 3, -3, 0, 3, -3, 3])
    idx = pair_indices(subject, rating, np.random.default_rng(3)); audit = pair_audit(subject, rating, idx)
    assert audit["positive_valid"] and audit["cross_negative_valid"] and audit["hard_negative_valid"]
    assert audit["cross_negative_anchors"] > 0 and audit["hard_negative_anchors"] > 0
    # Construct aligned far pairs: their cosine exceeds the .2 repulsion margin,
    # so this also proves both corrected negative losses can activate.
    ze = torch.nn.functional.normalize(torch.tensor([[1., .2, .1], [.2, 1., .1], [.1, .2, 1.],
                                                      [1., .1, .2], [.2, 1., .1], [.1, 1., .2],
                                                      [.7, .4, .2], [.2, .7, .4]]), dim=1).requires_grad_()
    zf = torch.nn.functional.normalize(torch.tensor([[.9, .3, .2], [.3, .9, .2], [.2, .3, .9],
                                                      [.8, .2, .3], [.3, .8, .2], [.2, .8, .3],
                                                      [.6, .5, .3], [.3, .6, .5]]), dim=1).requires_grad_()
    logits = torch.zeros(8, 2, requires_grad=True); y = torch.tensor([0, 1] * 4)
    terms, total = loss_terms(ze, zf, logits, y, "M2", tuple(torch.tensor(x) for x in idx))
    assert terms["cross_neg"].item() > 0 and terms["hard_neg"].item() > 0
    total.backward()
    assert ze.grad is not None and float(ze.grad.abs().sum()) > 0
    audit["synthetic_cross_negative_loss"] = float(terms["cross_neg"].detach())
    audit["synthetic_hard_negative_loss"] = float(terms["hard_neg"].detach())
    return audit


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", type=Path, required=True); ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS))); ap.add_argument("--models", default=",".join(MODELS)); ap.add_argument("--smoke-subject", type=int); ap.add_argument("--weighted-ce", action="store_true")
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True); set_seed(3407)
    audit = test_pair_sampler(); (args.out / "pair_sampler_unit_test.json").write_text(json.dumps({"passed": True, "synthetic_audit": audit}, indent=2))
    m = pd.read_csv(args.manifest); xe, xf = data_arrays(args.data_dir, m); all_rows=[]; all_predictions=[]; all_embeddings=[]; all_pairs=[]; all_learning=[]
    for value in map(int, args.seeds.split(",")):
        set_seed(value); result = run_loso(xe, xf, m, args.out, value, tuple(args.models.split(",")), args.weighted_ce, args.smoke_subject)
        rows, predictions, embeddings, pairs, learning = result; all_rows.append(rows); all_predictions.append(predictions); all_embeddings += embeddings; all_pairs.append(pairs); all_learning.append(learning)
        print(f"seed={value} rows={len(rows)}", flush=True)
    pd.concat(all_rows, ignore_index=True).to_csv(args.out / "fold_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(args.out / "oof_predictions.csv", index=False)
    (args.out / "oof_embeddings.json").write_text(json.dumps(all_embeddings))
    pd.concat(all_pairs, ignore_index=True).to_csv(args.out / "pair_diagnostics.csv", index=False)
    pd.concat(all_learning, ignore_index=True).to_csv(args.out / "learning_diagnostics.csv", index=False)
    (args.out / "run_config.json").write_text(json.dumps({"seeds": args.seeds, "models": args.models, "weighted_ce": args.weighted_ce, "full_batch": True, "pair_rating": "original valence/arousal", "classification_label": "*_fixed"}, indent=2))


if __name__ == "__main__": main()
