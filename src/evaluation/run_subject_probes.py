#!/usr/bin/env python3
"""Raw-feature subject-identification diagnostic with within-subject trial split."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def stack(path: Path, key: str) -> np.ndarray:
    cells = loadmat(path)[key]
    return np.stack([np.asarray(x, dtype=float) for x in cells.flat])


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument('--data-dir', type=Path, required=True); ap.add_argument('--manifest', type=Path, required=True); ap.add_argument('--output', type=Path, required=True); ap.add_argument('--seed', type=int, default=3407)
    a = ap.parse_args(); m = pd.read_csv(a.manifest)
    eeg = stack(a.data_dir/'Dt_EEGFeatures.mat', 'EEGFeatures_58'); emo = stack(a.data_dir/'Dt_EMOFeatures.mat', 'EMOFeatures_58')
    s, c = m.subject_id.to_numpy()-1, m.clip_id.to_numpy()-1
    xe, xf = eeg[s,c], emo[s,c]
    arrays = {'raw_eeg': (xe, m.eeg_valid.to_numpy(bool)), 'raw_face': (xf, m.emo_valid.to_numpy(bool)), 'early_fusion': (np.c_[xe,xf], (m.eeg_valid&m.emo_valid).to_numpy(bool))}
    rows=[]
    for name,(x,valid) in arrays.items():
        x=np.where(np.isfinite(x),x,np.nan); y=m.subject_id.to_numpy()[valid]
        x=x[valid]; classes=np.unique(y)
        # A stratified trial split deliberately measures retained identity, not
        # cross-subject generalization. Subjects with <2 usable trials cannot
        # enter this diagnostic and are listed in the output.
        keep=np.isin(y, classes[np.array([np.sum(y==k) >= 2 for k in classes])]); x,y=x[keep],y[keep]; classes=np.unique(y)
        splitter=StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=a.seed)
        tr,te=next(splitter.split(x,y))
        pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('variance',VarianceThreshold()),('scale',StandardScaler()),('model',LogisticRegression(max_iter=1000,solver='lbfgs',multi_class='multinomial',C=1.0,random_state=a.seed))])
        pipe.fit(x[tr],y[tr]); prob=pipe.predict_proba(x[te]); pred=pipe.classes_[np.argmax(prob,axis=1)]
        rows.append({'representation':name,'n_subjects':len(classes),'chance_top1':1/len(classes),'n_train':len(tr),'n_test':len(te),'subject_top1':accuracy_score(y[te],pred),'subject_top5':top_k_accuracy_score(y[te],prob,labels=pipe.classes_,k=min(5,len(classes))),'subject_macro_f1':f1_score(y[te],pred,average='macro',zero_division=0)})
    out=pd.DataFrame(rows); a.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(out.to_string(index=False))

if __name__=='__main__': main()
