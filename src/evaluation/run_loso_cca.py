#!/usr/bin/env python3
"""A minimal fold-fitted CCA-style cross-modal alignment comparator."""
from __future__ import annotations
import argparse
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

def cells(path,key):
    a=loadmat(path)[key]; return np.stack([np.asarray(x,dtype=float) for x in a.flat])
def prep(): return Pipeline([('impute',SimpleImputer(strategy='median')),('var',VarianceThreshold()),('scale',StandardScaler())])
def metric(y,p):
    return {'accuracy':accuracy_score(y,p>=.5),'uar':balanced_accuracy_score(y,p>=.5),'f1':f1_score(y,p>=.5,zero_division=0),'auc':roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan}
class RegularizedCCA:
    """Closed-form ridge CCA; avoids iterative convergence differences per fold."""
    def __init__(self,n_components=32,ridge=1e-3): self.n_components=n_components; self.ridge=ridge
    @staticmethod
    def invsqrt(a):
        v,u=np.linalg.eigh(a); return (u * (1/np.sqrt(np.maximum(v,1e-10)))) @ u.T
    def fit_transform(self,x,y):
        n=max(len(x)-1,1); ix=self.invsqrt(x.T@x/n+self.ridge*np.eye(x.shape[1])); iy=self.invsqrt(y.T@y/n+self.ridge*np.eye(y.shape[1]))
        u,_,vt=np.linalg.svd(ix@(x.T@y/n)@iy,full_matrices=False); k=min(self.n_components,u.shape[1],vt.shape[0]); self.wx=ix@u[:,:k]; self.wy=iy@vt.T[:,:k]
        return x@self.wx,y@self.wy
    def transform(self,x,y): return x@self.wx,y@self.wy
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--seed',type=int,default=3407); a=ap.parse_args()
    m=pd.read_csv(a.manifest); e=cells(a.data_dir/'Dt_EEGFeatures.mat','EEGFeatures_58'); f=cells(a.data_dir/'Dt_EMOFeatures.mat','EMOFeatures_58'); s=m.subject_id.to_numpy()-1; c=m.clip_id.to_numpy()-1
    xe=np.where(np.isfinite(e[s,c]),e[s,c],np.nan); xf=np.where(np.isfinite(f[s,c]),f[s,c],np.nan); valid=(m.eeg_valid&m.emo_valid).to_numpy(bool)
    rows=[]
    for task in ['valence','arousal']:
      y=m[f'{task}_fixed'].to_numpy(int)
      for held in range(1,59):
        tr=valid&(m.subject_id.to_numpy()!=held); te=valid&(m.subject_id.to_numpy()==held)
        if not te.any() or len(np.unique(y[tr]))<2: continue
        pe,pf=prep(),prep(); et,ft=pe.fit_transform(xe[tr]),pf.fit_transform(xf[tr]); ee,ff=pe.transform(xe[te]),pf.transform(xf[te])
        cca=RegularizedCCA(n_components=32,ridge=1e-3); ze_t,zf_t=cca.fit_transform(et,ft); ze,zf=cca.transform(ee,ff)
        for name,ztr,zte in [('cca_shared',(ze_t+zf_t)/2,(ze+zf)/2),('cca_concat',np.c_[ze_t,zf_t],np.c_[ze,zf])]:
          clf=LogisticRegression(max_iter=1000,solver='liblinear',class_weight='balanced',random_state=a.seed+held).fit(ztr,y[tr]); p=clf.predict_proba(zte)[:,1]
          rows.append({'task':task,'model':'cca_logreg','modality':name,'fold_subject':held,'n_train':int(tr.sum()),'n_test':int(te.sum()),**metric(y[te],p)})
    out=pd.DataFrame(rows); a.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); summary=out.groupby(['task','model','modality'])[['accuracy','uar','f1','auc']].agg(['mean','std','count']); summary.to_csv(a.output.with_name(a.output.stem+'_summary.csv')); print(summary.to_string())
if __name__=='__main__':main()
