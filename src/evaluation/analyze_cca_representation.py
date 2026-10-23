#!/usr/bin/env python3
"""Exploratory representation diagnostic for the fitted linear aligned space.

This is deliberately marked exploratory: the CCA transform is fitted on all
paired trials for visualization/probing, while emotion metrics remain LOSO.
"""
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
from run_loso_cca import RegularizedCCA

def stack(p,k):
 a=loadmat(p)[k]; return np.stack([np.asarray(x,dtype=float) for x in a.flat])
def pre(): return Pipeline([('imp',SimpleImputer(strategy='median')),('var',VarianceThreshold()),('scale',StandardScaler())])
def cosine(a,b): return np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-12)
def subject_probe(name,x,y,seed):
 ss=StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=seed); tr,te=next(ss.split(x,y)); clf=LogisticRegression(max_iter=1000,solver='lbfgs',C=1.0,random_state=seed).fit(x[tr],y[tr]); p=clf.predict_proba(x[te]); pred=clf.classes_[p.argmax(1)]
 return {'representation':name,'subject_top1':accuracy_score(y[te],pred),'subject_top5':top_k_accuracy_score(y[te],p,labels=clf.classes_,k=min(5,len(clf.classes_))),'subject_macro_f1':f1_score(y[te],pred,average='macro',zero_division=0),'n_subjects':len(clf.classes_),'note':'global unsupervised CCA transform; exploratory probe'}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--results-dir',type=Path,required=True);ap.add_argument('--seed',type=int,default=3407);a=ap.parse_args();a.results_dir.mkdir(parents=True,exist_ok=True)
 m=pd.read_csv(a.manifest); e=stack(a.data_dir/'Dt_EEGFeatures.mat','EEGFeatures_58');f=stack(a.data_dir/'Dt_EMOFeatures.mat','EMOFeatures_58');s=m.subject_id.to_numpy()-1;c=m.clip_id.to_numpy()-1;valid=(m.eeg_valid&m.emo_valid).to_numpy(bool);m=m.loc[valid].reset_index(drop=True);xe=np.where(np.isfinite(e[s[valid],c[valid]]),e[s[valid],c[valid]],np.nan);xf=np.where(np.isfinite(f[s[valid],c[valid]]),f[s[valid],c[valid]],np.nan)
 pe,pf=pre(),pre();xe,xf=pe.fit_transform(xe),pf.fit_transform(xf); cca=RegularizedCCA(32,1e-3);ze,zf=cca.fit_transform(xe,xf);zs=(ze+zf)/2
 y=m.subject_id.to_numpy(); keep=np.isin(y,[k for k in np.unique(y) if np.sum(y==k)>=2]); y=y[keep]
 probes=pd.DataFrame([subject_probe('cca_eeg_latent',ze[keep],y,a.seed),subject_probe('cca_face_latent',zf[keep],y,a.seed),subject_probe('cca_shared_latent',zs[keep],y,a.seed),subject_probe('cca_concat_latent',np.c_[ze[keep],zf[keep]],y,a.seed)]);probes.to_csv(a.results_dir/'cca_subject_probe_results.csv',index=False)
 rng=np.random.default_rng(a.seed);n=len(m); records=[]
 def sampled(category,choices):
  ii=[];jj=[]
  for i in rng.permutation(n):
   opts=choices(i)
   if len(opts): ii.append(i);jj.append(rng.choice(opts))
   if len(ii)>=min(10000,n):break
  q=cosine(ze[np.array(ii)],zf[np.array(jj)]);records.append({'pair_type':category,'n_pairs':len(q),'mean_similarity':q.mean(),'std_similarity':q.std(ddof=1),'median_similarity':np.median(q)})
 subj=m.subject_id.to_numpy();clip=m.clip_id.to_numpy();val=m.valence.to_numpy();aro=m.arousal.to_numpy()
 sampled('same_subject_same_clip',lambda i:np.array([i]))
 sampled('same_subject_different_clip',lambda i:np.flatnonzero((subj==subj[i])&(clip!=clip[i])))
 sampled('different_subject_same_clip',lambda i:np.flatnonzero((subj!=subj[i])&(clip==clip[i])))
 sampled('different_subject_different_clip',lambda i:np.flatnonzero((subj!=subj[i])&(clip!=clip[i])))
 sampled('different_subject_similar_affect',lambda i:np.flatnonzero((subj!=subj[i])&(np.abs(val-val[i])<=1)&(np.abs(aro-aro[i])<=1)))
 pd.DataFrame(records).to_csv(a.results_dir/'pair_similarity_analysis.csv',index=False);print(probes.to_string(index=False));print(pd.DataFrame(records).to_string(index=False))
if __name__=='__main__':main()
