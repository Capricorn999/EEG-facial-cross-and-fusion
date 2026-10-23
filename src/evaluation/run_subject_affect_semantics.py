#!/usr/bin/env python3
"""Controlled statistical and few-shot analysis of ASCERTAIN subject effects."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import pearsonr
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEEDS=(3407,2027,911); KS=(0,1,2,4,8)
def stack(p,k): return np.stack([np.asarray(v,dtype=float) for v in loadmat(p)[k].flat])
def pipe(alpha=10.): return Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('ridge',Ridge(alpha=alpha))])
def score(y,p):
    return {'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**.5,
            'r':pearsonr(y,p).statistic if len(y)>2 and np.std(y)>0 and np.std(p)>0 else np.nan}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 m=pd.read_csv(a.manifest); valid=(m.eeg_valid&m.emo_valid&m.valence.notna()&m.arousal.notna()).to_numpy();m=m.loc[valid].reset_index(drop=True)
 e=stack(a.data_dir/'Dt_EEGFeatures.mat','EEGFeatures_58');f=stack(a.data_dir/'Dt_EMOFeatures.mat','EMOFeatures_58');s=m.subject_id.to_numpy()-1;c=m.clip_id.to_numpy()-1
 xe=np.where(np.isfinite(e[s,c]),e[s,c],np.nan); xf=np.where(np.isfinite(f[s,c]),f[s,c],np.nan); xs={'eeg':xe,'face':xf,'fusion':np.c_[xe,xf]}
 # Phase 1 descriptive tables and fixed-effect variance decomposition.
 subject=[]; clip=[]; variance=[]
 for task in ('valence','arousal'):
  y=m[task].to_numpy(float); pop=y.mean()
  subject.extend([{'task':task,'subject_id':sid,'n':len(q),'mean':q[task].mean(),'sd':q[task].std(),'mean_minus_population':q[task].mean()-pop,'class_positive_fraction':(q[f'{task}_fixed']==1).mean()} for sid,q in m.groupby('subject_id')])
  clip.extend([{'task':task,'clip_id':cid,'n':len(q),'mean':q[task].mean(),'median':q[task].median(),'sd':q[task].std(),'iqr':q[task].quantile(.75)-q[task].quantile(.25),'class_positive_fraction':(q[f'{task}_fixed']==1).mean()} for cid,q in m.groupby('clip_id')])
  for name,cols in [('intercept',[]),('clip',['clip_id']),('subject',['subject_id']),('clip_subject',['clip_id','subject_id'])]:
   if not cols: pred=np.full(len(y),pop)
   else:
    design=pd.get_dummies(m[cols].astype(str),drop_first=True); pred=Ridge(alpha=1e-6).fit(design,y).predict(design)
   variance.append({'task':task,'model':name,'in_sample_r2':r2_score(y,pred),'residual_variance':float(np.var(y-pred,ddof=1))})
 pd.DataFrame(subject).to_csv(a.out/'subject_rating_distributions.csv',index=False);pd.DataFrame(clip).to_csv(a.out/'clip_rating_agreement.csv',index=False);pd.DataFrame(variance).to_csv(a.out/'variance_decomposition.csv',index=False)
 # LOSO P0 feature and clip controls, rating plus derived binary measures.
 loso=[]; adaptation=[]
 for task in ('valence','arousal'):
  y=m[task].to_numpy(float); binary=m[f'{task}_fixed'].to_numpy(int)
  for held in sorted(m.subject_id.unique()):
   te=m.subject_id.to_numpy()==held; tr=~te
   if te.sum()<2: continue
   clipmean=m.loc[tr].groupby('clip_id')[task].mean(); fallback=y[tr].mean(); cp=m.clip_id.map(clipmean).fillna(fallback).to_numpy(float)
   for mod,X in xs.items():
    model=pipe().fit(X[tr],y[tr]); p=model.predict(X[te]); row={'task':task,'subject_id':held,'modality':mod,'model':'P0_feature','n_query':int(te.sum()),**score(y[te],p)};loso.append(row)
    for label,pred in [('clip_mean',cp[te]),('global_mean',np.full(te.sum(),fallback))]: loso.append({'task':task,'subject_id':held,'modality':mod,'model':label,'n_query':int(te.sum()),**score(y[te],pred)})
    # matched target support/query protocol.
    target=np.flatnonzero(te)
    for seed in SEEDS:
     rng=np.random.default_rng(seed+held); order=rng.permutation(target)
     for k in KS:
      if len(target)<=k or k>len(target)-2: continue
      support=order[:k]; query=order[k:]
      base=pipe().fit(X[tr],y[tr]); base_sup=base.predict(X[support]) if k else np.array([]); base_q=base.predict(X[query])
      methods={'P0':base_q}
      if k:
       bias=float(np.mean(y[support]-base_sup));methods['P1']=base_q+bias
       if k>=2 and np.std(base_sup)>1e-8: methods['P2']=np.polyfit(base_sup,y[support],1)[0]*base_q+np.polyfit(base_sup,y[support],1)[1]
       else: methods['P2']=base_q+bias
       # Projection is fit only on non-target subjects; residual head is strongly regularized.
       imp=SimpleImputer(strategy='median').fit(X[tr]);sc=StandardScaler().fit(imp.transform(X[tr]));ncomp=min(10,X.shape[1],tr.sum()-1);pc=PCA(n_components=ncomp,random_state=seed).fit(sc.transform(imp.transform(X[tr])))
       zsup=pc.transform(sc.transform(imp.transform(X[support])));zq=pc.transform(sc.transform(imp.transform(X[query])))
       residual=Ridge(alpha=20.).fit(zsup,y[support]-base_sup).predict(zq);methods['P3']=base_q+residual;methods['P4']=.5*methods['P1']+.5*methods['P3']
      for name,pred in methods.items(): adaptation.append({'task':task,'subject_id':held,'modality':mod,'seed':seed,'K':k,'model':name,'n_support':k,'n_query':len(query),**score(y[query],pred)})
   # residual prediction of deviations from training-subject clip means using fusion only
   residual_y=y-cp; model=pipe().fit(xs['fusion'][tr],residual_y[tr]); pred=model.predict(xs['fusion'][te]); loso.append({'task':task,'subject_id':held,'modality':'fusion','model':'P0_clip_residual','n_query':int(te.sum()),**score(residual_y[te],pred)})
 pd.DataFrame(loso).to_csv(a.out/'loso_rating_and_clip_controls.csv',index=False);pd.DataFrame(adaptation).to_csv(a.out/'fewshot_adaptation.csv',index=False)
 # Subject identity probes: random trial split is intentionally an identity-decoding diagnostic, not LOSO affect evaluation.
 probes=[]
 for mod,X in xs.items():
  z=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]).fit_transform(X); yid=m.subject_id.to_numpy(); rng=np.random.default_rng(3407); tr=[];te=[]
  for sid in np.unique(yid):
   ix=np.flatnonzero(yid==sid);rng.shuffle(ix); cut=max(1,int(.8*len(ix)));tr.extend(ix[:cut]);te.extend(ix[cut:])
  clf=LogisticRegression(max_iter=1000,C=1,solver='lbfgs').fit(z[tr],yid[tr]);probes.append({'modality':mod,'n_subjects':len(clf.classes_),'top1':clf.score(z[te],yid[te])})
 pd.DataFrame(probes).to_csv(a.out/'subject_id_probes.csv',index=False)
 # Required figures.
 for task in ('valence','arousal'):
  q=pd.DataFrame(subject).query('task==@task');fig,ax=plt.subplots(figsize=(9,4));ax.bar(q.subject_id,q['mean_minus_population']);ax.axhline(0,color='k',lw=.8);ax.set(title=f'{task}: subject mean minus population mean',xlabel='subject',ylabel='rating difference');fig.tight_layout();fig.savefig(a.out/f'subject_bias_{task}.png',dpi=160);plt.close(fig)
  q=pd.DataFrame(clip).query('task==@task');fig,ax=plt.subplots(figsize=(9,4));ax.bar(q.clip_id,q.sd);ax.set(title=f'{task}: clip disagreement (between-subject SD)',xlabel='clip',ylabel='SD');fig.tight_layout();fig.savefig(a.out/f'clip_disagreement_{task}.png',dpi=160);plt.close(fig)
  q=pd.DataFrame(adaptation).query('task==@task and modality=="fusion"').groupby(['K','model']).mae.agg(['mean','sem']).reset_index();fig,ax=plt.subplots(figsize=(7,4));
  for name,g in q.groupby('model'): ax.errorbar(g.K,g['mean'],yerr=g['sem'],label=name,marker='o')
  ax.set(title=f'{task}: fusion few-shot adaptation',xlabel='target-subject support K',ylabel='query MAE');ax.legend(ncol=3);fig.tight_layout();fig.savefig(a.out/f'adaptation_{task}.png',dpi=160);plt.close(fig)
 (a.out/'run_config.json').write_text(json.dumps({'seeds':SEEDS,'K':KS,'paired_valid_trials':len(m),'protocol':'LOSO target subject; fold-local preprocessing/PCA/clip means'},indent=2))
 print(json.dumps({'paired_trials':len(m),'loso_rows':len(loso),'adaptation_rows':len(adaptation)},indent=2))
if __name__=='__main__': main()
