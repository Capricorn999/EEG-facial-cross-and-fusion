#!/usr/bin/env python3
"""Global frozen-encoder geometry and subject-probe diagnostics for M0/M1/M2."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
from scipy.io import loadmat
from scipy.stats import mannwhitneyu
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from run_affect_contrastive import stack, fit_fold, seed

def effect(a,b):
    u=mannwhitneyu(a,b,alternative='two-sided')
    return float(u.pvalue),float(2*u.statistic/(len(a)*len(b))-1)
def probe(x,y,seedv):
    keep=np.isin(y,[v for v in np.unique(y) if (y==v).sum()>=2]);x,y=x[keep],y[keep]
    ss=StratifiedShuffleSplit(1,test_size=.2,random_state=seedv);tr,te=next(ss.split(x,y))
    clf=LogisticRegression(max_iter=1000,solver='lbfgs',random_state=seedv).fit(x[tr],y[tr]);p=clf.predict_proba(x[te]);pred=clf.classes_[p.argmax(1)]
    return len(clf.classes_),accuracy_score(y[te],pred),top_k_accuracy_score(y[te],p,labels=clf.classes_,k=min(5,len(clf.classes_))),f1_score(y[te],pred,average='macro',zero_division=0)
def pairs(ze,zf,m,task,rng):
    y=m[f'{task}_fixed'].to_numpy();s=m.subject_id.to_numpy();n=len(m);out={'A':[], 'B':[], 'C':[], 'D':[]}
    for i in range(n):
        opts={'A':np.array([i]),'B':np.flatnonzero((s==s[i])&(np.abs(y-y[i])>=1)),'C':np.flatnonzero((s!=s[i])&(np.abs(y-y[i])==0)),'D':np.flatnonzero((s!=s[i])&(np.abs(y-y[i])>=1))}
        for k,v in opts.items():
            if len(v):out[k].append(float(ze[i]@zf[rng.choice(v)]))
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=3407);a=ap.parse_args();seed(a.seed);a.out.mkdir(parents=True,exist_ok=True)
    dev='cuda' if torch.cuda.is_available() else 'cpu';m=pd.read_csv(a.manifest);mask=(m.eeg_valid&m.emo_valid&m.valence.notna()&m.arousal.notna()).to_numpy();m=m.loc[mask].reset_index(drop=True)
    e=stack(a.data_dir/'Dt_EEGFeatures.mat','EEGFeatures_58');f=stack(a.data_dir/'Dt_EMOFeatures.mat','EMOFeatures_58');ss=m.subject_id.to_numpy()-1;cc=m.clip_id.to_numpy()-1;xe=np.where(np.isfinite(e[ss,cc]),e[ss,cc],np.nan);xf=np.where(np.isfinite(f[ss,cc]),f[ss,cc],np.nan)
    rng=np.random.default_rng(a.seed);val_subjects=set(rng.choice(np.unique(m.subject_id),size=5,replace=False));val=m.subject_id.isin(val_subjects).to_numpy();train=~val;test=np.ones(len(m),bool);similarity=[];probes=[]
    for task in ['valence','arousal']:
        for mode in ['M0','M1','M2']:
            _,ze,zf=fit_fold(xe,xf,m,task,mode,train,val,test,dev,a.seed);zs=(ze+zf)/2;nsub,t1,t5,f1=probe(zs,m.subject_id.to_numpy(),a.seed)
            probes.append({'task':task,'model':mode,'representation':'shared','n_subjects':nsub,'subject_top1':t1,'subject_top5':t5,'subject_macro_f1':f1,'note':'global encoder trained with held-out validation subjects; frozen probe'})
            ps=pairs(ze,zf,m,task,rng)
            for typ,v in ps.items():similarity.append({'task':task,'model':mode,'pair_type':typ,'n_pairs':len(v),'mean_similarity':np.mean(v),'median_similarity':np.median(v),'std_similarity':np.std(v,ddof=1)})
            pcd,ecd=effect(ps['C'],ps['D']);pcb,ecb=effect(ps['C'],ps['B'])
            similarity += [{'task':task,'model':mode,'pair_type':'C_vs_D_test','n_pairs':min(len(ps['C']),len(ps['D'])),'mean_similarity':np.nan,'median_similarity':pcd,'std_similarity':ecd},{'task':task,'model':mode,'pair_type':'C_vs_B_test','n_pairs':min(len(ps['C']),len(ps['B'])),'mean_similarity':np.nan,'median_similarity':pcb,'std_similarity':ecb}]
            fig,ax=plt.subplots(figsize=(7,4));ax.boxplot([ps[k] for k in 'ABCD'],tick_labels=['A exact','B same-subj diff-aff','C diff-subj same-aff','D diff-subj diff-aff'],showfliers=False);ax.set(title=f'{task} {mode}: cross-modal cosine similarity',ylabel='cosine similarity');fig.tight_layout();fig.savefig(a.out/f'similarity_{task}_{mode}.png',dpi=160);plt.close(fig)
    pd.DataFrame(probes).to_csv(a.out/'subject_probe_results.csv',index=False);pd.DataFrame(similarity).to_csv(a.out/'pair_similarity_affect_contrastive.csv',index=False);print(pd.DataFrame(probes).to_string(index=False))
if __name__=='__main__':main()
