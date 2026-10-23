#!/usr/bin/env python3
"""Unseen-subject geometry and frozen training-subject identity probes for v2."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
sys.path.insert(0, str(Path(__file__).parent))
from run_affect_contrastive_v2 import data_arrays, fit_fold, set_seed, valid_mask


def subject_split(m, held_subjects, seed):
    valid = valid_mask(m); held_subjects = set(held_subjects)
    available = np.array(sorted(set(m.subject_id[valid]) - held_subjects))
    rng = np.random.default_rng(seed); val_subjects = set(rng.choice(available, size=max(1, round(.1 * len(available))), replace=False))
    test = valid & m.subject_id.isin(held_subjects).to_numpy(); val = valid & m.subject_id.isin(val_subjects).to_numpy(); train = valid & ~test & ~val
    return train, val, test


def sampled_pairs(ze, zf, m, task, rng):
    rating=m[task].to_numpy(int); subject=m.subject_id.to_numpy(int); clip=m.clip_id.to_numpy(int); rows=[]
    for i in range(len(m)):
        delta=np.abs(rating-rating[i]); same=subject==subject[i]; different=~same
        pools={"A":np.array([i]), "B":np.flatnonzero(same & (np.arange(len(m)) != i) & (delta >= 3)),
               "C":np.flatnonzero(different & (delta == 0)), "D":np.flatnonzero(different & (delta >= 3))}
        for category, candidates in pools.items():
            if candidates.size:
                j=int(rng.choice(candidates)); rows.append({"subject_id":subject[i],"category":category,"same_clip":bool(clip[i]==clip[j]),"similarity":float(ze[i]@zf[j])})
    return rows


def probe_train_subjects(net, pe, pf, xe, xf, m, train, seed):
    net.eval()
    with torch.no_grad():
        device=next(net.parameters()).device
        e=torch.tensor(pe.transform(xe[train]),dtype=torch.float32,device=device)
        f=torch.tensor(pf.transform(xf[train]),dtype=torch.float32,device=device)
        z=((net(e,f)[0]+net(e,f)[1])/2).cpu().numpy()
    y=m.loc[train,'subject_id'].to_numpy(); counts=pd.Series(y).value_counts(); keep=np.isin(y,counts[counts>=2].index); z,y=z[keep],y[keep]
    split=StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=seed); tr,te=next(split.split(z,y))
    clf=LogisticRegression(max_iter=1000,solver='lbfgs',random_state=seed).fit(z[tr],y[tr]); pred=clf.predict(z[te])
    return {"n_subjects":int(len(clf.classes_)),"top1":accuracy_score(y[te],pred),"macro_f1":f1_score(y[te],pred,average='macro',zero_division=0)}


def bootstrap_subject_difference(subject_values, left, right, rng, n=2000):
    wide=subject_values.pivot_table(index='subject_id',columns='category',values='similarity',aggfunc='mean')
    if left not in wide or right not in wide: return np.nan,np.nan,np.nan,0
    wide=wide.dropna(subset=[left,right])
    diff=(wide[left]-wide[right]).to_numpy(); point=float(diff.mean()) if len(diff) else np.nan
    boot=np.array([rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(n)]) if len(diff) else np.array([np.nan])
    return point,float(np.quantile(boot,.025)),float(np.quantile(boot,.975)),len(diff)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--seeds',default='3407,2027,911'); ap.add_argument('--groups',type=int,default=6)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); m=pd.read_csv(a.manifest); xe,xf=data_arrays(a.data_dir,m); subjects=np.array(sorted(m.loc[valid_mask(m),'subject_id'].unique())); groups=np.array_split(subjects,a.groups); device='cuda' if torch.cuda.is_available() else 'cpu'
    pairs=[]; probes=[]
    for seed in map(int,a.seeds.split(',')):
      for group_id,held in enumerate(groups):
        train,val,test=subject_split(m,held,seed+group_id)
        for task in ('valence','arousal'):
          if np.unique(m.loc[train,f'{task}_fixed']).size<2: continue
          for mode in ('M0','M1','M2'):
            set_seed(seed+group_id); result=fit_fold(xe,xf,m,task,mode,train,val,test,device,seed+group_id,return_fitted=True)
            _,_,ze,zf,_,_,_,_,_,_,_,net,pe,pf=result
            test_meta=m.loc[test].reset_index(drop=True); rng=np.random.default_rng(seed*100+group_id)
            for row in sampled_pairs(ze,zf,test_meta,task,rng): pairs.append({"seed":seed,"group":group_id,"task":task,"model":mode,**row})
            probe=probe_train_subjects(net,pe,pf,xe,xf,m,train,seed+group_id); probes.append({"seed":seed,"group":group_id,"task":task,"model":mode,**probe})
        print(f'seed={seed} group={group_id}',flush=True)
    pairs=pd.DataFrame(pairs); probes=pd.DataFrame(probes); pairs.to_csv(a.out/'unseen_subject_geometry_pairs.csv',index=False); probes.to_csv(a.out/'frozen_train_subject_probe.csv',index=False)
    summary=[]; rng=np.random.default_rng(991)
    for (seed,task,model),q in pairs.groupby(['seed','task','model']):
      for clip_scope,qq in [('all',q),('same_clip',q[q.same_clip]),('different_clip',q[~q.same_clip])]:
        if qq.empty: continue
        means=qq.groupby('category').similarity.agg(['mean','median','count']).reset_index()
        for _,r in means.iterrows(): summary.append({"seed":seed,"task":task,"model":model,"clip_scope":clip_scope,"measure":r.category,"mean":r['mean'],"median":r['median'],"n_pairs":int(r['count']),"n_subjects":qq.subject_id.nunique()})
        for left,right,name in [('C','D','C_minus_D'),('C','B','C_minus_B')]:
          point,lo,hi,n=bootstrap_subject_difference(qq,left,right,rng); summary.append({"seed":seed,"task":task,"model":model,"clip_scope":clip_scope,"measure":name,"mean":point,"ci95_low":lo,"ci95_high":hi,"n_subjects":n})
    pd.DataFrame(summary).to_csv(a.out/'unseen_subject_geometry_summary.csv',index=False)


if __name__=='__main__': main()
