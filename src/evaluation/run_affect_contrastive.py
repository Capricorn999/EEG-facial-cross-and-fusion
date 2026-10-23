#!/usr/bin/env python3
"""LOSO M0/M1/M2 affect-aware cross-modal contrastive pilot.

Main models are intentionally tiny.  All fitted preprocessing, validation
selection, pair construction and high-consensus clip selection are fold-local.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.io import loadmat
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu

def seed(v): random.seed(v);np.random.seed(v);torch.manual_seed(v);torch.cuda.manual_seed_all(v)
def stack(p,k):
 a=loadmat(p)[k];return np.stack([np.asarray(x,dtype=np.float32) for x in a.flat])
def prep():return Pipeline([('impute',SimpleImputer(strategy='median')),('var',VarianceThreshold()),('scale',StandardScaler())])
class Encoder(torch.nn.Module):
 def __init__(self,d):super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(d,64),torch.nn.LayerNorm(64),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(64,32))
 def forward(self,x):return torch.nn.functional.normalize(self.net(x),dim=1)
class Net(torch.nn.Module):
 def __init__(self,de,df):super().__init__();self.e,self.f=Encoder(de),Encoder(df);self.head=torch.nn.Linear(32,2)
 def forward(self,e,f):
  ze,zf=self.e(e),self.f(f);return ze,zf,self.head((ze+zf)/2)
def pair_indices(subject,label,mode,rng):
 n=len(subject);pos=np.full(n,-1);neg=np.full(n,-1);hard=np.full(n,-1)
 for i in range(n):
  other=np.arange(n); c=other[(subject!=subject[i])&(np.abs(label-label[i])==0)]
  d=other[(subject!=subject[i])&(np.abs(label-label[i])>=3)]
  b=other[(subject==subject[i])&(np.abs(label-label[i])>=3)]
  if len(c):pos[i]=rng.choice(c)
  if len(d):neg[i]=rng.choice(d)
  if len(b):hard[i]=rng.choice(b)
 return pos,neg,hard
def losses(ze,zf,logits,y,mode,indices):
 ce=torch.nn.functional.cross_entropy(logits,y); temp=.15; exact=torch.nn.functional.cross_entropy(ze@zf.T/temp,torch.arange(len(y),device=y.device)); total=ce+.25*exact
 if mode!='M0':
  pos,neg,hard=indices
  def attractive(ii,w):
   mask=ii>=0
   return w*(1-(ze[mask]*zf[ii[mask]]).sum(1)).mean() if mask.any() else 0.
  def repulsive(ii,w):
   mask=ii>=0
   return w*torch.relu((ze[mask]*zf[ii[mask]]).sum(1)-.2).mean() if mask.any() else 0.
  total=total+attractive(pos,.15)+repulsive(neg,.15)
  if mode=='M2':total=total+repulsive(hard,.15)
 return total
def metrics(y,p):return {'accuracy':accuracy_score(y,p>=.5),'uar':balanced_accuracy_score(y,p>=.5),'f1':f1_score(y,p>=.5,zero_division=0),'auc':roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan}
def consensus_clips(meta,train,task,condition):
 if condition not in ('consensus_half','support15_consensus_half'):return np.ones(len(meta),bool)
 st=meta.loc[train].groupby('clip_id')[task].std();good=set(st.nsmallest(max(1,len(st)//2)).index);return meta.clip_id.isin(good).to_numpy()
def fit_fold(xe,xf,meta,task,mode,train,val,test,device,seedv):
 pe,pf=prep(),prep(); et=torch.tensor(pe.fit_transform(xe[train]),device=device);ft=torch.tensor(pf.fit_transform(xf[train]),device=device);vt=(torch.tensor(pe.transform(xe[val]),device=device),torch.tensor(pf.transform(xf[val]),device=device));tt=(torch.tensor(pe.transform(xe[test]),device=device),torch.tensor(pf.transform(xf[test]),device=device))
 y=meta[f'{task}_fixed'].to_numpy(int);yt=torch.tensor(y[train],device=device);yv=torch.tensor(y[val],device=device);rng=np.random.default_rng(seedv);pi=pair_indices(meta.subject_id.to_numpy()[train],y[train],mode,rng);pi=tuple(torch.tensor(x,device=device) for x in pi)
 net=Net(et.shape[1],ft.shape[1]).to(device);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4);best=None;bad=0
 for epoch in range(40):
  net.train();ze,zf,lo=net(et,ft);loss=losses(ze,zf,lo,yt,mode,pi);opt.zero_grad();loss.backward();opt.step()
  net.eval()
  with torch.no_grad():vl=torch.nn.functional.cross_entropy(net(*vt)[2],yv).item()
  if best is None or vl<best[0]-1e-4:best=(vl,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()});bad=0
  else:bad+=1
  if bad>=8:break
 net.load_state_dict(best[1]);net.eval()
 with torch.no_grad():ze,zf,lo=net(*tt);prob=torch.softmax(lo,1)[:,1].cpu().numpy();return metrics(y[test],prob),ze.cpu().numpy(),zf.cpu().numpy()
def run_condition(xe,xf,m,task,mode,condition,device,seedv):
 paired=(m.eeg_valid&m.emo_valid&m.valence.notna()&m.arousal.notna()).to_numpy(bool);support=m.groupby('subject_id').apply(lambda x:int((x.eeg_valid&x.emo_valid).sum()),include_groups=False)
 rows=[];emb=[]
 for held in range(1,59):
  cond=np.ones(len(m),bool)
  if condition in ('support15','support15_consensus_half'):cond=m.subject_id.map(support).to_numpy()>=15
  raw_train=paired&cond&(m.subject_id.to_numpy()!=held); raw_test=paired&cond&(m.subject_id.to_numpy()==held)
  clipok=consensus_clips(m,raw_train,task,condition);raw_train&=clipok;raw_test&=clipok
  subjects=np.unique(m.subject_id.to_numpy()[raw_train]); rng=np.random.default_rng(seedv+held); vv=set(rng.choice(subjects,size=max(1,round(.1*len(subjects))),replace=False));val=raw_train&m.subject_id.isin(vv).to_numpy();train=raw_train&~val;test=raw_test
  if train.sum()<20 or val.sum()<2 or test.sum()<1 or len(np.unique(m.loc[train,f'{task}_fixed']))<2:continue
  out,ze,zf=fit_fold(xe,xf,m,task,mode,train,val,test,device,seedv+held);rows.append({'condition':condition,'task':task,'model':mode,'fold_subject':held,'n_train':int(train.sum()),'n_val':int(val.sum()),'n_test':int(test.sum()),**out})
  for j,k in enumerate(np.flatnonzero(test)):emb.append({'condition':condition,'task':task,'model':mode,'subject_id':int(m.subject_id.iloc[k]),'clip_id':int(m.clip_id.iloc[k]),'label':int(m[f'{task}_fixed'].iloc[k]),'ze':ze[j].tolist(),'zf':zf[j].tolist()})
 return pd.DataFrame(rows),emb
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--conditions',default='all');ap.add_argument('--modes',default='M0,M1,M2');ap.add_argument('--seed',type=int,default=3407);a=ap.parse_args();seed(a.seed);a.out.mkdir(parents=True,exist_ok=True);device='cuda' if torch.cuda.is_available() else 'cpu';m=pd.read_csv(a.manifest);e=stack(a.data_dir/'Dt_EEGFeatures.mat','EEGFeatures_58');f=stack(a.data_dir/'Dt_EMOFeatures.mat','EMOFeatures_58');s=m.subject_id.to_numpy()-1;c=m.clip_id.to_numpy()-1;xe=np.where(np.isfinite(e[s,c]),e[s,c],np.nan);xf=np.where(np.isfinite(f[s,c]),f[s,c],np.nan)
 conditions=['all','support15','consensus_half','support15_consensus_half'] if a.conditions=='all' else [a.conditions];rows=[];emb=[]
 for condition in conditions:
  for task in ['valence','arousal']:
   for mode in a.modes.split(','):
    r,z=run_condition(xe,xf,m,task,mode,condition,device,a.seed);rows.append(r);emb+=z;print(condition,task,mode,len(r),r.uar.mean() if len(r) else None,flush=True)
 out=pd.concat(rows,ignore_index=True);out.to_csv(a.out/'model_results.csv',index=False);(a.out/'oof_embeddings.json').write_text(json.dumps(emb));print('done',device,len(out))
if __name__=='__main__':main()
