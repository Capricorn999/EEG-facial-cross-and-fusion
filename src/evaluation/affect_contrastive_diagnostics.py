#!/usr/bin/env python3
"""Immutable-manifest diagnostics for the affect-aware contrastive campaign."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 m=pd.read_csv(a.manifest); m['paired_valid']=m.eeg_valid&m.emo_valid&m.valence.notna()&m.arousal.notna()
 support=m.groupby('subject_id').agg(valid_eeg_trials=('eeg_valid','sum'),valid_face_trials=('emo_valid','sum'),valid_paired_trials=('paired_valid','sum')).reset_index();support['paired_valid_percent']=100*support.valid_paired_trials/36
 for t in (15,20,25):support[f'eligible_ge_{t}']=support.valid_paired_trials>=t
 support.to_csv(a.out/'paired_support_by_subject.csv',index=False)
 fig,ax=plt.subplots(figsize=(6,4));ax.hist(support.valid_paired_trials,bins=np.arange(-.5,37.5,1),color='#3969ac',edgecolor='white');ax.set(xlabel='Valid paired trials per subject',ylabel='Subjects',title='Paired support distribution');fig.tight_layout();fig.savefig(a.out/'paired_support_histogram.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,4));ax.bar(support.subject_id,support.valid_paired_trials,color='#3969ac');[ax.axhline(t,color=c,ls='--',label=f'>={t}') for t,c in [(15,'#e68310'),(20,'#c13a37'),(25,'#6f4c9b')]];ax.set(xlabel='Subject ID',ylabel='Valid paired trials',title='Paired support by subject');ax.legend();fig.tight_layout();fig.savefig(a.out/'paired_support_by_subject.png',dpi=180);plt.close(fig)
 consensus=m.groupby('clip_id').agg(valence_mean=('valence','mean'),valence_std=('valence','std'),valence_iqr=('valence',lambda x:x.quantile(.75)-x.quantile(.25)),valence_n=('valence','count'),arousal_mean=('arousal','mean'),arousal_std=('arousal','std'),arousal_iqr=('arousal',lambda x:x.quantile(.75)-x.quantile(.25)),arousal_n=('arousal','count')).reset_index()
 for task in ['valence','arousal']:
  consensus[f'{task}_consensus_rank']=consensus[f'{task}_std'].rank(method='first').astype(int);consensus[f'{task}_high_half']=consensus[f'{task}_std']<=consensus[f'{task}_std'].median();consensus[f'{task}_high_third']=consensus[f'{task}_std']<=consensus[f'{task}_std'].quantile(1/3)
  fig,ax=plt.subplots(figsize=(11,4));q=consensus.sort_values(f'{task}_std');ax.bar(q.clip_id.astype(str),q[f'{task}_std'],color='#3969ac');ax.axhline(q[f'{task}_std'].median(),color='#e68310',ls='--',label='high-consensus half cutoff');ax.set(xlabel='Clip ID (ranked by consistency)',ylabel='Across-subject rating SD',title=f'{task.capitalize()} inter-subject consistency');ax.legend();fig.tight_layout();fig.savefig(a.out/f'{task}_clip_consistency.png',dpi=180);plt.close(fig)
 consensus.to_csv(a.out/'clip_affect_consensus.csv',index=False)
 summary={'paired_support':support.valid_paired_trials.describe(percentiles=[.25,.5,.75]).to_dict(),'subjects_per_threshold':{str(t):int((support.valid_paired_trials>=t).sum()) for t in (15,20,25)},'clips_high_half':{x:int(consensus[f'{x}_high_half'].sum()) for x in ['valence','arousal']},'clips_high_third':{x:int(consensus[f'{x}_high_third'].sum()) for x in ['valence','arousal']}}
 (a.out/'diagnostic_summary.json').write_text(pd.Series(summary).to_json(indent=2));print(summary)
if __name__=='__main__':main()
