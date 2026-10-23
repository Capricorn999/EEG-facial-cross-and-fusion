#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
def rb(a,b):
 d=np.asarray(a)-np.asarray(b);d=d[d!=0]
 if not len(d):return 0.
 r=pd.Series(np.abs(d)).rank().to_numpy();return float((r[d>0].sum()-r[d<0].sum())/r.sum())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();o=a.root/'results/affect_contrastive';m=pd.read_csv(o/'model_results.csv');probe=pd.read_csv(o/'affect_contrastive_subject_probe_results.csv');base=pd.read_csv(a.root/'results/metrics/baseline_results.csv')
 old=pd.read_csv(a.root/'results/metrics/representation_probe_results.csv').rename(columns={'emotion_task':'task','representation':'model'})
 old['representation']=old['model'];old['note']=old['probe_note'];old=old[['task','model','representation','subject_top1','subject_top5','subject_macro_f1','note']]
 new=probe[['task','model','representation','subject_top1','subject_top5','subject_macro_f1','note']]
 pd.concat([old,new],ignore_index=True).to_csv(o/'subject_probe_results.csv',index=False)
 sens=m.groupby(['condition','task','model']).agg(n_folds=('fold_subject','nunique'),accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),uar_mean=('uar','mean'),uar_std=('uar','std'),f1_mean=('f1','mean'),auc_mean=('auc','mean')).reset_index();sens.to_csv(o/'sensitivity_analysis.csv',index=False)
 tests=[]
 for task in ['valence','arousal']:
  pivot=m[(m.condition=='all')&(m.task==task)].pivot(index='fold_subject',columns='model',values='uar')
  ef=base[(base.task==task)&(base.model=='logreg')&(base.modality=='early_fusion')][['fold_subject','uar']].set_index('fold_subject').rename(columns={'uar':'EarlyFusion'})
  pivot=pivot.join(ef)
  for x,y in [('M1','M0'),('M2','M0'),('M2','M1'),('M1','EarlyFusion'),('M2','EarlyFusion')]:
   q=pivot[[x,y]].dropna();
   if np.allclose(q[x],q[y]): p=1.
   else:
    try:p=wilcoxon(q[x],q[y],method='auto').pvalue
    except ValueError:p=1.
   tests.append({'task':task,'comparison':f'{x} vs {y}','n_paired_subjects':len(q),'mean_difference_uar':(q[x]-q[y]).mean(),'wilcoxon_p':p,'rank_biserial':rb(q[x],q[y])})
 pd.DataFrame(tests).to_csv(o/'statistical_tests.csv',index=False)
 # UAR/leakage plot uses all-eligible UAR and frozen global probe for neural models.
 points=[]
 for task in ['valence','arousal']:
  for model in ['M0','M1','M2']:
   points.append({'task':task,'model':model,'uar':m[(m.condition=='all')&(m.task==task)&(m.model==model)].uar.mean(),'leakage':probe[(probe.task==task)&(probe.model==model)].subject_top1.iloc[0]})
 fig,axs=plt.subplots(1,2,figsize=(9,4))
 for ax,task in zip(axs,['valence','arousal']):
  q=pd.DataFrame(points).query('task==@task')
  ax.scatter(q.leakage,q.uar,s=60,color='#3969ac');[ax.annotate(r.model,(r.leakage,r.uar),xytext=(4,4),textcoords='offset points') for _,r in q.iterrows()];ax.set(title=task.capitalize(),xlabel='Frozen subject-probe top-1',ylabel='LOSO UAR');ax.axhline(.5,color='gray',lw=.8);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(o/'uar_vs_subject_leakage.png',dpi=180);plt.close(fig)
 pair=pd.read_csv(o/'pair_similarity_affect_contrastive.csv');sim=pair[pair.pair_type.isin(['A','B','C','D'])].pivot(index=['task','model'],columns='pair_type',values='mean_similarity').reset_index()
 ptest=pair[pair.pair_type.str.contains('_test')][['task','model','pair_type','median_similarity','std_similarity']].copy().rename(columns={'median_similarity':'p_value','std_similarity':'rank_biserial'})
 lines=['# Affect-aware cross-subject contrastive report','','## Locked scope','', 'This follow-up preserves the original manifest, quality filter, fixed labels and LOSO evaluation. It evaluates lightweight 88→64→32 EEG and 72→64→32 face encoders: M0 exact-pair neural contrastive control, M1 with cross-subject same-rating affect positives, and M2 with additional same-subject different-affect hard negatives.','', '## Diagnostic findings','', '- Paired support is substantially uneven: min 0, Q1 5, median 23, mean 20.3, Q3 36, max 36 trials per subject. 38, 32, and 26 subjects meet ≥15, ≥20, and ≥25 support respectively.','- Clip-consensus subsets were set by the lowest fold-local training-subject rating SD; they are sensitivity analyses, not performance-selected subsets.','', '## Main all-eligible LOSO results','', '| Task | M0 UAR | M1 UAR | M2 UAR | M0 ID top-1 | M1 ID top-1 | M2 ID top-1 |','|---|---:|---:|---:|---:|---:|---:|']
 for task in ['valence','arousal']:
  q=m[(m.condition=='all')&(m.task==task)].groupby('model').uar.mean();p=probe[probe.task==task].set_index('model').subject_top1;lines.append(f'| {task} | {q.M0:.3f} | {q.M1:.3f} | {q.M2:.3f} | {p.M0:.3f} | {p.M1:.3f} | {p.M2:.3f} |')
 lines += ['','M1/M2 reduce frozen-probe identity relative to M0, but none approaches chance (~2.17% for 46 probe subjects).','', '## Geometry result','', '| Task / model | A exact | B same-subject different-affect | C different-subject same-affect | D different-subject different-affect |','|---|---:|---:|---:|---:|']
 for _,r in sim.iterrows():lines.append(f'| {r.task} {r.model} | {r.A:.3f} | {r.B:.3f} | {r.C:.3f} | {r.D:.3f} |')
 lines += ['', 'Across all six trained task/model combinations, C was not significantly greater than D; C also remained below B except for the weak/non-significant valence M2 contrast. Thus the requested affect-over-subject geometry was not achieved.','', '## Paired statistical tests','', '| Task | Comparison | n | ΔUAR | Wilcoxon p | rank-biserial |','|---|---:|---:|---:|---:|---:|']
 for _,r in pd.DataFrame(tests).iterrows():lines.append(f'| {r.task} | {r.comparison} | {r.n_paired_subjects} | {r.mean_difference_uar:.3f} | {r.wilcoxon_p:.3g} | {r.rank_biserial:.3f} |')
 lines += ['', '## Decision','', '**Not supported; stop before adversarial or more complex modeling.** Cross-subject affective positives reduced subject-probe accuracy modestly but did not improve cross-subject affect geometry (C>D) or yield a statistically supported LOSO benefit. M2 did not improve M1, consistent with an ineffective hard-negative term at the locked margin.','', 'A future route should be a targeted loss-design diagnosis—not GRL—such as stronger/multiple hard negatives or continuous affect-distance weights, first validated by an in-training pair-loss audit.']
 (a.root/'reports'/'affect_contrastive_report.md').write_text('\n'.join(lines)+'\n')
 print(pd.DataFrame(tests).to_string(index=False))
if __name__=='__main__':main()
