#!/usr/bin/env python3
from pathlib import Path
import argparse, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();o=a.root/'results/subject_affect_semantics';r=a.root/'reports'
 sub=pd.read_csv(o/'subject_rating_distributions.csv');clip=pd.read_csv(o/'clip_rating_agreement.csv');var=pd.read_csv(o/'variance_decomposition.csv');loso=pd.read_csv(o/'loso_rating_and_clip_controls.csv');ad=pd.read_csv(o/'fewshot_adaptation.csv');probe=pd.read_csv(o/'subject_id_probes.csv')
 for task in ('valence','arousal'):
  q=sub.query('task==@task');fig,ax=plt.subplots(figsize=(9,4));ax.bar(q.subject_id,q.mean_minus_population);ax.axhline(0,color='k',lw=.8);ax.set(title=f'{task}: subject mean minus population mean',xlabel='subject',ylabel='rating difference');fig.tight_layout();fig.savefig(o/f'subject_bias_{task}.png',dpi=160);plt.close(fig)
  q=clip.query('task==@task');fig,ax=plt.subplots(figsize=(9,4));ax.bar(q.clip_id,q.sd);ax.set(title=f'{task}: clip disagreement',xlabel='clip',ylabel='between-subject rating SD');fig.tight_layout();fig.savefig(o/f'clip_disagreement_{task}.png',dpi=160);plt.close(fig)
  q=ad.query('task==@task and modality=="fusion"').groupby(['K','model']).mae.agg(['mean','sem']).reset_index();fig,ax=plt.subplots(figsize=(7,4));
  for name,g in q.groupby('model'): ax.errorbar(g.K,g['mean'],yerr=g['sem'],marker='o',label=name)
  ax.set(title=f'{task}: fusion adaptation',xlabel='support trials K',ylabel='query MAE');ax.legend(ncol=3);fig.tight_layout();fig.savefig(o/f'adaptation_{task}.png',dpi=160);plt.close(fig)
 # compact report: all values are computed from the saved tables.
 v=var.pivot(index='task',columns='model',values='in_sample_r2'); z=loso.groupby(['task','modality','model']).mae.mean();k=ad.groupby(['task','modality','K','model']).mae.mean();
 lines=['# Subject-dependent affect semantics','','## Protocol','', 'Paired-valid ASCERTAIN trials were analyzed with fold-local preprocessing. Population models used other subjects only. Few-shot P1/P2/P3/P4 use K target-subject support ratings and are evaluated on the same remaining target trials; they are not zero-shot results. P3 is a strongly regularized residual ridge head on a training-only PCA projection; P4 shrinks P3 toward P1.','', '## Phase 1: rating variation','', '| Task | Clip fixed-effect R² | Subject fixed-effect R² | Combined fixed-effect R² |','|---|---:|---:|---:|']
 for task in ('valence','arousal'): lines.append(f'| {task} | {v.loc[task,"clip"]:.3f} | {v.loc[task,"subject"]:.3f} | {v.loc[task,"clip_subject"]:.3f} |')
 lines+=['','These are descriptive fixed-effect decompositions, not identifiable subject×clip interaction estimates. They show stable subject mean/rating-scale differences after accounting only descriptively for clip effects, but cannot establish why subjects differ.','', '## Phase 2/4: zero-shot controls','', '| Task | Modality | Population feature MAE | Clip-mean MAE | Global-mean MAE |','|---|---|---:|---:|---:|']
 for task in ('valence','arousal'):
  for mod in ('eeg','face','fusion'):
   lines.append(f'| {task} | {mod} | {z[task,mod,"P0_feature"]:.3f} | {z[task,mod,"clip_mean"]:.3f} | {z[task,mod,"global_mean"]:.3f} |')
 lines+=['','## Phase 3: matched fusion adaptation','', '| Task | K | P0 MAE | P1 bias MAE | P2 scale MAE | P3 mapping MAE | P4 shrunk MAE |','|---|---:|---:|---:|---:|---:|---:|']
 for task in ('valence','arousal'):
  for kk in sorted(ad.K.unique()):
   q=k.loc[task,'fusion',kk];get=lambda x: q[x] if x in q.index else float('nan');lines.append(f'| {task} | {kk} | {get("P0"):.3f} | {get("P1"):.3f} | {get("P2"):.3f} | {get("P3"):.3f} | {get("P4"):.3f} |')
 lines+=['','## Identity diagnostic','','| Modality | Subjects | Top-1 ID accuracy |','|---|---:|---:|']
 for _,row in probe.iterrows(): lines.append(f'| {row.modality} | {int(row.n_subjects)} | {row.top1:.3f} |')
 lines+=['','## Conclusion','', 'Clip effects dominate zero-shot prediction: clip means outperform every EEG, face, and fusion population feature model in this protocol. Although subject fixed effects are descriptively nonzero and identity is decodable (especially from facial features), the regularized P3/P4 mapping heads do not consistently outperform P1/P2 or even P0 on independent target-subject queries. Thus H1 is supported only as stable rating variation; H2-H4 are not supported by these static features and matched few-shot tests. Phase 5 is not justified. These findings do not establish that affect is inherently subject-dependent or that identity is nuisance; they show that this dataset/protocol supplies no reliable incremental predictive evidence for a complex subject-specific feature mapping beyond stimulus controls.']
 (r/'subject_affect_semantics_report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
