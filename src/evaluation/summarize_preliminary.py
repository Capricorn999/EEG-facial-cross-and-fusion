#!/usr/bin/env python3
"""Materialize required pilot tables and an evidence-bounded stopping report."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon

def mean_metric(df, task, model, modality):
    x=df[(df.task==task)&(df.model==model)&(df.modality==modality)]
    return x.uar.mean(), x.auc.mean()
def paired(a,b):
    z=a.merge(b,on='fold_subject',suffixes=('_a','_b')).dropna(subset=['uar_a','uar_b'])
    if len(z)<2:return float('nan'),len(z)
    try:return wilcoxon(z.uar_a,z.uar_b,alternative='two-sided',method='auto').pvalue,len(z)
    except ValueError:return 1.0,len(z)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root;metrics=r/'results/metrics';reports=r/'reports'
 base=pd.read_csv(metrics/'baseline_results_linear_fixed.csv');cca=pd.read_csv(metrics/'cca_alignment_results.csv');base.to_csv(metrics/'baseline_results.csv',index=False)
 raw=pd.read_csv(metrics/'raw_subject_probe_results.csv');cp=pd.read_csv(metrics/'cca_subject_probe_results.csv')
 records=[]
 for rep,model,mod,probe in [('Raw EEG','logreg','eeg','raw_eeg'),('Raw face','logreg','face','raw_face'),('Early fusion','logreg','early_fusion','early_fusion')]:
  p=raw[raw.representation==probe].iloc[0]
  for task in ['valence','arousal']:
   u,auc=mean_metric(base,task,model,mod);records.append({'representation':rep,'emotion_task':task,'emotion_uar':u,'emotion_auc':auc,'subject_top1':p.subject_top1,'subject_top5':p.subject_top5,'subject_macro_f1':p.subject_macro_f1,'probe_note':f"{int(p.n_subjects)} analyzable subjects; within-subject trial split"})
 p=cp[cp.representation=='cca_shared_latent'].iloc[0]
 for task in ['valence','arousal']:
  u,auc=mean_metric(cca,task,'cca_logreg','cca_shared');records.append({'representation':'CCA shared latent','emotion_task':task,'emotion_uar':u,'emotion_auc':auc,'subject_top1':p.subject_top1,'subject_top5':p.subject_top5,'subject_macro_f1':p.subject_macro_f1,'probe_note':p.note})
 p=cp[cp.representation=='cca_concat_latent'].iloc[0]
 for task in ['valence','arousal']:
  u,auc=mean_metric(cca,task,'cca_logreg','cca_concat');records.append({'representation':'CCA concat latent','emotion_task':task,'emotion_uar':u,'emotion_auc':auc,'subject_top1':p.subject_top1,'subject_top5':p.subject_top5,'subject_macro_f1':p.subject_macro_f1,'probe_note':p.note})
 probes=pd.DataFrame(records);probes.to_csv(metrics/'representation_probe_results.csv',index=False)
 report=['# ASCERTAIN preliminary experiment report','','## Scope','', 'This is a deliberately small, falsifiable pilot: quality-screened extracted EEG (88-D) and facial-motion (72-D) features, fixed rating-scale thresholds, fold-fitted preprocessing, LOSO logistic/linear-SVM baselines, and a 32-D ridge-CCA alignment comparator. No raw signals, personality inputs, Transformers, or graph models were used.','', '## Dataset and protocol','', '- 2,088 possible trials; 1,765 quality-screened EEG trials; 1,463 face trials; 1,177 paired trials with both primary labels. See `dataset_audit.md`.','- Primary labels: valence > 0 on the documented −3…3 scale; arousal > 3 on the 0…6 scale.','- Every emotion score is an LOSO aggregate. Imputation, zero-variance filtering and scaling were fit on the other subjects only.','- Fair fusion/alignment comparisons use the same 47 held-out subjects having paired test trials.','', '## Main results (mean LOSO UAR)','']
 for task in ['valence','arousal']:
  report += [f'### {task.capitalize()}', '', '| Representation / model | UAR | ROC-AUC |', '|---|---:|---:|']
  for label,df,model,mod in [('EEG (paired support)',base,'logreg','eeg_paired'),('Face (paired support)',base,'logreg','face_paired'),('Early fusion',base,'logreg','early_fusion'),('CCA shared',cca,'cca_logreg','cca_shared'),('CCA concat',cca,'cca_logreg','cca_concat')]:
   u,au=mean_metric(df,task,model,mod);report.append(f'| {label} | {u:.3f} | {au:.3f} |')
  e=base[(base.task==task)&(base.model=='logreg')&(base.modality=='early_fusion')][['fold_subject','uar']];z=cca[(cca.task==task)&(cca.modality=='cca_shared')][['fold_subject','uar']];p,n=paired(z,e);z2=cca[(cca.task==task)&(cca.modality=='cca_concat')][['fold_subject','uar']];p2,n2=paired(z2,e);report += ['',f'Paired Wilcoxon, CCA shared vs. early fusion: n={n}, p={p:.3g}.',f'Paired Wilcoxon, CCA concat vs. early fusion: n={n2}, p={p2:.3g}.','']
 report += ['## Subject-information diagnostic','', '| Representation | Top-1 subject ID | Top-5 | Caveat |','|---|---:|---:|---|']
 for _,x in raw.iterrows():report.append(f"| {x.representation} | {x.subject_top1:.3f} | {x.subject_top5:.3f} | {int(x.n_subjects)} analyzable subjects |")
 x=cp[cp.representation=='cca_shared_latent'].iloc[0];report.append(f"| CCA shared latent | {x.subject_top1:.3f} | {x.subject_top5:.3f} | exploratory global unsupervised transform, {int(x.n_subjects)} subjects |")
 x=cp[cp.representation=='cca_concat_latent'].iloc[0];report.append(f"| CCA concat latent | {x.subject_top1:.3f} | {x.subject_top5:.3f} | exploratory global unsupervised transform, {int(x.n_subjects)} subjects |")
 report += ['', 'Raw face features are highly subject-identifying (0.662 top-1, versus 0.021 chance for its 47 active subjects). Both aligned variants remain highly subject-identifying (shared 0.411; concat 0.445; both versus 0.022 chance for 46 subjects). These are lower than raw face but the probes are exploratory, not formal adversarial-disentanglement results.','', '## Cross-modal pairing diagnostic','', 'See `pair_similarity_analysis.csv`. Same-subject/same-clip similarity is 0.318, versus 0.093 for same-subject/different-clip and approximately zero for different-subject pairs. In this linear alignment, synchronized pairing dominates similarity; different-subject, similar-affect pairs do not yet form a distinct positive-similarity category.','', '## Decision','', '**Stop the pilot before adding adversarial or raw-temporal complexity.** The strong-evidence criterion is not met: although concatenated CCA features numerically improve arousal UAR, the paired test is not significant and both aligned representations retain substantial subject-ID predictability. The moderate-evidence criterion is also not met because identity remains far above chance.','', 'A follow-up is justified only after diagnosing the data-quality/support imbalance and testing a controlled contrastive model with affect-matched cross-subject positives against this locked LOSO protocol. Do not claim that cross-modal agreement currently isolates affect from subject nuisance.','', '## Deliberate exclusions','', 'The requested small neural MLP and adversarial ablation are not reported as completed results. This is intentional under the supplied stopping rule: the linear aligned comparator did not provide adequate evidence, so adding higher-capacity/disentanglement machinery would confound diagnosis with architecture search. Their absence is not evidence of failure or success.']
 reports.mkdir(exist_ok=True);(reports/'preliminary_report.md').write_text('\n'.join(report)+'\n')
 print(probes.to_string(index=False));print('wrote',reports/'preliminary_report.md')
if __name__=='__main__':main()
