#!/usr/bin/env python3
"""Pooled LOSO summaries, paired bootstrap intervals and Holm-adjusted tests."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def holm(p):
    out=np.empty(len(p)); order=np.argsort(p); running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(p)-rank)*p[index]); out[index]=min(1.,running)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); a=ap.parse_args(); out=a.root/'results/affect_contrastive_v2'
    folds=pd.read_csv(out/'fold_metrics.csv'); weighted=pd.read_csv(out/'weighted_ce/fold_metrics.csv'); baseline=pd.read_csv(a.root/'results/metrics/baseline_results.csv')
    # Pooled OOF statistics use every valid held-out trial once within each seed.
    pred=pd.read_csv(out/'oof_predictions.csv'); pooled=[]
    for (seed,task,model),q in pred.groupby(['seed','task','model']):
        y=q.label.to_numpy(); p=q.probability.to_numpy(); hat=p>=.5
        pooled.append({'seed':seed,'task':task,'model':model,'n_trials':len(q),'pooled_uar':balanced_accuracy_score(y,hat),'pooled_f1':f1_score(y,hat,zero_division=0),'pooled_auc':roc_auc_score(y,p)})
    pd.DataFrame(pooled).to_csv(out/'pooled_oof_metrics.csv',index=False)
    summary=folds.groupby(['task','model']).agg(n_fold_seed=('fixed_uar','size'),n_subjects=('fold_subject','nunique'),mean_fold_uar=('fixed_uar','mean'),sd_fold_uar=('fixed_uar','std'),mean_accuracy=('fixed_accuracy','mean'),mean_f1=('fixed_f1','mean'),mean_auc=('fixed_auc','mean'),single_class_folds=('fixed_single_class_test','sum'),mean_best_epoch=('best_epoch','mean')).reset_index()
    ws=weighted.groupby(['task','model']).agg(mean_fold_uar=('fixed_uar','mean'),mean_accuracy=('fixed_accuracy','mean'),mean_f1=('fixed_f1','mean'),predicted_positive_fraction=('fixed_predicted_positive_fraction','mean')).reset_index(); ws['model']='CE_weighted_exploratory'; ws.to_csv(out/'weighted_ce_summary.csv',index=False)
    summary.to_csv(out/'loso_summary.csv',index=False)
    comparisons=[]; rng=np.random.default_rng(77)
    for task in ['valence','arousal']:
      for seed in sorted(folds.seed.unique()):
        q=folds[(folds.task==task)&(folds.seed==seed)].pivot(index='fold_subject',columns='model',values='fixed_uar')
        for left,right in [('M1','M0'),('M2','M0'),('M2','M1'),('CE','M0')]:
          d=(q[left]-q[right]).dropna().to_numpy(); boot=[rng.choice(d,len(d),replace=True).mean() for _ in range(3000)]
          p=wilcoxon(d,method='auto').pvalue if np.any(d) else 1.
          comparisons.append({'task':task,'seed':seed,'comparison':f'{left}-{right}','n_subjects':len(d),'mean_delta_uar':d.mean(),'ci95_low':np.quantile(boot,.025),'ci95_high':np.quantile(boot,.975),'wilcoxon_p':p})
    comp=pd.DataFrame(comparisons); comp['holm_p']=holm(comp.wilcoxon_p.to_numpy()); comp.to_csv(out/'paired_model_comparisons.csv',index=False)
    early=baseline[(baseline.model=='logreg')&(baseline.modality=='early_fusion')][['task','fold_subject','uar']].rename(columns={'uar':'early_fusion_uar'})
    neural=folds[folds.seed==3407].pivot_table(index=['task','fold_subject'],columns='model',values='fixed_uar').reset_index().merge(early,on=['task','fold_subject'])
    early_rows=[]
    for task,q in neural.groupby('task'):
      for model in ['M0','M1','M2']:
        d=(q[model]-q.early_fusion_uar).dropna().to_numpy(); boot=[rng.choice(d,len(d),replace=True).mean() for _ in range(3000)]
        early_rows.append({'task':task,'model_vs_early_fusion':model,'n_subjects':len(d),'mean_delta_uar':d.mean(),'ci95_low':np.quantile(boot,.025),'ci95_high':np.quantile(boot,.975),'wilcoxon_p':wilcoxon(d,method='auto').pvalue if np.any(d) else 1.})
    early=pd.DataFrame(early_rows); early['holm_p']=holm(early.wilcoxon_p.to_numpy()); early.to_csv(out/'early_fusion_comparisons.csv',index=False)
    print(summary.to_string(index=False)); print('\n',comp.to_string(index=False)); print('\n',early.to_string(index=False))


if __name__=='__main__': main()
