# Subject-dependent affect semantics

Primary endpoints are continuous-rating MAE/RMSE on matched target-subject
query trials.  P0 is a population ridge model; P1 adapts only an output
intercept; P2 calibrates an output affine map when support permits; P3 fits a
strongly regularized target-subject residual head on a training-only PCA
projection; P4 shrinks P3 toward P1.  The practical primary effect is P3-P1.

All paired-valid trials and the immutable manifest are retained.  Preprocessing,
PCA, population models, and clip means use other subjects only.  K-shot target
support labels are used only by P1-P4, and every method shares query trials.
K={0,1,2,4,8}; support splits use seeds 3407, 2027, 911.

Stop rule: a shared/private model is not justified unless P3/P4 reproducibly
improve independent query prediction beyond P1/P2 and fusion improves beyond
both unimodal models under the same protocol.
