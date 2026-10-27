# Original-classifier completion

Classification now includes 5 original algorithm types and 10 complete
pipeline/model pairs: LogisticRegression, SVM_RBF, GaussianNB, KNN, XGBoost. All original algorithm types are complete.

The original spectral descriptor remains separately reported as exploratory,
outside the frozen 88-feature schema. Matching algorithms does not equate different
cohorts/features/validation. See results/abide_stage2_classification_extension/.

| Pipeline | Model | Balanced accuracy | Conditional 95% CI |
|---|---|---:|---|
| GSR | GaussianNB | 0.498 | 0.428–0.583 |
| GSR | KNN | 0.518 | 0.455–0.581 |
| GSR | LogisticRegression | 0.534 | 0.469–0.613 |
| GSR | SVM_RBF | 0.558 | 0.464–0.653 |
| GSR | XGBoost | 0.457 | 0.390–0.531 |
| noGSR | GaussianNB | 0.515 | 0.401–0.633 |
| noGSR | KNN | 0.542 | 0.470–0.627 |
| noGSR | LogisticRegression | 0.485 | 0.428–0.559 |
| noGSR | SVM_RBF | 0.485 | 0.385–0.591 |
| noGSR | XGBoost | 0.507 | 0.439–0.615 |

All intervals are conditional on fixed OOF predictions; none is a formal model permutation test. The 59 reported raw H correlations and group-model tables are unchanged.
