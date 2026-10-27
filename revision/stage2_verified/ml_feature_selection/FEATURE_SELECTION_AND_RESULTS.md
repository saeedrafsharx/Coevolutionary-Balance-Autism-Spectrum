# Feature selection and exploratory ML follow-up — October 27, 2026

## Did the original revised classifiers use feature selection?

Yes. All five classifier types used nested training-fold imputation, variance
filtering, scaling and ANOVA ranking. Inner CV chose 25%, 50% or 100% of features.
Reconstruction of all 50 outer fits reproduces their saved predictions. The
selected counts were 22 features in 11 fits, 44 in 17 fits, and 88 in 22 fits.
Choosing 100% is the no-reduction option within that trained selection search.
The baseline_feature_selection.csv audit contains every selected feature name.

## What was added?

A separate exploratory run removes near-exact redundant columns within training
folds and jointly compares all remaining features, ANOVA, mutual information and
L1-coefficient-ranked top 5/10/20 selection. All five algorithms use the original
five outer/three inner site-disjoint folds. No diagnosis-based trimming or ComBat
is used. All inner CV candidates and selected features are exported. L1 ties use
training ANOVA rank; zero-coefficient selections are counted, not hidden.

This follow-up was designed after seeing the original outer results and therefore
is not an independent validation of improvement. Its narrower XGBoost grid and
changed representation mean any gain is not attributable to selector method alone.
No classifier is selected from outer accuracy and no positive result is guaranteed.

## All results

| Pipeline | Model | Baseline BA | Selection BA | Change | Paired conditional 95% CI for change |
|---|---|---:|---:|---:|---|
| GSR | GaussianNB | 0.498 | 0.491 | -0.008 | -0.092, 0.074 |
| GSR | KNN | 0.518 | 0.442 | -0.076 | -0.193, 0.024 |
| GSR | LogisticRegression | 0.534 | 0.521 | -0.013 | -0.107, 0.036 |
| GSR | SVM_RBF | 0.558 | 0.534 | -0.023 | -0.097, 0.068 |
| GSR | XGBoost | 0.457 | 0.492 | +0.036 | -0.031, 0.103 |
| noGSR | GaussianNB | 0.515 | 0.493 | -0.022 | -0.086, 0.038 |
| noGSR | KNN | 0.542 | 0.557 | +0.014 | -0.098, 0.108 |
| noGSR | LogisticRegression | 0.485 | 0.544 | +0.059 | -0.015, 0.109 |
| noGSR | SVM_RBF | 0.485 | 0.536 | +0.051 | -0.038, 0.148 |
| noGSR | XGBoost | 0.507 | 0.500 | -0.006 | -0.149, 0.078 |

## Selection-method choices

- all: 7 of 50 outer-training fits.
- anova: 16 of 50 outer-training fits.
- l1: 10 of 50 outer-training fits.
- mutual_info: 17 of 50 outer-training fits.

## Figures and files

All five-model ROC curves, normalized confusion heatmaps, baseline-versus-selection
BA/AUC heatmaps, and selected-feature frequency heatmaps are in
results/abide_ml_feature_selection_review/figures/. Both PDF and PNG are supplied.
Heatmaps show all algorithms; frequency panels show the top 25 descriptive
frequencies with all 88 features provided as CSV. Frequencies are not biomarker
significance, model permutation importance or independent replication.
The original study group tests, nulls, and 59 H associations are unchanged.
