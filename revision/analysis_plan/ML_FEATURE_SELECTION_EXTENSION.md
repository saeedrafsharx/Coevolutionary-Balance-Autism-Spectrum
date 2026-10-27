# Exploratory feature-selection extension — October 27, 2026

The five-classifier outputs and their outer-fold results have already been viewed.
This follow-up is explicitly exploratory, not independent confirmatory evaluation.
It does not overwrite or replace the original five-model site-held-out analysis.
No desired accuracy, p-value, or winner is a stopping criterion.

Inputs remain the fixed 88-feature, 106-node, 140-person raw primary design.
No ComBat, diagnosis-based trimming, new subject inclusion, or imaging computation.
Use the exact five outer and three inner size-balanced site-disjoint folds from
the completed analysis. Test sites are never used to choose features or parameters.

Every training pipeline fits median imputation, zero-variance removal, near-exact
redundancy pruning (absolute training correlation >=1-1e-10; retain first feature
in fixed schema order), standardization, feature selection and the classifier.
Redundancy matters because fixed-block EnergyBlock and AvgFalffW are proportional,
and several state-configuration proportions are complementary. Dropping duplicates
is a representation change rather than evidence that those features were false.

The inner search compares these ten selector configurations: retain all remaining
features, ANOVA top 5/10/20, mutual information top 5/10/20 (3 neighbors; seed 101),
and L1-logistic coefficient-ranked top 5/10/20 (C=1, balanced class weights,
liblinear, seed 101). L1 coefficient ties, including zero coefficients, are broken
by training ANOVA rank and then fixed schema order; count zero-coefficient selections
explicitly. Actual K is capped at the available nonredundant feature count.

All five original algorithm types are evaluated. Fixed classifier search grids:
- Logistic regression: balanced weights, C=.01/.1/1/10.
- RBF SVM: balanced weights, C=.1/1/10 and gamma=scale.
- Gaussian naive Bayes: default training priors and variance smoothing.
- KNN: neighbors=3/5/7/9/11, uniform/distance weights.
- XGBoost: CPU histogram trees, seed 42, one thread, depth=2/3,
  estimators=100/200, learning rate=.05/.1, logloss evaluation.

The compact XGBoost grid is fixed before executing this follow-up. Feature-method,
K, and classifier parameters are selected jointly by inner balanced accuracy for
each model/outer fold. Cache training transforms only, keyed by exact training X/y.
Export every inner candidate score, selected feature names and ranks, outer-fold
parameters, complete OOF scores, balanced accuracy/AUC, fixed-prediction site-
bootstrap intervals (1000 replicates, seed 303) and all differences from baseline.
No best outer-scoring model is promoted to a validated final classifier. Reusing
known outer folds, the cohort-wide QC mask and unmodeled site variability limit
interpretation; external data would be needed to establish a genuine gain.

Separately reconstruct the original fitted ANOVA selectors from saved fold
parameters to document selected features, checking all reproduced OOF predictions
against their saved records. This reconstruction does not retune on outer tests.
ROC/confusion plots remain available for both baseline and exploratory follow-up;
feature selection-frequency heatmaps are descriptive stability summaries, not
biomarker significance or causal feature importance.
