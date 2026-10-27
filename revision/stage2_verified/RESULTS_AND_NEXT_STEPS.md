# Verified Stage 2 results and revision actions

Verified October 26, 2026 against `results/abide_stage2_repaired/`.
The repair is computationally complete; this does not mean the paper is ready for submission.
The submitted manuscript and original Stage 2 outputs remain preserved.

## Original-paper continuity audit — October 26, 2026

The main draft now has eight figures and dedicated original-matching Results
sections; S1 has six figures. All 364 original exported H correlations and
per-group FDR values were reproduced. The restored revised raw H-versus-metrics
analysis has 59 Pearson associations under the original per-group correction
convention, separately from the nonsignificant diagnosis tests. See the
[full continuity map and significant-comparison lists](continuity/CONTINUITY_AND_SIGNIFICANT_COMPARISONS.md).
These latest counts supersede the preceding five-main/eight-supplement figure
layout. Original inputs and findings remain preserved, with changed cohorts and
feature definitions explicitly distinguished.

## Classifier extension — October 26, 2026

Gaussian naive Bayes and KNN now have complete 140-person OOF predictions under
both pipelines, alongside unchanged logistic/SVM records. Their balanced accuracy
is .498/.515 and .518/.542 (GSR/no-GSR), respectively; all four conditional
intervals include .5. Four of the five original algorithm types are complete.
Only XGBoost remains pending. `bash Codes/finish_original_classifiers.sh` from a
normal terminal installs the CPU dependency and resumes missing folds without
rerunning imaging or existing classifiers. See [the completion report](CLASSIFIER_COMPLETION.md).
The complete group/correlation significance lists are unchanged.

## Feature selection and five-model completion — October 27, 2026

All five original classifiers are complete. Baseline ANOVA feature selection
was already nested: 22/44/88 features were selected in 11/17/22 outer fits, and
all 50 reconstructed predictions match. A separate exploratory search over
training-only redundancy pruning, ANOVA, mutual information and L1 ranking has
complete predictions for all ten model/pipeline pairs. Four point estimates
improve and six decline; every paired change interval includes zero. See
[the complete selection report](ml_feature_selection/FEATURE_SELECTION_AND_RESULTS.md).
ROCs, confusion heatmaps, performance and selected-feature frequency heatmaps
are saved as PDF/PNG. Main paper/S1 now have eight/ten figures. No new data or
Kaggle run is needed; earlier XGBoost-pending entries below are historical.

## What completed

- All 626 saved feature rows were reused, with their SHA-256 checked. No imaging,
  time-series, graph, or fALFF calculation was repeated in this repair.
- All six design/pipeline ComBat sensitivities succeeded and changed feature values.
- Each of four classifier/pipeline combinations has exactly 140 unique out-of-fold
  predictions. Each site appears in only one outer test fold per combination.
- Completed clinical, efficiency, and null outputs were preserved by the repair.

## Primary result

The primary analysis uses a fixed **106-ROI coverage-limited common-node set**.
It is not a complete CC200 whole-brain analysis. There are 140 eligible participants;
site-size exclusions leave 136 (67 ASD, 69 controls) for the adjusted group models.
These models include diagnosis, site, age, FIQ, and mean FD, with HC3 uncertainty.

| Pipeline | Adjusted ASD coefficient | 95% CI | p | Holm p |
|---|---:|---|---:|---:|
| GSR | 1.432 | −17.173 to 20.038 | .879 | 1.000 |
| No-GSR | 2.056 | −12.387 to 16.499 | .779 | 1.000 |

No group feature survives its specified multiplicity correction in the raw,
ComBat, or energy-trimming models across the three node designs. This is evidence
of no detected difference under these analyses, not proof of equivalence. The
exploratory equivalence tests are inconclusive. Larger node-set sensitivities
also change the participant sample, so differences between designs cannot be
attributed solely to node coverage.

## Prediction and supporting analyses

| Pipeline | Classifier | Balanced accuracy | Conditional 95% CI |
|---|---|---:|---|
| GSR | Logistic regression | .534 | .469–.613 |
| GSR | RBF SVM | .558 | .464–.653 |
| No-GSR | Logistic regression | .485 | .428–.559 |
| No-GSR | RBF SVM | .485 | .385–.591 |

Every interval includes .5. These intervals bootstrap sites conditional on fixed
OOF predictions; they do not include all model-training uncertainty. The node mask
was selected using cohort-wide QC, so prediction remains exploratory and conditional
on that mask. Replace the old incomplete-49-subject classification scores.

No corrected clinical correlation, PLSC latent-variable test, or group-by-efficiency
interaction is significant. Site-held-out PLSR has negative OOF R² for all five
scores. The earlier ASD-specific energy–efficiency trade-off is unsupported.
The restricted null preserves coarse network/hemisphere blocks, not full spatial
autocorrelation; null alignment is not evidence of diagnostic utility or dynamics.
Within-limbic features are unavailable in the primary mask because it includes
only one eligible limbic ROI; do not describe them as nonsignificant tests.

## Manuscript-ready primary-results paragraph

In the coverage-limited 106-region analysis, adjusted Hamiltonian values did not
show a detectable diagnosis effect in either preprocessing pipeline (GSR:
β=1.43, 95% CI −17.17 to 20.04; no-GSR: β=2.06, 95% CI −12.39 to 16.50;
both Holm-adjusted p=1.00; n=136). The ComBat and pooled energy-trimming
sensitivities likewise did not identify a corrected diagnosis effect. Exploratory
site-held-out classification produced balanced accuracies of 0.485–0.558, with
all conditional confidence intervals including 0.5. These findings do not
establish equivalence between groups or support diagnostic use of this measure.

## Control baseline and figure restoration — October 26, 2026

All 71 eligible male controls are now separately reported against their saved
null configurations: unrestricted mean z is −4.690/−5.159 and blocked mean z
is −2.313/−2.760 (GSR/no-GSR). See `control_nulls/`. This baseline describes
alignment, not an ASD contrast or a topology-rewired graph null. The main draft
now contains five figures, with eight additional figures in S1. Ethics wording
now explains ABIDE approvals/consent and the public deidentified secondary use,
without requiring a new approval number or asserting a formal exemption.

## Follow-up completed — October 26, 2026

Reviewer site/group summaries, influence diagnostics, and raw models restricted
to mixed-diagnosis sites are now in `diagnostics/`. The primary restriction gives
127 participants (58 ASD/69 controls), with coefficients 1.115 and 1.660 and
both Holm p=1.00. All 12 raw/ComBat diagnostic matrices have full rank.
A [replacement manuscript, supporting files, and response draft](../revised_paper/README.md)
are prepared and compile. The following tasks now describe the remaining
finalization work, rather than an unstarted manuscript rewrite.

## What happens next

1. **Rewrite the manuscript from the corrected analysis**, replacing the old
   cohort counts, ordinal network assignments, locally calculated spectral
   ratios, statistical results, predictive results, tables, and figures. Describe
   downloaded PCP time-series/fALFF derivatives separately from local feature
   construction. Report age 18–30, observed FIQ>80, male-only selection, missing
   handedness, participant flow, and the coverage-limited scope accurately.
2. **Resolve historical PCP fALFF provenance.** The derivative filenames and
   equal filtered/unfiltered-strategy fALFF files do not establish the historical
   processing branch. Document the historical implementation, filtering/input
   branch, band limits, ratio definition, and regional aggregation. A modern
   C-PAC implementation cannot establish what produced the historical release.
   If this cannot be verified, explicitly limit interpretation and determine
   whether independent recomputation is needed before submission.
3. **Complete reviewer-requested diagnostic checks** against the original
   checklist, including site-by-group raw/harmonized summaries and residual-site
   diagnostics; determine whether the both-diagnosis-site sensitivity is feasible.
   Successful ComBat execution alone does not establish removal of site effects.
4. **Prepare the point-by-point response and separate supporting information.**
   Explain the original-code discrepancies, the replacement analysis, missing
   tests, and limitations; avoid claiming that harmonization alone explains the
   difference from the submitted findings.
5. **Obtain author confirmations and submission materials:** institutional ethics
   wording for public deidentified data; approval of corrected scientific scope
   and conclusions; repository/archive links for code and allowed data manifests;
   final journal formatting, reference, cross-reference, and PDF checks.

No additional Kaggle run is needed to repair the reported execution failures.
Further analyses, if needed for outstanding reviewer points, should be specified
before running them and saved separately rather than overwriting this result set.

## Generated materials

`Hamiltonian_all_sensitivities.csv`, `primary_model_demographics.csv`,
`primary_model_site_counts.csv`, `classification_complete_OOF.csv`, and
`primary_results.pdf`/`.png` are generated directly from the completed outputs.
Rebuild with `MPLCONFIGDIR=/tmp/coev_matplotlib python3 Codes/summarize_stage2_revision.py`.
`verification.json` records the feature hash and completion checks.
