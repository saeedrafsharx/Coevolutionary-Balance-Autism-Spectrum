# Stage 2 review and partial local repair — October 26, 2026

The original uploaded `results/abide_stage2_review/` is preserved unchanged.
This directory contains recovered raw group models and copied outputs from the
original completed clinical/efficiency/null jobs. It does not yet contain a complete
ComBat sensitivity or valid full-cohort classifier rerun. Consult `repair_summary.json`.

## Run diagnosis

All **626 planned feature rows** were generated. Group analysis stopped when the
optional sensitivity-power helper evaluated a large noncentral-t survival tail
that returned NaN under the Kaggle SciPy version. Because final table export was
after that call, the group tables were missing even though feature extraction
was complete. This was a code defect, not evidence about the group contrast.

The helper now uses CDF complements and an adaptive small bracket. Optional
diagnostic failures cannot abort core model tables, and in-progress results are
checkpointed. Feature computation, clinical analyses, null simulations, and graph
efficiency do not need to be repeated.

Classification was labeled completed but generated only **49/140** out-of-fold
predictions per model/pipeline because single-class folds were skipped. Its old
aggregate scores and confidence intervals must not be reported for the full cohort.
The repair uses deterministic size-balanced site-disjoint folds; every outer and
inner split was checked against the actual 140-participant metadata and has both
classes. This change addresses split validity, not optimization of test accuracy.
Incomplete OOF outputs can no longer generate a full-cohort aggregate.

## Primary group models recovered locally

H is the **106-ROI coverage-limited common-node** Hamiltonian. Models adjust for
site, age, FIQ, and mean FD and use HC3 uncertainty. The sample is 136 participants
(67 ASD, 69 controls) after the frozen small-site rule.

| Pipeline | ASD coefficient | HC3 95% CI | Raw p | Holm p |
|---|---:|---|---:|---:|
| GSR | 1.432 | −17.173 to 20.038 | .879 | 1.000 |
| No-GSR | 2.056 | −12.387 to 16.499 | .779 | 1.000 |

No adjusted group difference was detected. Raw adjusted secondary Whole/Intra/Inter
families also contain no BH-significant results in any of the three frozen node
designs. H remains nonsignificant in the broader-181/full-200 and energy-trimming
sensitivities. These statements currently concern the **raw adjusted models**;
the complete corrected ComBat sensitivity has not yet been recovered.

The exploratory equivalence tests are inconclusive (GSR p=.193; no-GSR p=.223)
under the conventional ±0.2 pooled-SD bounds. Do not claim group equivalence or
absence of all effects. Approximate two-sample sensitivity power gives minimum
detectable d≈.484 at alpha=.05; this is not a covariate/site-adjusted power estimate.

## Numerical verification

Statsmodels is absent in this restricted local environment. The local repair uses
the same OLS/HC3 matrix estimator, explicitly marked in `group_models.csv`.
The Kaggle backend had already validated HC3 against that matrix formula.
Additionally, local estimates were checked against both saved actual-data
statsmodels group-by-efficiency models: coefficients, SEs, confidence limits, and
p-values agree at rtol/atol 1e-7. See `audit/actual_data_HC3_crosscheck.csv`.
This is an implementation cross-check, not independent replication of the data.

## Completed supporting results

- Partial H–efficiency correlations controlling mean FC, FD, age, IQ, and site
  are not significant after correction. ASD global-efficiency partial r is .017
  with GSR and .077 without GSR. Group-by-efficiency interactions are also not
  significant (Holm p=.635 and .940). The old ASD-specific trade-off claim is unsupported.
- No clinical feature correlation survives FDR. Adjusted H correlations with ADOS
  Total are near zero (GSR r=.008; no-GSR r=.061). Per-score ASD sample sizes are
  28–50, not the old paper's counts.
- Site-held-out PLSR has negative OOF R² and worse MSE than fold-training-mean
  baselines for all five scores in both pipelines.
- No exploratory PLSC latent variable survives BH correction. A nominal no-GSR
  five-score LV3 p=.00699 becomes q=.0979; it should not be presented as confirmed.
- Null scores were computed for 140 participants in each pipeline under both nulls.
  Mean z is approximately −4.70/−5.04 for unrestricted node permutations and
  −2.32/−2.68 for network/hemisphere blocks (GSR/no-GSR). These describe alignment
  relative to specified randomizations, not ASD discrimination or proof of dynamics.
  The block null does not preserve full spatial autocorrelation.

## Finish the two outstanding computations

From the repository root, run in a **normal terminal with Internet access**:

```sh
bash Codes/finish_stage2.sh
```

The script creates a project-local `.venv_stage2`, installs missing statistics
packages there, and reruns only group models/ComBat sensitivities and classification
from the saved feature CSV. It preserves the completed clinical/null/efficiency
outputs and writes to this repaired directory. It does not redo image processing,
feature extraction, null simulations, or graph calculations. The sandbox cannot
download these dependencies, so only the raw core was recovered here.

Check `repair_summary.json` for `full_repair_complete: true` afterward. Statistical
interpretation, upstream fALFF provenance, revised scope/handedness disclosure, and
the manuscript/response package still require review before submission.
