# Reviewer follow-up diagnostic specification

Specified October 26, 2026, before running these additional checks. These are
exploratory reviewer-requested diagnostics, not additions to the primary family.
No thresholds or sample definitions will be chosen from their p-values.

For each frozen node design and pipeline, use the existing model participant set
and compare saved raw and ComBat features; do not refit ComBat or change features.
Export site-by-diagnosis H counts, means, and SDs. Fit the original adjusted H model
to examine leverage, residuals, and Cook's distance; retain participants regardless
of diagnostic flags. Report descriptive partial R² for site conditional on diagnosis,
age, IQ, and mean FD, and within-site median-centered Levene statistics for the
biological-covariate residuals. Post-ComBat comparisons are descriptive because
harmonization was estimated on this same sample; they are not independent proof
of removal of site effects.

Fit an additional raw adjusted H model restricted to sites containing both groups,
using the existing minimum site-size rule. Report coefficients, HC3 intervals,
explicit exclusions and Holm correction over the two pipelines within each design.
A restricted ComBat refit is outside this check. The primary model is unchanged.
