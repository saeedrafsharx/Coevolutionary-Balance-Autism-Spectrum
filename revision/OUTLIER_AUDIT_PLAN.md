# Original exclusion and outlier audit

## Evidence still required

The repository currently contains acquisition and metadata-repair code, but not
the original analysis code implementing exclusions, outlier removal, harmonization,
or network statistics. Audit the original analysis notebook/scripts before choosing
the revision's analysis sample. Preserve their outputs and participant identifiers.

The manuscript describes group-specific whole-brain-energy Tukey fences, removing
values below Q1 − 1.5 IQR or above Q3 + 1.5 IQR, and a common retained participant
set for subsequent analyses. Other passages mention excluding outliers in null
distributions. The code must establish the actual behavior and processing order.

## Checks to reproduce from code

- Identify every exclusion: initial selection, demographic eligibility, missing IQ,
  motion/QC, missing derivatives, invalid ROIs, small sites, energy trimming, and
  clinical-score missingness. Distinguish imaging exclusions from score-specific
  complete cases.
- Establish whether trimming uses raw or harmonized energy, empirical energy or
  null distributions, and separate diagnosis groups, sites, or a pooled sample.
- Record quantile implementation, fence multiplier, boundary inclusivity, missing
  and nonfinite handling, and whether filtering runs once or iteratively.
- Check whether the mask is recomputed by feature, analysis, or preprocessing
  strategy, or reused consistently by subject ID. Verify joins do not align rows
  by position or silently change the cohort.
- Record precisely which participants are removed at each step, with values,
  thresholds, reasons, and group/site counts. Reconcile 182 initial download IDs,
  exported feature cohorts, and the manuscript's final 78/86 and 78/87 samples.
- Check prediction code for diagnosis-dependent trimming or filtering informed by
  held-out data. Separate descriptive-analysis decisions from validation design.

## Required audit outputs

Create a participant-level exclusion ledger, stage-by-stage flow table, per-group
fences, and original-versus-reproduced retained-ID comparison for each pipeline.
Retain the pre-trimming data. Reproduce the original implementation faithfully
before proposing changes; reproducibility does not itself establish appropriateness.

Reviewer 1 specifically requests untrimmed sensitivity results because trimming on
energy within diagnosis groups conditions selection on the outcome. Once the
original code is understood, define the revised primary cohort and compare trimmed
and untrimmed results transparently. Acquisition eligibility flags alone do not
establish the final analysis cohort or justify rewriting its demographics.

## Kaggle inputs

The two dataset roots provided by the author are recorded in
`Codes/kaggle_data_paths.json`. The patch is a separate dataset, and both inputs
must remain available to the validation and reanalysis notebooks.
