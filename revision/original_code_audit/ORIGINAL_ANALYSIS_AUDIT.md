# Audit of the four original analysis notebooks

**Stage 1 follow-up:** the legacy masks and numeric fences are now reconstructed
for all 171 participants with finite/valid time series. All retention decisions
match the historical exports. The old 15/14 exclusions comprise 11 undefined-H
cases from constant ROI time series plus only 4/3 finite-energy IQR outliers.
The earlier statements below describe the pre-reconstruction evidence. See
`results/abide_stage1_assessment/ASSESSMENT.md` for the resolved flow and remaining QC.

Reviewed October 24, 2026. Original notebooks and data were not modified. This audit
uses their code, saved execution output, and existing feature CSVs. It does not
claim to have regenerated imaging-derived features from the new Kaggle datasets.

## Main finding

The notebooks explain the exported cohorts and the harmonization failure. They also
show material differences from the manuscript in outlier removal, the nodal fALFF
measure, anatomical network assignment, and protected ComBat covariates. Updating
the prose alone would not establish the intended analysis: features and results
need to be rebuilt with a documented implementation.

## Notebook roles

| Notebook in `Codes/` | Role and evidence |
|---|---|
| `gsr-plsc-pls-analysis.ipynb` | Cell 0 calculates features, trims energy, attempts ComBat, and performs statistics/PLSR/ML. Later cells add PLSC and signed modularity. Saved output shows 182 extracted participants, then 167 after trimming and all-NaN ComBat followed by raw fallback. |
| `no-gsr-analysis.ipynb` | Its single cell implements the parallel feature/analysis pipeline. Saved output shows 182 extracted participants, then 168 after trimming and the same raw fallback. |
| `gsr-harmonization-fix.ipynb` | Loads an already trimmed raw feature CSV, removes sites with fewer than three participants, reruns ComBat, and calculates statistics/behavioral analyses. Saved GSR execution reaches 164 participants, 78 ASD/86 controls. It does not repair feature definitions or rerun outlier selection. |
| `ml-gsr-nogsr-rerun.ipynb` | Reads the old `features_ComBat_*.csv` names, uses all candidate features with fold-wise imputation/scaling, and chooses the final model by CV accuracy. It does not point to `_FIXED.csv`, rebuild features, remove outliers, or rerun ComBat. |

Cell numbers above use the JSON notebook's zero-based indices.

## 1. Exact outlier rule and order

Both feature notebooks define:

```python
vals = df[col].dropna()
q1, q3 = vals.quantile([0.25, 0.75])
lo = q1 - 1.5 * (q3 - q1)
hi = q3 + 1.5 * (q3 - q1)
return df[(df[col] >= lo) & (df[col] <= hi)].copy()
```

They call this function **once on the whole `features_df`**, using
`Whole_Energy_global`, before ComBat. There is no diagnosis-wise `groupby` or
separate ASD/control fence. Quantiles use pandas' default linear interpolation;
fence endpoints are included. Missing energy is omitted from quantile estimation
and fails the final comparisons. Infinity is not explicitly handled. This is a
single-pass rule, not iterative trimming.

The resulting participant table is reused across the main downstream analyses
within that pipeline, with additional score-specific missing-data restrictions.
GSR and no-GSR compute their own energies and masks, so the pipeline samples differ.

**Correction to the manuscript and earlier audit assumptions:** trimming was pooled
across diagnoses and applied to raw energy **before** harmonization. It was not
diagnosis-specific trimming of post-ComBat energy. The reviewer's concern about
outcome-based selection still motivates an untrimmed sensitivity analysis, but the
response should accurately distinguish the implemented rule from the described one.

## 2. Reconciled participant flow

| Stage | GSR ASD / controls | No-GSR ASD / controls | Sites |
|---|---:|---:|---:|
| Initial feature extraction, saved runs | 90 / 92 | 90 / 92 | 14 |
| After pooled raw-energy IQR | 80 / 87 | 80 / 88 | 13 |
| After excluding sites with fewer than 3 retained participants | 78 / 86 | 78 / 87 | 11 |

GSR trimming removes 15 participants (10 ASD, 5 controls); no-GSR removes 14
(10 ASD, 4 controls). The only SDSU participant is absent from both trimmed exports,
explaining the first site-count change, 14→13. CMU and UCLA_1 then contribute one
and two retained participants and explain 13→11. Thus the paper's final counts are
consistent with **two sequential exclusions**, not simply 14 sites minus two.

`participant_exclusion_ledger.csv` gives the individual IDs and stage flags for both
pipelines. The IQR exclusions are inferred by comparing all 182 source IDs with the
post-IQR exports, supported by saved logs showing 182 successful extractions. They
are not an original exclusion log containing each removed participant's energy.

**Remaining reproducibility gap:** `features_raw_*.csv` is saved *after* trimming.
The original pre-trimming energies and original numeric fences are not in these
exports. Computing quartiles on the retained table would not reconstruct the
original thresholds. Recover those features or recompute the legacy features from
the original ROI files to reproduce the numerical fences exactly.

The code's pre-feature exclusions are: unmapped file ID, fewer than 50 time points,
one-dimensional data, or ROI count other than 200; other errors are silently caught
with `except: pass`. The saved runs nevertheless report extracting all 182. These
notebooks do not implement the manuscript's age, sex, FIQ>80, right-handedness,
mean-FD, or percentage-FD inclusion/exclusion rules. Missing FIQ becomes NaN and
is median-filled in the ComBat covariate table rather than excluding the participant.

## 3. The downloaded fALFF maps were not used here

Both feature notebooks contain an active `compute_falff(ts, tr=2.0)` function and
call it with the loaded ROI time series, without overriding the default TR. Its
numerator sums **squared FFT magnitudes** at absolute frequencies 0.01–0.08 Hz;
its denominator sums squared magnitudes over the entire FFT. The result is a
regional fractional-power measure, not extraction from the downloaded voxelwise
fALFF maps and not the manuscript's described amplitude calculation.

There is no load of a fALFF NIfTI map in these feature paths. The GSR notebook uses
the earlier `abide_global/rois_cc200` dataset; the no-GSR path is named
`abide-nogsr-filt-noglobal`. The original GSR downloader requested `filt_global`.
Consequently the concern about deriving the ratio from filtered time series is
substantive for the supplied code, rather than merely a wording question about
upstream downloaded fALFF.

This refines the author's acquisition clarification: ROI time series and imaging
preprocessing were downloaded, and fALFF maps were downloaded, but these notebooks
still calculate the measure used for nodal states locally and ignore those maps.
It does not establish what an unprovided earlier notebook may have done.

The revision must choose and document the intended nodal measure. If using upstream
fALFF maps, extract them using actual atlas labels and coverage checks and establish
their upstream computation and frequency band. If calculating the measure locally,
define it accurately, use verified TRs and appropriate unfiltered inputs, and
justify how it relates to conventional fALFF. Neither route can reuse the old node
states without checking them.

## 4. Network assignment is not the stated Yeo overlap procedure

Both feature notebooks hard-code consecutive ordinal ROI ranges:

| ROI positions (1-based) | Assigned name |
|---|---|
| 1–30 | Visual |
| 31–55 | SomatoMotor |
| 56–80 | DorsalAttn |
| 81–105 | Salience/VentAttn |
| 106–125 | Limbic |
| 126–160 | Frontoparietal |
| 161–200 | Default |

The code calls `roi_to_net.get(i+1)` for each time-series column. It does not inspect
the column's atlas label, load Yeo, resample an atlas, or compute Dice overlap.
There is no evidence here that those ordinal blocks correspond to anatomical
Yeo networks. The manuscript's network names therefore need a genuine spatial
mapping and complete recomputation of intra-/inter-network features.

No common scan-length truncation is implemented in these feature loops; each
participant's original loaded time-series length is used. That also differs from
the Methods. This is a reporting discrepancy, not a recommendation to truncate
the new data automatically.

## 5. Harmonization failure and incomplete fix

Saved GSR output reports **15,364/15,364 NaNs**; no-GSR reports **15,456/15,456 NaNs**.
Both code paths explicitly substitute the raw feature matrix when more than half
the ComBat output is NaN, or when ComBat raises an exception. They then save that
matrix under `features_ComBat_*.csv`. This directly explains the previously verified
identity of the raw and “ComBat” exports.

The fix notebook excludes small sites and its saved output shows finite results
and a maximum raw-versus-adjusted difference of 89.6852 in the GSR run. That
establishes that this fix execution applied an adjustment. It does not prove
protection of diagnosis or validity of the underlying features.

**Important API issue:** all supplied ComBat calls pass `covars` and `batch_col`,
but omit `categorical_cols=['Group']` and `continuous_cols=['IQ']`. Under the
[documented neuroCombat implementation](https://github.com/Jfortin1/neuroCombat/blob/master/neuroCombat/neuroCombat.py),
those lists default to empty; the design includes only declared covariates and
batch columns. Merely including Group/IQ columns in the DataFrame does not protect
them. The notebooks do not pin/save the installed package version, so retain that
environment caveat, but their calls do not substantiate the manuscript's protected
diagnosis/IQ claims. Correcting this changes the harmonization model and may change
the results again. Do not interpret loss of significance as proof that only site
variance was removed.

The fix checks NaNs with an assertion, but does not assert all values are finite;
the “must differ from raw” check is printed rather than enforced. It saves files
with `_FIXED` names, which are not the ML rerun's inputs.

## 6. Prediction and statistical issues carried forward

- Original ML median-fills features on the full table, selects RF features once
  from the outer training set before tuning folds, and chooses its displayed best
  model by held-out accuracy. These are not the fully nested preprocessing and
  model-selection steps described in the manuscript.
- The ML rerun improves fold-wise imputation/scaling and chooses by CV score, but
  reads old non-`_FIXED` exports and uses all features. It cannot demonstrate a
  corrected harmonized classifier merely from the filenames.
- All harmonization occurs before downstream train/test splitting in these
  workflows. Descriptive group analysis and prediction need distinct designs.
- PLSR scales/imputes predictors before CV, and `cv=5` on continuous outcomes is
  not the manuscript's claimed stratified regression CV.
- PLSC code and its saved reports **are now available** in the GSR and fix notebooks;
  the earlier checklist's “no PLSC outputs supplied” statement no longer describes
  this repository. Its permutation design and sample provenance still need review.
- No whole-brain energy null-ensemble implementation was found in these four
  notebooks. The manuscript's null-model result remains untraced to code here.

## Required next step

Build a validation/reanalysis notebook using the two recorded Kaggle roots that
first preserves all available participants and produces a stage-by-stage ledger.
Audit ROI headers, atlas alignment and coverage, actual fALFF input provenance,
and raw versus corrected feature definitions **before** setting energy outlier
thresholds or testing groups. Retain all pre-trimming feature rows and numerical
fences, and include the reviewer-requested untrimmed sensitivity analysis.

Resolve demographic eligibility separately from energy trimming. Reproduce the
legacy procedure as an explicitly labeled diagnostic comparison if needed; do not
treat its trimmed CSVs as the untrimmed dataset or carry their masks automatically
into a corrected feature definition. Regenerate dependent exhibits and rewrite
Methods around the final implemented workflow.

Supporting files in this directory:

- `participant_flow.csv`: the reconciled flow above.
- `participant_exclusion_ledger.csv`: 364 pipeline–participant rows with source
  demographic fields, stage flags, and exported energy where available.
- `saved_notebook_outputs.json`: preserved textual execution output from the four
  notebooks, identified by notebook and cell index.
