# Stage 1 assessment — October 25, 2026

Reviewed `results/abide_stage1_review/` without modifying its outputs or running
group-effect tests. Candidate feature extraction is useful but is **not yet ready
for inferential analysis**, principally because ROI coverage and cohort eligibility
remain unresolved.

## 1. Technical results

- Requested: 182 participants, two pipelines, 364 expected feature rows.
- Successful: **168 identical participants in each pipeline**, 336 feature rows.
- Those participants comprise **79 ASD and 89 controls across 13 sites**.
- No missing feature columns were found in either completed candidate table.
- Eleven participants failed in both pipelines because at least one ROI time
  series is constant. Three additional participants failed because an ROI has no
  finite in-mask fALFF voxels: 50236 (ROI 43), 51459 (43 and 52), and 51474 (120).
- The 14 technical failures comprise 11 ASD and three controls. This imbalance
  should be reported and examined, not interpreted as a diagnostic finding.
- Stage 1 removed no energy outliers and performed no ComBat or group tests.

## 2. Legacy exclusions are now reconstructed

For all **171 participants with usable time series**, the reconstructed legacy
IQR retention decisions match the historical exports exactly in both pipelines.
This improves on the earlier code-only audit:

| Cause of original exclusion | GSR | No-GSR |
|---|---:|---:|
| Constant ROI time series, giving nonfinite legacy H | 11 | 11 |
| Finite H outside the pooled IQR fences | 4 | 3 |
| Retained after original rule | 167 | 168 |

The original function calculates quartiles after `dropna()` and then filters on
the fences. Its reported “outlier removal” therefore combined **undefined energy
and finite extreme energy**. It did not identify 15/14 finite-energy outliers.
The reconstructed fences use the 171 finite energies, consistent with that code:

| Pipeline | Q1 | Q3 | Lower fence | Upper fence |
|---|---:|---:|---:|---:|
| GSR | −192.074258 | −64.081223 | −384.063812 | 127.908330 |
| No-GSR | −399.435263 | −176.671901 | −733.580307 | 157.473142 |

These numeric fences now fill the earlier reproducibility gap. The 11 constant
time-series cases explain exclusion of the only SDSU participant and the first
14→13 site change. Small-site exclusion then explains 13→11 in the old analysis.
See `legacy_exclusion_reconciliation.csv` for individual IDs and reasons.

## 3. Downloaded fALFF versus legacy calculation

For all 168 completed participants, each pipeline's filtered/unfiltered strategy
fALFF files have identical SHA-256 hashes, identical extracted ROI means, and zero
nodal-state differences. The GSR and no-GSR fALFF maps differ from one another;
their masks are byte-identical across pipelines for all 168 completed participants.

The downloaded-map states differ from the old fixed-TR spectral-ratio states by
an average of **92.50/200 ROIs with GSR** and **91.70/200 without GSR**. This is a
substantial change in the measure entering H, not simply numerical rounding.
It does not itself establish diagnostic effects or physical validity.

File identity across strategy paths does **not** establish the historical upstream
computation order, the denominator spectrum, or the exact frequency band. Continue
the source/configuration verification before writing the revised fALFF Methods.

## 4. Spatial mapping and coverage

The CC200 atlas contains 200 labels. Liberal-mask Yeo overlap assigns **184** ROIs
and leaves **16 unassigned**:

`9, 18, 30, 45, 52, 68, 80, 84, 86, 94, 130, 135, 148, 176, 178, 194`.

Zero overlap is not automatically an atlas error because Yeo7 is cortical; inspect
the locations before choosing their interpretation. Nine assignments change with
the thin versus liberal cortical mask. The three-slice preview has no obvious
gross mismatch, but it is not a full registration or region-by-region validation.
Thirty-eight ROIs have less than 50% overlap with the Yeo cortical label support.

fALFF coverage is a separate and more consequential issue. Among completed
participants, mean coverage across all participant–ROI pairs is 95.86%, but the
minimum is **1/156 voxels (0.64%)**. Examples include participant 51268/ROI 43 and
51020/ROI 43 with one and two in-mask voxels respectively. There are, on average,
9.51 ROIs below 80% coverage per participant (range 0–42).

Only **six** completed participants have ≥80% coverage in every ROI; **92** have
≥50% coverage in every ROI. Those counts are diagnostics, not proposed thresholds.
Applying an arbitrary all-ROI coverage requirement would drastically change the
cohort. Nor should a one-voxel mean silently be treated as equally representative
of its full ROI. Inspect mask support, exact constant ROI labels, anatomical
locations, and cross-pipeline consistency across **all 182**, including failures,
before selecting a defensible fixed ROI/participant policy or sensitivity analysis.
No policy should be selected to optimize group significance.

## 5. Candidate cohort definitions, before further coverage/site decisions

| Definition | ASD | Controls | Total | Sites |
|---|---:|---:|---:|---:|
| All technically complete | 79 | 89 | 168 | 13 |
| Also age 18–30 and observed FIQ >80 | 60 | 69 | 129 | 12 |
| Also explicitly categorized right-handed | 30 | 34 | 64 | 9 |

These are not final analysis samples. Of the 129 age/IQ-qualified participants,
64 have category R, six category L, 58 missing handedness category, and one a
sentinel value. Quantitative handedness scores require separate documented
interpretation. Do not call the 129 “right-handed” or exclude all missing categories
without examining those scores. Several sites are diagnosis-only or very small
even after age/IQ restrictions; adjusted modeling/harmonization must account for
the actual resulting design.

## Next step

Run `Codes/abide-revision-stage1b-qc.ipynb` with the original derivative dataset
attached. It inspects all participant–ROI combinations, allows zero support and
constant time series **for observation only**, and saves representative coverage
previews. It does not fit graph models or inferential statistics and is lighter
than Stage 1. Return `abide_stage1b_qc_review.zip` under `results/`.

In parallel, confirm whether the primary revision cohort should retain the stated
age 18–30/FIQ>80 criteria. Handedness, ROI coverage, site modeling, upstream fALFF
provenance, and the legacy bipolarity descriptor need explicit decisions before
finalizing inference. The broader cohort can remain available for a transparently
defined sensitivity analysis; do not mix these definitions in the demographics.

Supporting assessment tables: `participant_QC_and_eligibility.csv`,
`cohort_options.csv`, `legacy_exclusion_reconciliation.csv`,
`ROI_coverage_summary.csv`, and `readiness.json`.
