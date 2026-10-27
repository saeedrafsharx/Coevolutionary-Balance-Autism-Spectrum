# Stage 1B assessment — October 25, 2026

Reviewed all QC tables and representative previews without modifying source outputs
or running group-effect tests. Scientific exclusion rules remain unselected.

## Confirmed findings

Stage 1B inspected all **182 participants in both pipelines**, producing 72,800
participant–ROI observations with zero processing failures. Successful QC inspection
does not mean all observations are suitable for analysis: this notebook deliberately
records constant time series and zero mask support instead of rejecting them.

The constant-time-series flags, finite masked voxel counts, and coverage fractions
agree exactly across GSR and no-GSR for every participant–ROI combination.

- Eleven participants have constant time series in one or more ROIs; **21 distinct
  ROIs** are affected somewhere in the sample.
- Fourteen participants have an ROI with zero finite in-mask fALFF support;
  **24 distinct ROIs** are affected. This includes the eleven above and the three
  additional fALFF-only failures found in Stage 1.
- The union of constant/unsupported labels contains **26 ROIs**. The remaining
  **174 ROIs** never have either undefined measure across all 182 participants.
  This is a descriptive common set, not an adopted or sufficient coverage rule.
- Constant time series and spatial support are separate checks: participant 51466
  has constant time series in ROIs 104, 113, and 169 despite 10, one, and one
  in-mask voxels respectively.

The previews show affected parcels outside or at the edge of available image
support: examples include 50236/ROI 43 (0/156 supported voxels), 51268/ROI 43
(1/156), 51459/ROI 52 (0/161), and 51017/ROI 10 (3/204). These support a spatial
availability problem rather than an energy-outlier explanation. The snapshots do
not distinguish field-of-view limitation, dropout, registration, and masking causes.

## Participant retention under full-200-ROI rules

Each row requires nonconstant time series and nonzero fALFF support in every ROI,
plus the stated minimum fraction. These thresholds are **diagnostics only**.

| Minimum coverage in every ROI | All downloaded: ASD / controls | Also age 18–30, observed FIQ >80: ASD / controls |
|---|---:|---:|
| Nonzero support | 79 / 89 (168) | 60 / 69 (129) |
| 10% | 67 / 83 (150) | 50 / 66 (116) |
| 20% | 60 / 74 (134) | 43 / 58 (101) |
| 50% | 44 / 48 (92) | 31 / 38 (69) |
| 80% | 3 / 3 (6) | 1 / 2 (3) |

The age/IQ-qualified pool before full-200 QC is **140**. A different fixed ROI
policy would require recomputing participant eligibility, not automatically carrying
forward the Stage 1 mask of 129. Handedness still needs separate interpretation.

An all-parcel 80% rule would remove almost the entire sample. Nonzero-only support
still allows one-voxel parcel means. Neither should become the primary rule simply
because it yields a convenient sample size or significant result.

## Anatomical network assignment also needs refinement

Stage 1 assigns any parcel with a unique positive Dice maximum. This includes
ROI 10 with **1.47%** cortical label overlap and ROI 120 with **0.40%**, both labeled
“Visual.” A tiny intersection does not establish that an entire parcel belongs to
that cortical network. ROI 52 has zero overlap and remains unassigned.

There are **38 parcels below 50% cortical label overlap**, including the 16 already
unassigned. An explicit cortical eligibility rule should precede maximum-overlap
assignment. Requiring a majority of parcel voxels within cortical label support is
one interpretable candidate, not a previously prespecified or validated threshold.
Inspect locations and thin/liberal sensitivity when documenting the final rule.

Whole-brain inclusion and cortical-network assignment are separate decisions.
Unassigned parcels need not automatically disappear from a whole-brain measure.
Any reduced-node analysis must disclose its actual node set and scope.

## Recommended next step

No further broad QC acquisition run is needed to establish these counts. Lock a
revision-stage analysis design before examining corrected group effects:

1. **Demographic scope:** retaining age 18–30/FIQ>80 as primary offers continuity
   with the paper; the broader downloaded cohort can be a sensitivity sample.
   Resolve handedness from the available category/scores separately.
2. **Measurement availability:** define a fixed coverage-qualified node set shared
   across participants/pipelines and/or justified participant-level criteria. The
   174 undefined-free nodes are only a starting diagnostic. Do not replace missing
   signals with zero or use a different graph size for each participant.
3. **Cortical eligibility:** prevent tiny intersections from assigning noncortical
   or predominantly uncovered parcels to named Yeo networks.
4. **Sensitivity plan:** specify broader-cohort, ROI/coverage, GSR/no-GSR, and
   untrimmed/legacy-trimmed comparisons transparently. These are revision-stage
   methodological choices, not a priori study design.

A changed node set requires recomputing the participant median states, H, graph
descriptors, and block features. After the manifests are fixed, rebuild features and
perform inference. Upstream fALFF provenance and the original ComBat/prediction
issues also remain to be resolved before resubmission.

Supporting tables in this directory: `all182_participant_QC.csv`,
`ROI_QC_and_anatomical_support.csv`, `full200_coverage_policy_comparison.csv`,
`174_ROIs_without_undefined_measure_descriptive_only.csv`, and
`38_ROIs_with_low_cortical_overlap.csv`. No exclusions have been applied.
