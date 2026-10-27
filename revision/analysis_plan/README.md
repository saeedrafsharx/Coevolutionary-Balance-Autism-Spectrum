# Revision analysis plan

**Current plan:** [Stage 2 protocol](STAGE2_PROTOCOL.md) and
`stage2_frozen_design.json` now fix the coverage/ROI design and inferential workflow.
The pending-coverage text below records the earlier decision stage; it is superseded
by that protocol and its explicit primary/sensitivity manifests.

## Confirmed primary demographic criteria

The author confirmed on October 25, 2026:

- Age at scan **18–30 years inclusive**.
- Observed full-scale IQ **strictly greater than 80**.
- Missing/sentinel IQ is not imputed to make a participant eligible.

Applied to the downloaded cohort of 182, these criteria yield **140 eligible
participants: 69 ASD and 71 controls**. The manifest is demographic eligibility,
not a final analysis cohort. The previous count of 129 also imposed availability
of all 200 ROIs; do not carry that mask forward before choosing the ROI policy.

`confirmed_cohort_specification.json` records the confirmed criteria and pending
decisions. `primary_demographic_manifest_140.csv` contains eligible participants;
`all182_demographic_eligibility_ledger.csv` records all inclusion/exclusion reasons.
Original metadata are retained without overwriting source datasets.

## Coverage design still pending

`coverage_design_options_not_applied.csv` describes fixed-node alternatives using
only time-series constancy and mask support within the eligible pool. No corrected
group effects or significance values were used, and no option has been selected.
Thresholds are prospective revision-stage choices, not the original study's
prespecified methods or universal validity standards.

For example, requiring at least 50% coverage and nonconstant time series in at
least 95% of eligible participants qualifies 181 ROIs; requiring every selected
ROI to meet that coverage rule then leaves 104 participants (48 ASD, 56 controls).
Requiring the same quality in all 140 eligible participants instead yields a
common 106-ROI set with all 140 retained. These options change anatomical scope
and sampling differently; neither is adopted merely to retain a convenient N.

Whole-brain node eligibility must be separated from cortical network assignment.
Tiny Yeo overlaps should not label a predominantly noncortical parcel as a named
cortical system. Any changed node mask requires rebuilding median nodal states
and all dependent features with identical masks across participants/pipelines.

Handedness remains a reporting/eligibility question, not an assumed right-handed
status for all 140. Site structure, upstream fALFF provenance, harmonization, and
prediction validation must also be resolved before the final statistical run.
