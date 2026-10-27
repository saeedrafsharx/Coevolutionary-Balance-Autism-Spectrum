# Data acquisition clarification

**Subsequent code evidence:** the four analysis notebooks now supplied show that
both main feature pipelines call `compute_falff(ts, tr=2.0)` on ROI time series and
do not read the downloaded fALFF maps. Thus the author's acquisition description
below does not establish which measure entered the analysis. See
[the original-code audit](original_code_audit/ORIGINAL_ANALYSIS_AUDIT.md). Imaging
preprocessing/time-series acquisition remain upstream; the active spectral-ratio
calculation in these notebooks is local and must be reconciled with the manuscript.

## Author clarification — October 24, 2026

The author downloaded **precomputed ABIDE derivatives**. The author did not compute
the source ROI time series, preprocess the imaging, or calculate the source fALFF
maps locally. The original acquisition notebook is now available in `Codes/`.

The manuscript's current first-person preprocessing/fALFF description must therefore
be checked against **PCP's actual upstream methods**, separately from any subsequent
local parcellation of fALFF maps or network/feature analysis. The earlier revision
checklist's fALFF concern is an upstream-provenance question; it is not evidence that
the author personally applied a filter before calculating fALFF.

## What the original notebook establishes

- It contains 182 explicit participant IDs.
- It requested C-PAC `filt_global/falff` and `filt_global/rois_cc200` derivatives.
- Its existence checks did not verify complete downloads or content checksums.
- The strategy name alone does not establish which temporal branch fed upstream
  fALFF computation. This still needs source/configuration verification.

## Revised acquisition

The modified notebook preserves those IDs, requests both derivatives under all four
PCP strategies, includes unfiltered masks and CC200/Yeo reference resources, and
exports participant metadata, eligibility flags, scan-header TR, and provenance.
It does not perform scientific exclusions or reanalysis. The initial 200-person
cohort described by the manuscript cannot be recovered from the 182-ID list alone.

Upstream [PCP configuration](https://github.com/preprocessed-connectomes-project/abide/blob/master/preprocessing/settings/pipeline_config_abide_rerun.yml)
specifies fALFF cutoffs 0.01–0.1 Hz, versus 0.01–0.08 Hz in the current manuscript.
The [PCP QA documentation](https://preprocessed-connectomes-project.org/abide/quality_assessment.html)
defines `func_perc_fd` as the percentage of frames above 0.2 mm. Neither value should
be silently relabeled to match the current manuscript.

These are documented discrepancies to resolve, not a finding that every upstream
map is invalid or that local recomputation is necessarily required.
