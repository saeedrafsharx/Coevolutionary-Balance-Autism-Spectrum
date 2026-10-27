# Acquisition and cohort audit — October 24, 2026

**Resolved:** `Data/abide_metadata_patch_local_v2/repair_summary.json` now reports
182/182 TRs and complete reference files using the documented mirror. The last
CMU conflict was resolved from `FILE_ID=CMU_a_0050649`, giving TR=2.0 s. Both mirror
atlas payloads were verified as float32 with exact expected size and labels 0–7,
but their headers declared `bitpix=4` rather than 32. Separately named normalized
copies change only header byte 72; voxel bytes and spatial transforms are unchanged.
Use the `*.normalized.nii.gz` copies and retain their provenance JSONs. The older
failure descriptions below document the investigation, not the current repair status.

Inputs: `Data/scan_headers.json`, `Data/resource_manifest.json`, and
`Data/participants_requested.csv`, exported by the author's Kaggle run.

## Download diagnosis

The run summary reports all **1,820 derivative files** valid for all **182 subjects**.
Those derivative files are not present locally for independent inspection in this
audit; the three uploaded metadata files establish the following:

- All **182 scan-header requests succeeded**. Every header has `time_unit_code: 0`
  (unspecified), so assigning TR in seconds would require additional evidence.
  This is not a failed-header-download problem. The saved binary headers can retain
  a numeric temporal interval, but its magnitude alone does not establish units.
- The **CC200 atlas, pipeline configuration, and ROI specification downloaded**.
- The **Yeo archive returned 404** at the supplied HTTPS URL. The repair notebook
  uses the FTP endpoint used in the official
  [Nilearn atlas fetcher](https://github.com/nilearn/nilearn/blob/0.12.1/nilearn/datasets/atlas.py).
- Four PCP documentation pages failed because the HTTPS certificate did not match
  the hostname. The repair notebook retries a GitHub Pages host with certificate
  verification enabled; availability there remains untested from this workspace.

The earlier downloader should have anticipated unspecified temporal units and
provided a metadata recovery route. It now preserves `raw_pixdim4` for future runs;
it still refuses to guess a unit or supply a site-average TR.

## Cohort does not match the manuscript's stated criteria

| Check | Downloaded cohort |
|---|---:|
| Total | 182: 90 ASD, 92 controls |
| Sites | 14 |
| Sex | 182 male |
| Age range | 17.0705–29.98 years |
| Younger than 18 | 28: 12 ASD, 16 controls |
| FIQ ≤80 (observed values) | 7, all ASD |
| FIQ missing (`-9999`) | 8: 3 ASD, 5 controls |
| Explicit right-handed category | 92 |
| Explicit left-handed category | 8 |
| Explicit ambidextrous category | 3 |
| Handedness category missing/sentinel | 79 |
| Mean FD >0.5 mm or missing | 0 |

The age and FIQ flags overlap: **42 distinct participants** fail at least one of
those two stated criteria or have missing FIQ. The age-18–30/FIQ>80 subset contains
**140 participants (69 ASD, 71 controls)**. Adding an explicit right-handed category
leaves **71 (36 ASD, 35 controls)**; this is a descriptive count, not a recommendation
to exclude everyone with a missing category. Some participants have quantitative
handedness scores, which require a documented interpretation before classification.

These original 90/92 counts match the manuscript's null-model table. This helps
identify a possible analysis stage but does not validate its results or establish
the claimed initial 100/100 selection.

| Descriptive statistic (available data) | ASD | Controls |
|---|---:|---:|
| Age, mean (SD), years | 21.69 (3.67) | 22.15 (3.76) |
| FIQ, mean (SD), missing values excluded | 108.03 (15.76) | 115.08 (10.53) |
| Mean FD, mean (SD), mm | 0.0846 (0.0460) | 0.0765 (0.0380) |

These summaries describe the downloaded 182, not the final analysis sample. They
cannot be substituted for the manuscript's final-sample demographics without
recovering the analysis-stage participant manifest. No matching claim is verified.

The site-by-group table also shows diagnosis-only sites: CALTECH (10 ASD), CMU
(1 ASD), and SDSU (1 ASD). Site confounding and the flow from 14 to 13 to 11 sites
therefore require explicit reconciliation rather than a numerical correction alone.

## Saved audit tables

- `participant_eligibility_audit.csv`: source demographic/QC fields and explicit
  criterion flags for every participant; no automatic exclusion decision.
- `site_by_group_downloaded.csv`: original site-by-diagnosis counts.

## Next actions

**Repair follow-up:** the uploaded repository tree is complete and contains
site-level `task-rest_bold.json` files, not subject-named sidecars. The first repair
matched zero candidates for every participant because its matching rule was wrong.
The updated code finds candidate sources for all 182 participants in that exact
tree: one candidate for 164, four MaxMun variants for 17, and two CMU variants for
one. This verifies source discovery, not the TR values themselves. Split-site TRs
must agree before an inherited value is accepted. The first repair did successfully
recover all documentation pages; its FTP request for Yeo was refused.

The updated repair uses original-named Yeo7 files from a documented HTTPS teaching
repository mirror, validating spatial dimensions and labels 0–7. This is not a claim
of byte equality with the unavailable original distribution archive. Individual file
URLs, hashes, and the original archive failure are preserved separately.

1. Keep the successful derivative dataset. Attach it to
   `Codes/abide-metadata-repair.ipynb` on Kaggle and enable Internet. The notebook
   writes only a small supplemental bundle; it does not download participant
   derivatives again or mutate the mounted dataset.
2. Inspect `repair_summary.json`, `TR_recovery.csv`, and
   `resource_repair_manifest.json`. Exact-subject resting-state BIDS sidecars are
   sought in the official [FCP-INDI metadata repository](https://github.com/FCP-INDI/indi_bidsification/tree/master/ABIDE/scan_jsons).
   Candidate files are pinned to its retrieved commit. Missing, unidentifiable, or
   conflicting TRs remain unresolved; live recovery has not been verified locally.
3. Before inference, resolve the cohort definition: document the actual legacy
   cohort and limitations, or apply justified eligibility criteria and rerun with
   the resulting sample. Do not silently relabel the current cohort as meeting the
   paper's adult/right-handed/FIQ>80 criteria or introduce replacement subjects.
4. Continue upstream fALFF provenance verification. None of these acquisition
   checks demonstrates that the derivative's calculation used a filtered signal;
   the author's clarification about downloading precomputed derivatives still applies.
