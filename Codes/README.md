# ABIDE acquisition and revision analyses

## Current release

The final article, plots, supporting data/code and point-by-point response are in
[revision/resubmission](../revision/resubmission/README.md). The root manuscript and
original notebooks are historical records. Follow the frozen Stage 2 protocol for
the current analysis; sections below retain the acquisition/repair execution history.

## Original classifiers and feature selection

All five original algorithm types are now complete under nested site-held-out
validation: logistic regression, RBF SVM, Gaussian naive Bayes, KNN and XGBoost.
Every classifier/pipeline has 140 OOF predictions in
`results/abide_stage2_classification_extension/`. The original ANOVA feature
selection was fitted inside inner training folds and its 50 selected-feature
sets have been reconstructed with matching predictions.

A separate, explicitly exploratory feature-selection comparison is complete in
`results/abide_ml_feature_selection_review/`. It compares all remaining features,
ANOVA, mutual information and L1 rankings after training-only removal of near-
exact redundant columns. It does not replace the original five-model results.
ROC curves, normalized confusion heatmaps, performance and feature-frequency
heatmaps are saved as PDF/PNG; all selected feature names and inner CV results
are exported.

```sh
MPLCONFIGDIR=/tmp/coev_matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv_stage2/bin/python Codes/run_ml_feature_selection.py
MPLCONFIGDIR=/tmp/coev_matplotlib .venv_stage2/bin/python Codes/update_ml_feature_selection_materials.py
```

The runner resumes completed fold checkpoints. No imaging or graph recalculation
is required. `finish_original_classifiers.sh` remains available to reconstruct
missing baseline classifier folds; all are currently complete.

## Repair the completed Stage 2 run

The uploaded run generated all 626 feature rows, but the sensitivity-power helper
aborted final group tables and the classifier had incomplete OOF folds. From a
normal terminal at the repository root, run:

```sh
bash Codes/finish_stage2.sh
```

This installs missing packages in `.venv_stage2` and invokes
`repair_stage2_results.py --complete` using the existing review CSVs. Output is
`results/abide_stage2_repaired/`. No imaging, graph, or null-model computation is
repeated. The dependency-free raw core has already been recovered locally; the
complete command finishes ComBat sensitivity and full-cohort classification.
See the repaired directory's assessment and `repair_summary.json`.

## Main reanalysis — Stage 2

Run **`abide-revision-stage2.ipynb`** on Kaggle with both existing datasets attached.
Their paths and frozen demographic/ROI masks are embedded. Enable Internet for
missing dependencies, use CPU, and run all cells with defaults. Local efficiency
and nested validation take time; progress and outputs are saved incrementally.

The primary graph uses 106 common coverage-qualified ROIs and 140 age/IQ-eligible
people; adjusted models use 136 after recorded small-site exclusions. Broader-181
and full-200 graphs are sensitivities. Read
`revision/analysis_plan/STAGE2_PROTOCOL.md` for scope and limitations.

Outputs include adjusted effects/CIs, multiple-testing correction, corrected ComBat
sensitivity, null scores, clinical/PLS results, and exploratory site-held-out models.
Inspect job status and ComBat diagnostics. Return
`/kaggle/working/abide_stage2_review.zip` under `results/`, preserving full outputs
and participant arrays separately.

## After the first Stage 1 run

The reviewed run has 168 technically complete participants in both pipelines, but
some ROI means have extremely sparse mask support. Run
**`abide-revision-stage1b-qc.ipynb`** with the original derivative dataset attached
to inspect all 182 participants and exact constant/unsupported ROIs without
repeating graph calculations. It applies no exclusion policy. Return
`/kaggle/working/abide_stage1b_qc_review.zip` under `results/`. See
`results/abide_stage1_assessment/ASSESSMENT.md` for the current assessment.

## Revision Stage 1 — run this next

Upload **`abide-revision-stage1.ipynb`** to Kaggle and attach both datasets named
in `kaggle_data_paths.json`. Their exact roots are already configured. Run all
cells on CPU; NiBabel installs only if missing, requiring Internet for that step.
Keep `COMPUTE_LOCAL_EFFICIENCY=True` for the full candidate export. This step can
take time because it evaluates shortest paths in many neighborhood subgraphs.

The notebook reads downloaded fALFF maps, binds time-series columns to actual
CC200 labels, constructs spatial Yeo overlap mappings, computes untrimmed raw
features, and reconstructs the legacy fixed-TR energy/outlier mask for comparison.
It saves coverage/technical failures, input hashes, participant arrays, and all
IQR flags without removing outliers. It does not run ComBat or group statistics.
The anatomical mapping and scientific cohort remain provisional pending review.

Download `/kaggle/working/abide_stage1_review.zip` and put it in this repository's
`Data/` directory. It contains the tables/configuration/preview needed to review
the run, excluding the larger participant arrays and FC matrices. Preserve the
full `/kaggle/working/abide_revision_stage1` output separately for reanalysis.

The testable implementation is `abide_stage1.py`; the notebook embeds the same
code and runs without uploading that helper separately. Local tests cover label
ordering, affine resampling, Dice mapping, masked means, energy decomposition,
efficiency, median ties, pooled IQR, and a full synthetic run with mocked image I/O.
Full image-file validation requires the Kaggle run; the participant derivatives
and NiBabel are not available in the local sandbox.

## Acquisition and metadata repair

To repair from a local terminal, with Python, NumPy, pandas, and Internet access:

```sh
python3 Codes/repair_abide_locally.py
```

This reads the supplied files in `Data/`, reuses the existing repair cache, and
writes `Data/abide_metadata_patch_local/`. It also accepts `--dataset-root`,
`--output-root`, `--cache-root`, and `--offline`. A session with
`CODEX_SANDBOX_NETWORK_DISABLED=1` performs cache validation only and reports the
missing downloads explicitly. Run the same command in a normal terminal to finish
retrieval. Keep the original derivative dataset when incorporating the patch.

If the full derivative download has already completed, use
**`abide-metadata-repair.ipynb`** with that dataset attached on Kaggle. It attempts
TR recovery from upstream BIDS sidecars and retries failed reference
resources, writing only a small supplement. It does not repeat derivative downloads.
Read `revision/ACQUISITION_AUDIT.md` for the observed failures and cohort discrepancies.
The revised repair handles the repository's actual site-level metadata layout,
labels inherited TR sources, and rejects conflicting acquisition variants. Yeo7
maps are acquired from an explicitly identified HTTPS mirror with checksums and
label validation; the original archive failure remains in the provenance record.

Upload **`abide-gsr-downloader.ipynb`** to Kaggle, enable Internet, and run all cells.
It is self-contained: you do not need to upload the Python helper or local data.
CPU is sufficient. The filename is retained for continuity; the revised notebook
downloads **both GSR and no-GSR** strategies.

The default acquisition preserves the original notebook's **182 IDs**, without
applying scientific exclusions or substituting participants. Change the explicit
`SUBJECT_IDS` configuration only if a documented original cohort is recovered.
Age, sex, FIQ, and mean-FD eligibility flags are reported for review, not used to
silently discard participants. All original phenotype columns, including clinical
scores, handedness, diagnosis, and QC information, are preserved.

For each participant, the notebook requests:

- CC200 ROI time series and fALFF maps for `filt_global`, `filt_noglobal`,
  `nofilt_global`, and `nofilt_noglobal` (eight files).
- Brain masks for the two `nofilt_*` strategies (two files).
- A header-only read from a 4D `nofilt_noglobal` derivative for TR and scan length.

It also retrieves the upstream CC200 atlas, the Yeo atlas archive, phenotype CSV,
and PCP methods/configuration snapshots. A default run requests **1,820 derivative
files**, plus reference resources and small header requests. No full 4D scans are
retained. No fALFF, time series, filtering, truncation, or ComBat computation is
performed locally.

Create the Kaggle dataset from the entire **`/kaggle/working/abide_revision`**
directory. Keep its `metadata/`, `provenance/`, `atlases/`, receipts, and README.
Inspect `metadata/completion_summary.json`, `subject_completeness.csv`, and failure
manifests first. Unavailable derivatives remain explicit failures; another strategy
is never silently substituted. Rerunning validates cached files against their URL
receipts, SHA-256 checksums, and file structure before reuse. Optional ZIP creation
is disabled by default to avoid duplicating storage.

This acquisition alone does not establish the actual upstream fALFF computation
branch or resolve the paper's 0.08-Hz versus published configuration's 0.1-Hz upper
cutoff. The QA documentation defines `func_perc_fd` at **0.2 mm**, which cannot be
used as the paper's percentage of frames exceeding **0.5 mm**. Per-volume motion
traces are not included in these downloads. See the output README for remaining
reanalysis decisions.

`abide_download.py` is the testable copy of the helpers embedded in the notebook.
If editing helpers, synchronize the notebook's `download-helpers` cell; an offline
test checks equality. Run verification from the repository root:

```sh
python3 -m unittest discover -s Codes -v
```

The tests use synthetic files and mocked HTTP responses, including a complete
acquisition simulation. They do not establish live S3 availability. The original
notebook is preserved at `revision/original_abide-gsr-downloader.ipynb`.
