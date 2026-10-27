> **Superseded working draft.** Use [../resubmission/README.md](../resubmission/README.md) for the final manuscript, current fALFF clarification, response and upload files. The text below records an earlier preparation stage.

# Working revision package

This folder preserves the submitted root manuscript and contains a replacement
revision draft, not a submission-ready release.

## Read first

- `manuscript_revision.pdf`: corrected main draft; editable source is the `.tex` file.
- `response_to_reviewers.md`: point-by-point response with unresolved items marked.
- `supporting_information.pdf`: separate S1 supplement.
- `S2_supporting_tables_DRAFT.zip`: indexed files with verified checksums.
- `FALFF_PROVENANCE_GATE.md`: exact remaining source-provenance questions and
  optional inquiry text; nothing has been sent to third parties.
- `S2_file_index.csv`: inventory of supporting data.
- `FIGURE_INVENTORY.csv`: disposition of every original image and its corrected replacement.
- `../stage2_verified/control_nulls/`: explicit 71-male-control null baseline.

The main draft has eight figures and S1 has ten, including the exploratory
feature-selection extension. Dedicated global,
within-network, between-network, H--metric, clinical/PLS, classification and
GSR/no-GSR Results sections preserve the original scientific organization.
The original Introduction--Results--Discussion--Methods order is restored.

See [the continuity and significance audit](../stage2_verified/continuity/CONTINUITY_AND_SIGNIFICANT_COMPARISONS.md)
and its full analysis map. All 364 original exported H correlations and FDR
probabilities were numerically reproduced. The original no-GSR ASD efficiency
association exists in those supplied data; its absence in a different feature/
cohort definition does not establish falsity. The revised raw descriptive
H analysis has 59 per-group-BH-significant associations, separately from the
nonsignificant diagnosis contrasts and adjusted efficiency tests.

<!-- CLASSIFICATION_README_BEGIN -->
Classification now includes 5 original algorithm types and 10 complete
pipeline/model pairs: LogisticRegression, SVM_RBF, GaussianNB, KNN, XGBoost. All original algorithm types are complete.

The original spectral descriptor remains separately reported as exploratory,
outside the frozen 88-feature schema. Matching algorithms does not equate different
cohorts/features/validation. See results/abide_stage2_classification_extension/.
<!-- CLASSIFICATION_README_END -->

The new exploratory restriction to mixed-diagnosis sites gives n=127 (58/69),
with no detected adjusted diagnosis effect. Site partial R² decreases after
ComBat, but those fitting-sample summaries do not prove elimination of site effects.
See `../stage2_verified/diagnostics/` for all checks. No features were recomputed.

## Rebuild locally

From the repository root, with the Stage 2 environment available:

```sh
.venv_stage2/bin/python Codes/stage2_reviewer_diagnostics.py
MPLCONFIGDIR=/tmp/coev_matplotlib .venv_stage2/bin/python Codes/build_revision_exhibits.py
.venv_stage2/bin/python Codes/report_control_nulls.py
MPLCONFIGDIR=/tmp/coev_matplotlib .venv_stage2/bin/python Codes/restore_revision_figures.py
.venv_stage2/bin/python Codes/audit_manuscript_continuity.py
MPLCONFIGDIR=/tmp/coev_matplotlib .venv_stage2/bin/python Codes/build_continuity_materials.py
.venv_stage2/bin/python Codes/package_revision_support.py
```

Then, from this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript_revision.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supporting_information.tex
```

The main draft uses the original PLOS template/style; the bibliography is an
isolated revision copy with added verified references. Original source files and
analysis uploads remain unchanged. Files containing `DRAFT` require author review
before public deposit. Supporting ZIP regeneration does not publish anything.

## Required before submission

1. Resolve the historical fALFF input/ratio/band/smoothing provenance and finalize
   the Methods and corresponding responses. Independent recomputation may be
   needed if adequate evidence cannot be obtained.
2. Confirm funding, interests, and author contribution statements against the
   submission record. The Methods now explain the public deidentified secondary-
   analysis basis and the original ABIDE approvals/consent; no new institutional
   approval number or formal exemption is asserted.
3. Author review of the corrected 106-node scope, relative-state interpretation,
   cohort/handedness disclosure, results, and limitations; final literature check.
4. Deposit the permitted code/data release and replace the public-archive placeholder.
5. Final editorial pass, page/line references in the response, tracked-changes
   manuscript, and journal figure/file checks. The current PDFs are review drafts.

A replacement analysis changes many sections; the marked manuscript should be
made against the preserved submitted source after the scientific wording is
settled. This folder does not claim that the final marked submission exists.

## Feature-selection extension

All five original classifiers are complete. Feature selection was already applied
within training folds; its reconstructed selected feature names and score checks
are now saved. An additional explicitly exploratory comparison evaluates ANOVA,
mutual information, L1 ranking and no further selection after training-only
redundancy pruning. No clear gain is established; baseline results are preserved.

Read [the feature-selection report](../stage2_verified/ml_feature_selection/FEATURE_SELECTION_AND_RESULTS.md).
Plots (PDF/PNG) are in results/abide_ml_feature_selection_review/figures/.
The main paper retains eight figures; S1 now has ten, including follow-up ROC,
confusion, performance and feature-selection frequency heatmaps.

To reproduce this optional follow-up and refresh its materials:

```sh
MPLCONFIGDIR=/tmp/coev_matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv_stage2/bin/python Codes/run_ml_feature_selection.py
MPLCONFIGDIR=/tmp/coev_matplotlib .venv_stage2/bin/python Codes/update_ml_feature_selection_materials.py
```

The runner reuses complete fold checkpoints. The main five-model pipeline is
unchanged; follow-up output has its own plan, source hashes, versions and results.
