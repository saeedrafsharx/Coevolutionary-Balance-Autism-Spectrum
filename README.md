# Coevolutionary balance of resting-state brain networks in autism

Revision materials for the PLOS ONE manuscript by S. Rezaei Afshar and G. R. Jafari.
The current article and exact analysis release are in **[revision/resubmission](revision/resubmission/README.md)**.

## Current manuscript and submission files

- [Clean manuscript](revision/resubmission/manuscript_submission.pdf): PLOS-format
  article with inline captions and separate figure files.
- [Illustrated manuscript](revision/resubmission/manuscript_illustrated.pdf): same
  article with plots placed beside the results, for reading.
- [Point-by-point response](revision/resubmission/response_to_reviewers.pdf).
- [Marked manuscript](revision/resubmission/manuscript_marked.pdf).
- [Supporting information](revision/resubmission/supporting_information.pdf).
- [Complete upload package](revision/resubmission/PLOS_revision_upload.zip): clean
  and marked articles, response, supporting PDF, data/code archives, LaTeX source
  and 11 TIFF figures. The journal submission itself is performed separately.

All final documents build successfully; checks are recorded in
[validation.json](revision/resubmission/validation.json). The original root-level
`manuscript.tex`, `Bibliography.bib`, and `Images/` are historical submission
materials, **not the current revised article**.

## Data and analysis scope

ABIDE I source data and PCP C-PAC derivatives underlie this secondary analysis.
MRI preprocessing, regional time-series extraction and voxelwise fALFF computation
were performed upstream. Local code extracts regional fALFF map values and
computes FC, graph summaries, statistical tests and prediction models.
Provider documentation specifies a 0.01–0.1 Hz fALFF band; exact historical
fALFF denominator/filter-order details have not been independently verified.
This documentation limitation is explicitly described in the Methods and reviewer
response; no provider confirmation is claimed.

The revision uses 140 eligible adult male participants, a common coverage-qualified
106-node primary graph, and 136 participants for the adjusted group model.
All five original classifier families are retained with nested site-disjoint
validation. The broader feature-selection follow-up is explicitly exploratory.
See the manuscript for complete definitions, uncertainty and limits of inference.

## Repository layout

| Path | Contents |
| --- | --- |
| `Codes/` | Acquisition, QC, statistical analysis, prediction, reporting and tests |
| `Data/` | Download metadata, atlas/reference resources and original result exports |
| `results/` | Stage-specific QC, verified reanalysis and complete classifier outputs |
| `revision/analysis_plan/` | Frozen revision design, participant/ROI manifests and follow-up plans |
| `revision/stage2_verified/` | Audits, diagnostic summaries and continuity analyses |
| `revision/resubmission/` | Final manuscript source, PDFs, figures and submission archives |
| `revision/revised_paper/` | Superseded working-draft source retained for historical audit |
| `revision/original_code_audit/` | Audit of original notebooks and processing assumptions |
| `Images/` | Original manuscript figures; retained as historical records |

Raw participant MRI is not included. Input Kaggle datasets are
`saeedrezaeiafshar/abide-full-derivatives-dataset` and
`saeedrezaeiafshar/abide-metadata-patch`; see [Codes/README.md](Codes/README.md)
for notebook configuration. Complete machine-readable revised results are also
provided in `revision/resubmission/S2_Data.zip`; the corresponding analysis code
is supplied as `S3_Code.zip`.

## Build and verify

With TeX Live, BibTeX and latexmk installed:

```sh
cd revision/resubmission
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript_submission.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript_illustrated.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supporting_information.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error response_to_reviewers.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript_marked.tex
```

Python dependencies and versions used are documented in S3 Code and the results'
audit records. Original and revised notebooks are distinguished in Codes/README.md.
Do not rerun historical notebooks as a substitute for the frozen revision analysis.
Local environments, temporary caches, LaTeX intermediates, decision-letter/private
correspondence and session identifiers are excluded from version control.
