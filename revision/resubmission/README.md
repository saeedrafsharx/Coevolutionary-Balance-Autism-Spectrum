> **Submitted:** The user confirmed approval of revision PONE-D-26-34219R1 in the journal portal. See submission_status.json and portal_pdf_review.json for the approved build record.

# Final revision files — November 5, 2026

Start with **manuscript_illustrated.pdf** to read the complete article with plots
beside the results. The original section order and group/model colors are preserved.

Follow [UPLOAD_README.txt](UPLOAD_README.txt) and [UPLOAD_ORDER.csv](UPLOAD_ORDER.csv)
when uploading. Extract PLOS_revision_upload.zip and upload its files individually.

## Files for the PLOS revision

- `manuscript_submission.pdf`: clean manuscript, inline figure captions, continuous
  page/line numbers, and separate supporting-information captions after references.
- `manuscript_marked.pdf`: marked main-text changes against the original submission.
  Includes the removed original main tables in a clearly labeled deletion appendix,
  in addition to marked narrative changes and replacement tables.
- `response_to_reviewers.pdf`: separate point-by-point response with manuscript locations.
- `supporting_information.pdf`: S1 Appendix with 13 supporting figures and diagnostics.
- `S2_Data.zip`: indexed analysis data, complete results and provenance.
- `S3_Code.zip`: revision analysis/reporting code, plans and execution documentation.
- `upload_figures/Fig1.tif`–`Fig11.tif`: 600-dpi RGB LZW TIFF figures.
- `manuscript_source.zip`: clean LaTeX source, bibliography, style and inline tables.
- `PLOS_revision_upload.zip`: all upload files assembled together; excludes the
  illustrated reading copy and LaTeX auxiliary files.

The illustrated reading copy and the submission-format manuscript contain the same
article. Use the submission-format copy for upload and supply figure files separately.
The point-by-point response locations refer to manuscript_submission.pdf.

## fALFF clarification retained in the submission

PCP/C-PAC computed the downloaded voxelwise fALFF maps. Local processing extracts
regional map means and calculates FC/network/statistical quantities; it does not
compute the voxelwise fALFF denominator. Provider configuration documents a
0.01–0.1 Hz fALFF band. Exact historical denominator/filter-order details are not
independently verified and are disclosed as an upstream-documentation limitation.
The response does not claim provider confirmation that has not been obtained.
This uncertainty alone does not establish that the maps are invalid or that
recomputation is mandatory. The editor will assess this transparent explanation.

The manuscript uses the supplied S2/S3 files as its exact analysis release rather
than relying on the old GitHub README. No external repository was edited and no
files were submitted to the journal automatically. Retain the existing accurate
portal funding, competing-interest and author declarations when uploading.

## Local rebuild

Run from this directory:

```
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript_illustrated.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript_submission.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supporting_information.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error response_to_reviewers.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript_marked.tex
```

The root manuscript, bibliography and AGENTS.md are unchanged. See validation.json
for build, callout, archive-integrity and figure checks.
