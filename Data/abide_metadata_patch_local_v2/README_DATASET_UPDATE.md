# Completed metadata and atlas supplement

Add this entire folder alongside the original `abide_revision` folder in your
updated Kaggle dataset. Keep the original participant derivatives and metadata.
This supplement does not contain the 1,820 participant derivative files.

- `repair_summary.json`: metadata repair is complete; all 182 TRs are resolved.
- `TR_recovery.csv` and `scan_headers_resolved.json`: TRs and upstream source URLs.
  These use inherited acquisition-site metadata, with CMU_a selected explicitly
  from the participant FILE_ID. Scientific cohort eligibility remains unconfirmed.
- `atlases/Yeo7_mirror/*.normalized.nii.gz`: use these maps in reanalysis. Both
  are 256 x 256 x 256 x 1; the last dimension is singleton. The source files had
  datatype=float32 but bitpix=4 instead of 32. The corrected copies change only
  the bitpix field, preserving every voxel byte and spatial transform field.
- The corresponding original `.nii.gz` and `.rejected` files are retained as
  source evidence; use the normalized copies for standard NIfTI readers.
- Each normalized copy has a `.provenance.json` recording source and output hashes,
  the header correction, and the unchanged voxel-payload hash.
- `yeo_mirror_manifest.json` records the HTTPS mirror and downloaded file hashes.
  The original-distribution archive remains unavailable; do not claim it was fetched.
- `sidecars/` and `provenance/` preserve the metadata sources and documentation.

No scan preprocessing, fALFF calculation, participant exclusion, image resampling,
network calculation, or harmonization was performed by this repair.
