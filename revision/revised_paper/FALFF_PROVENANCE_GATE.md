# Historical fALFF provenance: submission gate

## Verified evidence

The revision calculates FC locally from downloaded filtered CC200 series and
aggregates downloaded voxelwise fALFF maps locally. It does not compute a fresh
voxelwise fALFF denominator. A `nofilt` folder name alone cannot resolve the
reviewers' filtering concern.

- [PCP C-PAC documentation](https://preprocessed-connectomes-project.org/abide/cpac.html)
  supplies the stated nuisance model and FC filtering range.
- [Archived release configuration](https://github.com/preprocessed-connectomes-project/abide/blob/master/preprocessing/settings/pipeline_config_abide_rerun.yml)
  specifies fALFF high-pass .01 Hz and low-pass .1 Hz. This does not match the
  submitted .01–.08-Hz description.
- [PCP derivative mapping](https://github.com/preprocessed-connectomes-project/abide/blob/master/map_rawout_to_organized_s3.py)
  maps `falff` to `falff_to_standard_smooth`. It indicates that the distributed
  derivative is a standard-space smoothed fALFF output, rather than a fresh ROI
  spectral calculation. It does not fully identify its input branch or kernel.
- [Current C-PAC implementation](https://github.com/FCP-INDI/C-PAC/blob/main/CPAC/alff/alff.py)
  documents a filtered/unfiltered temporal-standard-deviation ratio. Current code
  is not release-specific proof and its description should not be substituted
  for the historical implementation.
- Saved Stage 1 checks found identical filtered/unfiltered-strategy fALFF maps
  for successfully processed participants within each global-signal choice.
  Equality supports shared outputs, not proof of a valid unfiltered denominator.

Historical tag URLs could not be retrieved through the available web tool; this
is an access limitation, not evidence that a version or source does not exist.

## Evidence needed to close this item

Identify the release's actual C-PAC commit/version and recover its ALFF workflow
and pipeline connections (or equivalent archived execution provenance). Verify:

1. The denominator receives unfiltered nuisance residuals, and whether numerator
   filtering or other transformations precede the calculation.
2. The actual ratio definition (amplitude sums, square-root power, temporal SD,
   or another implementation), frequency bounds, and treatment of the sampled
   spectrum and varying TR.
3. Whether the downloaded `falff_to_standard_smooth` output includes smoothing,
   its kernel, and its relationship to the functional mask and ROI extraction.
4. The link from that code/configuration to the distributed PCP release, rather
   than merely a plausible historical software implementation.

If those facts cannot be established, decide whether independently recomputing
voxelwise fALFF from suitably documented unfiltered data is required. This would
be a new analysis branch with fresh provenance and downstream feature/inference
reruns, not relabeling the existing map or repeating the completed repair.

## Optional inquiry text for the authors to send

We are revising an ABIDE I analysis using PCP C-PAC `falff` derivatives downloaded
from the public `fcp-indi` bucket. Could you identify the original C-PAC version or
commit and the workflow/input branch generating `falff_to_standard_smooth` in
this release? The archived rerun configuration sets fALFF bounds to .01–.1 Hz.
We need to verify whether its denominator used unfiltered nuisance residuals,
the exact ratio definition and TR handling, and the spatial smoothing kernel.
A release-specific code reference or execution/configuration record would help
us describe the derivative accurately in response to methodological review.

No inquiry has been sent from this workspace.
