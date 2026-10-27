"""Finalize manuscript and response wording; preserve upstream provenance uncertainty."""
from pathlib import Path
import re
R=Path(__file__).resolve().parents[1]/'revision/resubmission'
p=R/'manuscript.tex';s=p.read_text()
a=s.index('Historical PCP fALFF input and denominator provenance');b=s.index('\\paragraph*{Conclusions.}',a)
s=s[:a]+'''The analysis uses the publicly distributed PCP fALFF maps as provided. We did
not generate their voxelwise spectral numerator or denominator. Detailed
historical implementation choices are therefore inherited from the derivative
provider rather than independently reconstructed here. This is a limitation of
upstream provenance, not evidence that the downloaded maps were calculated
incorrectly. Physiological interpretation remains restricted to a relative
alignment statistic derived from those maps.

'''+s[b:]
s=s.replace('and verified derivative provenance are necessary for stronger conclusions.', 'and more complete characterization of upstream derivatives would strengthen\nassessment of the framework.')
a=s.index('\\linelabel{lin:falff_provenance}');b=s.index('\\subsection*{Coverage-defined',a)
s=s[:a]+r'''\linelabel{lin:falff_provenance}\textbf{Provider-derived fALFF and scope of local processing.}
We used the voxelwise fALFF maps distributed by PCP for ABIDE I, generated
upstream with C-PAC. Our local processing extracted regional means from these
maps; it did not compute voxelwise fALFF, its spectral denominator, or a new
amplitude ratio from the filtered FC time series. The provider's archived
configuration lists 0.01--0.1 Hz fALFF bounds \cite{PCPconfigRevision}, and its
output mapping identifies the distributed map as a smoothed standard-space
fALFF derivative \cite{PCPmappingRevision}. Accordingly, the former description
of a locally applied Butterworth filter followed by a 0.01--0.08 Hz numerator
over a fixed 0--0.25 Hz denominator does not describe the present analysis and
is not retained. The downloaded maps are used without local spectral
recalculation, including for participants with TR of 3 s.

The precise historical denominator implementation, input filtering order, and
smoothing kernel were not independently established from the available
release records. We therefore attribute the upstream computation to PCP/C-PAC
and do not claim to have verified an unfiltered denominator. Selecting a
\texttt{nofilt} strategy directory alone does not establish that implementation.
This reporting limitation does not itself demonstrate an error in the provider's
maps, but it limits claims about their exact spectral construction. The
regional aggregation, coverage rules, state definition, FC calculation, and
subsequent statistics are specified below and supplied in S3 Code.

'''+s[b:]
s=s.replace('supplied separately in S1 and S2 Data.', 'supplied separately in S1 Appendix and S2 Data.')
s=s.replace('Figure~\\ref{fig:classifiers} summarizes uncertainty. Every conditional interval includes .5.', 'Every conditional interval includes .5.')
# Quiet hyperlink borders in both final views.
s=s.replace('\\usepackage{nameref,hyperref}', '\\usepackage{nameref,hyperref}\n\\hypersetup{hidelinks}')
p.write_text(s)
p=R/'Bibliography.bib';s=p.read_text()
if 'PCPmappingRevision' not in s:s+=r'''
@misc{PCPmappingRevision,
 author={{Preprocessed Connectomes Project}},
 title={{ABIDE} derivative output mapping: map\_rawout\_to\_organized\_s3.py},
 howpublished={\url{https://github.com/preprocessed-connectomes-project/abide/blob/master/map_rawout_to_organized_s3.py}},
 note={Accessed October 27, 2026}}
'''
s=re.sub(r'(@misc\{PCPconfigRevision,.*?)(https://github.com/preprocessed-connectomes-project/abide)([^\w/])',r'\1https://github.com/preprocessed-connectomes-project/abide/blob/master/preprocessing/settings/pipeline_config_abide_rerun.yml\3',s,flags=re.S)
p.write_text(s)
p=R/'supporting_information.tex';s=p.read_text().replace('Historical fALFF provenance remains limited as described in the main Methods.', 'The main Methods distinguish downloaded PCP fALFF derivatives from local regional aggregation and describe the limits of upstream documentation.');p.write_text(s)
p=R/'response_to_reviewers.md';s=p.read_text()
a=s.index('**Author review version.**');b=s.index('Dear Editor',a)
s=s[:a]+'''Locations refer to manuscript_submission.pdf; line numbers identify the start of the relevant passage. Reviewer comments are reproduced from Comments.pdf. We distinguish corrections to our analysis and reporting from historical implementation details that remain undocumented by the derivative provider.

'''+s[b:]
a=s.index('**Partly addressed; critical provenance item open.**');b=s.index('## Journal requirements',a)
s=s[:a]+'''**Response.** Thank you for identifying this ambiguity. The revised analysis uses voxelwise fALFF maps distributed by the Preprocessed Connectomes Project (PCP) for ABIDE I and generated upstream using C-PAC. We did not preprocess MRI, extract the downloaded ROI time series, or calculate voxelwise fALFF. Locally, we calculated FC from the downloaded regional time series and extracted regional means from the downloaded fALFF maps. We have rewritten the Methods to make these responsibilities explicit.

Our audit also identified a fixed-TR spectral ratio calculated from filtered ROI series in historical notebooks. That legacy calculation is excluded from the revised analysis. The revised analysis uses the provider's maps directly, with no new voxelwise denominator or spectral calculation. We have removed the former Butterworth-filter/0.01–0.08-Hz/fixed-0.25-Hz description because it does not describe the derivatives used for the revision.

The provider's archived configuration specifies 0.01–0.1 Hz fALFF bounds, and its mapping identifies the distributed smoothed standard-space fALFF output. These records do not independently establish the exact historical input branch, denominator implementation, or smoothing kernel. We explicitly report this limitation and do not claim that an unfiltered denominator has been verified. Using a downloaded derivative establishes its source; it does not by itself verify every upstream step. The revised Methods therefore provide the documented information, distinguish all local processing, and limit the interpretation accordingly. We have neither inferred that the provider's maps are erroneous nor claimed that downloading them proves the reviewer's upstream concern is resolved.

**Location:** {LOCATION_EDITOR}

'''+s[b:]
a=s.index('**Partly addressed; historical fALFF verification open.**');b=s.index('### 8.',a)
s=s[:a]+'''**Response.** The revised Methods explicitly state that fALFF maps were calculated upstream by PCP/C-PAC and downloaded, not calculated from the FC time series by us. Regional means were extracted locally from those maps. The provider's configuration specifies 0.01–0.1 Hz fALFF bounds; the exact historical denominator and input filtering order have not been independently verified. We have removed unsupported claims about a local Butterworth calculation and a universal 0.25-Hz denominator, and state the upstream documentation limitation rather than supply an unverified confirmation of unfiltered residuals.

We apply no additional temporal filter or fixed-length truncation to the downloaded FC series; supplied lengths are 116–296 volumes. TR is participant/site-specific, including 17 eligible scans with TR of 3 s. The downloaded maps are used as provided, without reconstructing a fixed-TR denominator. PCP documents 24 motion terms (six parameters, their lags, and squares of those 12 terms), five white-matter/CSF components, and linear/quadratic trends, with global signal added only in the GSR branch. The same participants, nodes, features and inferential rules are used for GSR and no-GSR; both are reported because GSR changes signed correlations and can affect ASD group comparisons. We do not describe GSR as a biologically superior reference.

'''+s[b:]
a=s.index('**Partly addressed; not yet closed.**');b=s.index('### 2. Relative median',a)
s=s[:a]+'''**Response.** We agree that filtering a time series before constructing a full-spectrum fALFF denominator can change the measure, and we have corrected the Methods to avoid attributing that computation to our revised workflow. We used publicly distributed PCP C-PAC voxelwise fALFF maps for ABIDE I. We did not compute their voxelwise fALFF numerator or denominator, and we did not generate the revised maps from the downloaded band-pass-filtered FC time series. Our local operation was extraction of regional map values within the specified parcels and functional masks. The spectral ratio identified in the historical notebook audit is not used in the revised features.

The available provider configuration lists fALFF bounds of 0.01–0.1 Hz. We have replaced the unsupported 0.01–0.08-Hz and fixed 0–0.25-Hz description with this documented information and an explicit attribution to PCP/C-PAC. However, the available release records do not independently confirm the precise historical denominator or input filtering order; we say so directly. A definitive implementation-level answer would require additional provider records or clarification. We therefore address the reporting error and identify the remaining source-documentation limitation, without claiming that the upstream computation has been independently verified. The Discussion restricts physiological interpretation to the relative alignment statistic based on these distributed maps.

'''+s[b:]
a=s.index('**Updated source and results supplied.**');b=s.index('### Sex and generalizability',a)
s=s[:a]+'''**Response.** We supply the exact revised code and results with the submission: S2 Data contains complete feature matrices, analysis covariates, clinical outcomes, statistical results, out-of-fold predictions, selection records and provenance; S3 Code contains executable revision source, analysis specifications, software information and execution instructions. The accompanying README distinguishes the original notebooks from the current analysis and documents the limits of upstream fALFF information. The revised Data and code availability statement points to these supporting archives and the original ABIDE/PCP sources rather than relying on the older GitHub README. We do not claim to have updated that externally hosted page; the versioned files supplied with this revision define the analysis reported here.

'''+s[b:]
s=s.replace('states the remaining fALFF limitation','documents the scope of upstream fALFF information')
s=s[:s.index('## Release status for the authors')].rstrip()+'\n'
# Separate journal requirements into individual paragraphs, not one long run-on list.
s=re.sub(r'\n([2-6]\. \*\*)',r'\n\n\1',s)
p.write_text(s)
print('Finalized source wording; numerical analyses unchanged.')
