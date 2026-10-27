# Stage 2 frozen revision protocol

This supersedes the earlier pending-coverage discussion. `stage2_frozen_design.json`
and the design-specific participant/ROI CSVs fix the actual inputs before corrected
group-effect testing. Decisions used demographic criteria and measurement QC,
not corrected diagnostic p-values. These are revision-stage choices, not an
original preregistration.

## Cohorts and nodes

Primary eligibility is age **18–30 inclusive**, observed **FIQ >80**, without IQ
imputation: **140 existing male participants, 69 ASD/71 controls**. Handedness is
reported as recorded; there is no claim that everyone is right-handed.

| Design | Fixed ROIs | Feature sample | Site-adjusted model sample |
|---|---:|---:|---:|
| Primary common-node network | 106 | 140: 69 ASD/71 controls | 136: 67/69 |
| Broader-node sensitivity | 181 | 104: 48/56 | 101: 47/54 |
| Full-200 sensitivity | 200 | 69: 31/38 | 64: 30/34 |

Every selected ROI must have nonconstant time series and ≥50% finite in-mask fALFF
support in every retained participant, in both pipelines. Primary nodes qualify
across all 140 people. Broader nodes first qualify at 95% prevalence, then people
must meet the rule throughout that fixed set. The full-200 design requires the
same quality throughout all parcels. Exact masks are embedded in the notebook.

The primary graph is a **coverage-limited 106-node network**, not the unchanged
200-ROI whole-brain graph. Medians, nodal states, H, and graph/block features are
rebuilt. The coverage threshold is a transparent methodological choice, not a
universal validity standard. The two larger graphs assess sensitivity to scope.

## Anatomy

Network assignment requires ≥50% cortical Yeo support before maximum Dice assignment.
Other nodes remain unassigned in cortical blocks but remain in global measures.
Only one limbic node remains in the primary graph: within-limbic features are
structurally unavailable, not null effects. The old second-eigenvector bipolarity
descriptor is saved under a legacy name and excluded from inferential families.

## Inference

Primary H uses raw features adjusted for diagnosis, site, age, FIQ, and mean FD,
with HC3 95% CI and Holm correction across GSR/no-GSR. Secondary Whole/Intra/Inter
families use BH across both pipelines within each design/model, excluding primary
H and legacy bipolarity. Sensitivity analyses are exploratory.

Sites with fewer than three design participants are omitted from adjusted models,
with explicit reasons and participant manifests, to avoid singleton leverage and
unstable uncertainty. All 140 still receive primary feature exports. Covariate
design rank is checked explicitly.

ComBat is sensitivity only and explicitly protects group, age, IQ, and FD. Failure
does not substitute raw data under a harmonized label. Primary energy is untrimmed;
pooled 1.5×IQR trimming before small-site filtering is a sensitivity. Conventional
±0.2 pooled-SD equivalence bounds are exploratory, not a clinical SESOI. Sensitivity
power uses a two-sample approximation, not covariate-adjusted power.

Efficiency correlations control mean FC, FD, age, IQ, and site; group interactions
test specificity directly. Nulls retain FC and state counts. Network/hemisphere
blocks preserve coarse organization, not full spatial autocorrelation. Zero null
variance is explicitly unavailable. Permutation counts and seeds are recorded.

## Clinical and predictive complements

Clinical missingness is score-specific. Outputs include correlations, adjusted H
correlations, nested site-held-out PLSR, and exploratory nuisance-residualized PLSC
with within-site permutations. Limited complete cases and residual exchangeability
assumptions remain explicit.

Classification nests imputation, scaling, selection, and tuning within site-held-out
folds; no diagnosis-protected ComBat is used. The upstream node mask was selected
using whole-cohort QC: prediction is **exploratory conditional on that mask**, not
fully prospective validation of acquisition/QC. Bootstrap intervals condition on
fixed OOF predictions and omit some training uncertainty. No final model is selected
by held-out accuracy.

## Execution and review

Run `Codes/abide-revision-stage2.ipynb` on Kaggle with both existing datasets and
defaults. HC3 is checked against its matrix formula before long calculations.
Frozen sample/ROI violations stop inference; later analysis failures are logged.
Return `abide_stage2_review.zip` under `results/`; preserve full outputs and arrays.

Offline tests cover frozen QC masks, design rank, multiple-testing corrections,
partial correlations, power, zero-variance nulls, and notebook synchronization.
Full imaging/statistical execution needs Kaggle; local derivatives and statistical
packages are absent. Results must be reviewed before manuscript conclusions.

Upstream fALFF band/input provenance, the changed anatomical scope, handedness
disclosure, model diagnostics, and sensitivity agreement remain required for the
revision. A completed notebook is not a submission-ready paper.
