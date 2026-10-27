# Revision checklist

Manuscript: **Coevolutionary balance of resting-state brain networks in autism**  
Submission: **PONE-D-26-34219**  
Decision letter: `Comments.pdf`, dated October 19, 2026  
Revision deadline in that letter: **December 18, 2026, 11:59 PM** (timezone unspecified).

## Current status — October 26, 2026

The Stage 2 statistical repair is complete, including all six ComBat sensitivities
and full-cohort classification. See [verified results and remaining actions](stage2_verified/RESULTS_AND_NEXT_STEPS.md).
The original audit below is retained as a historical requirements inventory; its
initial missing-input statements and unchecked boxes are not a current execution
status. The [replacement manuscript and response drafts](revised_paper/README.md) are now
prepared, with site diagnostics and separate supporting information. Historical
fALFF provenance, final author review, public deposit, and final marked submission
files remain open. The main draft now explicitly reports all 71 male controls
against their null configurations and includes eight main figures; S1 contains
ten figures, including feature-selection ROC and heatmaps. All five classifier
algorithm types are complete; the separate selection-method follow-up remains
exploratory and does not establish a reliable gain. The complete original-to-revision analysis map and original/revised
significance lists are in [the continuity audit](stage2_verified/continuity/CONTINUITY_AND_SIGNIFICANT_COMPARISONS.md). Ethics wording explains the original ABIDE approvals and consent
and the secondary use of public deidentified data.

## Assessment and scope

**Original-code audit now available:** see
[the analysis audit](original_code_audit/ORIGINAL_ANALYSIS_AUDIT.md). The supplied
notebooks show pooled raw-energy trimming before ComBat, local fixed-TR spectral
ratio calculations that ignore downloaded fALFF maps, ordinal ROI-to-network
assignments, and missing explicit protected-covariate arguments to ComBat. These
findings supersede provisional descriptions below and must be resolved before
reusing existing features or conclusions. The notebook outputs also trace the
182→167/168→164/165 cohort flow and document the raw fallback directly.

**Acquisition clarification:** the author subsequently confirmed that time series,
preprocessing, and fALFF were supplied as ABIDE derivatives, not computed locally.
The downloader is now in `Codes/`. See [data acquisition notes](DATA_ACQUISITION_NOTES.md)
for the evidence and revised acquisition plan; preprocessing checks below must
distinguish upstream PCP processing from subsequent local analysis.

This revision requires analysis verification before editorial polishing. I reviewed `manuscript.tex`, the editor/reviewer letter, the CSV schemas and results in both `Data/` pipelines, and selected embedded figure PDFs. The manuscript and supplied outputs represent inconsistent analysis states. Neither should be treated as the authoritative final result without recovering the generating code and provenance.

The supplied data contain participant-level derived features and analysis summaries, not imaging data, regional time series, regional fALFF, FC matrices, preprocessing configurations, or executable analysis code. Age, framewise displacement (FD), TR, and participant-level clinical scores are absent from the supplied feature tables. Several requested checks therefore require additional original research materials. No scientific analyses or manuscript text have been changed during this assessment.

**Priority:** P0 = resolve before trusting results; P1 = analysis/reviewer response; P2 = presentation/submission. All boxes are initially unchecked; some journal requirements already appear satisfied but need final verification.

## 1. P0 — Reconcile the data and analysis versions

| Verified discrepancy | Evidence in this checkout | Required resolution |
|---|---|---|
| Exported “ComBat” features equal raw features | Matching by `Subject`, all 92 network features have absolute difference **0**, in both pipelines: 15,364 GSR and 15,456 no-GSR feature cells | Recover harmonization code/logs and establish whether exports were mislabeled, overwritten, or produced by a fallback. These files do not demonstrate successful harmonization. |
| Cohort differs from manuscript | GSR feature tables: **80 ASD/87 Control**; no-GSR: **80/88**. Manuscript: **78/86** and **78/87** | Produce participant manifests for every processing stage and rerun from a verified cohort. |
| Excluded sites remain in exports | Both “ComBat” tables contain **13 sites**, including CMU (1 ASD) and UCLA_1 (1 ASD/1 Control) | Reconcile initial 14-site claim, exported 13 sites, and final 11-site claim. Removing these two sites from these exports happens to yield the reported group counts, but does not establish that harmonization or other processing occurred. |
| Results contradict the universal-null claim | GSR `statistical_comparison_GSR.csv` reports three significant inter-network comparisons: SomatoMotor–Frontoparietal **q=.00434**, DorsalAttn–Salience/VentAttn **q=.03263**, SomatoMotor–Salience/VentAttn **q=.03263** | Recompute after the verified pipeline; do not adopt these as corrected findings or retain the manuscript's null conclusion by assumption. |
| Whole-brain results differ | GSR energy export: ASD **−142.65**, Control **−119.99**, p=.0935, q=.2494; manuscript: −139.7/−118.8, p=.048, q=.19. No-GSR bipolarity export: p=.00786, q=.06285; manuscript: .026/.21 | Replace all reported numbers from one reproducible final run. |
| Statistical-test counts differ | GSR normality export: **52 Mann–Whitney, 37 t, 3 Welch**; no-GSR: **80/10/2**. Manuscript presents 80/10/2 as the general result | Report pipeline-specific counts and verify each decision against its actual normality tests. |
| Model outputs differ | GSR CSV best test accuracy .588; no-GSR .529. Manuscript reports .647 and .588. Both feature-importance files contain **92** features | Reconcile feature sets, model selection, metrics, and figures. |
| Clinical sample sizes differ | CSV score-specific ranges: GSR **31–61**, no-GSR **32–62**; main text **29–59**; supplementary tables still include n=81 and eight scores | Rebuild clinical tables using the final participant manifest and missing-data rules. |

- [ ] **Freeze the baseline**: preserve the submitted version, all current exports, original scripts/configurations, random seeds, package versions, and run logs. Assign a distinct output directory to the revision. **Done when:** every final table and figure traces to a named run and participant manifest. [R1.1–3, R1.10; R2.3]
- [ ] **Recover missing inputs**: original time series or imaging/preprocessing outputs, per-ROI fALFF, FC matrices, subject metadata (age, IQ, FD, TR, scan length), clinical scores, QC/exclusion records, null replicates, and analysis scripts. Record what cannot be recovered. **Done when:** each requested analysis has identified inputs or an explicit limitation. [Editor; R1.2–7,10; R2.1]
- [ ] **Validate harmonization explicitly**: document implementation, batch labels, protected covariates, missing-IQ handling, exclusions, and fit/apply order. `IQ`/`FIQ` entries include missing/sentinel values (4 GSR, 5 no-GSR). Check numerical changes and residual site structure; fail visibly rather than falling back to raw features. **Done when:** valid exports and diagnostics agree with the Methods. [R1.2–3,10]

## 2. P0 — Resolve fALFF and preprocessing validity

- [ ] **Establish the actual fALFF input** from code/configurations, not recollection. The current Methods describe band-pass filtering before fALFF and mix amplitude and power in its definition. If fALFF actually used unfiltered nuisance residuals, correct the description and provide evidence. If it used filtered signals, recompute fALFF and every dependent feature/analysis in both pipelines; recompute FC too if its preprocessing changes. **Done when:** a reproducible preprocessing diagram and code identify separate fALFF and FC branches. [Editor; R1.7; R2.1]
- [ ] **Specify the spectral calculation**: amplitude versus power, summation/normalization, frequency bins, DC treatment, detrending, FFT settings, and voxel-to-ROI aggregation. State the actual sampling-dependent denominator: Nyquist is 1/(2TR), so 0.25 Hz cannot be the acquired upper limit for TR≥2.5 s. Document site TRs, any resampling, and frequency handling. [R1.7; R2.1]
- [ ] **Document both nuisance models exactly**: list all motion terms making up the claimed 24 regressors, tissue signals, CompCor component counts/selection, trends, GSR, filtering order, censoring, and discarded initial volumes. State the truncation length in volumes and duration by TR. Explain why GSR is primary; determine whether the pipelines differ only in GSR or also in tissue regression. [R1.7]

The original [fALFF methods paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3902859/) describes computing the spectral ratio without band-pass filtering, using summed amplitudes across the low-frequency and full sampled bands. This supports treating this check as a prerequisite to reanalysis.

## 3. P1 — Cohort, site confounding, motion, and trimming

- [ ] Create a participant-flow diagram showing initial selection, QC exclusions by reason, small-site exclusions, harmonization, trimming, and final analysis sizes separately for both pipelines. Resolve Table 1's 90/92 and t-test df=89/91 against the stated 78/86. Clarify why incomplete clinical data are called an exclusion when score-wise missingness is later allowed. [R1.2]
- [ ] Add site-by-diagnosis counts and per-site/group raw and harmonized H means; include age, IQ, mean FD, and TR summaries. Flag single-group sites, including CALTECH in the supplied tables. [R1.2–3]
- [ ] Fit the requested complementary **unharmonized H model with diagnosis, site, FD, age, and IQ**; report coefficients, uncertainty, diagnostics, and identifiability limitations. Add a sensitivity analysis limited to sites with both groups if feasible. Revise the categorical claim that protecting diagnosis guarantees preservation of genuine biology or proves site caused the original effect. [R1.2]
- [ ] Report **untrimmed results** alongside the original group-specific energy-trimmed analysis. Describe trimming stage and thresholds, and compare an identical participant subset across GSR/no-GSR. Avoid diagnosis-dependent exclusion in predictive evaluation. [R1.3]
- [ ] Add FD group summaries and motion-adjusted sensitivity analyses; document retained scan lengths/censoring. **Done when:** the conclusions can be compared across untrimmed, adjusted, and matched-pipeline analyses. [R1.3,7]

## 4. P1 — Predictive and multivariate analyses

- [ ] Rebuild classification with learned transformations inside training folds: missing-value handling, harmonization, scaling, feature selection, and tuning. **Do not use held-out diagnosis to harmonize test subjects**; specify how the trained transformation handles held-out batches or use an alternative validation design. Remove “leakage-safe” until verified. Fit feature selection within each tuning fold, not once before CV. [R1.3]
- [ ] Use the same documented candidate feature set in both pipelines and justify whether efficiency is included in univariate, PLSR, PLSC, and classification analyses. Publish the feature list and actual selected count. [R2.3]
- [ ] Distinguish inner tuning scores from outer/held-out performance; describe any site-aware validation and its feasibility. Report class baseline, balanced accuracy, AUC, F1 definition, uncertainty, model-selection rule, and prediction files with participant IDs. Do not infer “chance” solely from nonsignificant univariate tests. [R1.1,3; R2.3]
- [ ] Audit PLS scaling, continuous-outcome fold construction (explain the claimed “stratified” regression CV), and component selection. Recover/reproduce the claimed PLSC 5,000-permutation results: no PLSC outputs are supplied. Specify clinical blocks, missingness, permutation constraints, and inference. Negative predictive R² alone does not invalidate descriptive latent components. [R1.10; R2.3; additional audit finding]

Training-only fitting is supported by the official [scikit-learn leakage guidance](https://scikit-learn.org/1.7/common_pitfalls.html). Diagnosis-protected descriptive harmonization and deployable classification need separate designs.

## 5. P1 — Energy–efficiency interpretation and null models

- [ ] Add H–efficiency partial correlations controlling for **whole-brain mean FC and FD**, with sample sizes, confidence intervals, and multiplicity handling. Whole-brain mean FC is not directly provided in these feature tables; recover FC matrices or a correctly weighted whole-brain summary. Test the ASD–TD difference in association if retaining a group-specific claim. [R1.4]
- [ ] Explain mathematical coupling. For fixed states, shifting all weights by m gives ΔH = −m[(Σs)²−N]/2; with exactly balanced states and N=200 this is **+100m**. Account for median ties rather than assuming exactly 100/100 states. The sign of an empirical H–efficiency correlation is not established by this identity alone. [R1.4]
- [ ] Remove or qualify “robust,” “ASD-specific,” causal “cost,” “trade-off,” and independent-validation claims throughout Abstract, Results, Discussion, and captions. Current Results make stronger claims than Discussion. [R1.4]
- [ ] State exactly what the null randomizes and preserves: node labels, edge signs, signed weights, magnitudes, topology, and spatial structure. Resolve inconsistent descriptions and justify the Maslov–Sneppen analogy. [R1.5]
- [ ] Report participant-level null z-scores, null SDs, empirical tail probabilities, number of permutations, seeds, and group summaries; handle zero null SD explicitly. Assess a spatially constrained null preserving relevant spatial organization **if feasible**; otherwise explain unavailable inputs and the interpretation limit. [R1.5]

## 6. P1 — Statistical uncertainty and scientific framing

- [ ] Add effect sizes with **95% CIs** for the key group contrasts and correlations, choosing measures appropriate to each estimand/test. Publish full feature-level statistics rather than offering them only on request. [R1.6]
- [ ] Recalculate sensitivity power using the verified sample and stated test/alpha; contextualize the reviewer's approximate d=.44 rather than treating it as a computed result for every FDR-adjusted analysis. Explain the optimistic original d=.5 assumption. Add an equivalence test with justified bounds **or** a Bayes factor with specified priors; explain inconclusive evidence where appropriate. [R1.6]
- [ ] Replace physiological “high/low neuronal activity” implications with **above/below participant-median fALFF states**. Explain that fALFF is a relative spectral measure, identical values can get opposite labels across participants, and median splitting discards global differences. Correct the Discussion's description of favorable links: positive links favor same-state pairs, including low–low pairs. [R2.2]
- [ ] Expand sex/generalizability limitations beyond one sentence: explicitly identify the male-only abstract scope; discuss underrepresentation, later recognition of autistic women, possible sex effects on FC/fALFF and the energy framework, and why neither null effects nor within-ASD associations generalize automatically. Propose inclusive recruitment and adequately powered sex-by-diagnosis analyses. [R1: Sex and generalizability]
- [ ] Review and integrate relevant literature on adult ASD fALFF; prior ABIDE structural-balance studies including the authors' younger/no-GSR cohort; Ising/energy-landscape studies; signed-network methods and spatial nulls; small multisite ASD effects; and harmonization under group/site imbalance. Distinguish this static Hamiltonian from measured dynamics. Evaluate relevance rather than adding citations mechanically. [R1.9; Journal requirement 6]
- [ ] After final reanalysis, rewrite the Abstract, Results, Discussion, and conclusion to reflect the actual evidence. Remove stale “encouraging” ML wording and unsupported certainty that harmonization isolated site effects. Correct grammar, repeated paragraphs, and statements calling p=.058/.059 nominally significant. [R1.8; additional audit findings]

## 7. P2 — Rebuild exhibits and public materials

- [ ] Regenerate every affected exhibit from the final run, especially **Figs 2,4,6,7,8,S2; Tables S5–S6**. Resolve Fig 2's two actual panels versus five described in its caption, Fig 3/Table 3 feature mismatch, and Fig S4/Table 10 accuracies. Maintain an old-to-new numbering map for the response. [R1.1]
- [ ] Audit figure titles as well as filenames: `Images/model_comparison_GSR.pdf` currently says **“Model Comparison (No GSR)”**. Selected whole-brain/inter-network/model PDFs in `Images/` are byte-identical to their `Data/` counterparts, so simply copying those exports will not resolve the analysis mismatch. [R1.1,10]
- [ ] Rebuild supplementary tables: resolve 92×8 incorrectly labeled as 460, 90 features/eight scores versus the actual 92/five design, obsolete clinical sample sizes, and missing `significant_uncorrected.csv`. Include actual full rankings/results instead of ellipses or promises of online files. [R1.1,10; R2.3]
- [ ] Update the shared code/data repository README, methods, counts, examples, and data dictionary; supply executable scripts/configurations and a versioned release. Identify public ABIDE access instructions and any restrictions accurately. No README or public repository URL was found in the supplied manuscript/data, so the posted version remains unverified. [R1.10]

## 8. P2 — Journal requirements and submission package

- [ ] Check journal formatting, figure specifications, and file naming against the journal's requirements at submission. [Journal 1; figure instructions]
- [ ] Keep the ethics statement **only in Methods**. It is already there in this source; verify the final clean and marked manuscripts. Confirm the institutional exemption wording is accurate. [Journal 2]
- [ ] Add/verify explicit narrative callouts for **Figs 3,4,6,8 and Table 12**, using stable LaTeX labels and checking the final compiled numbers. [Journal 3–4]
- [ ] Extract supplementary figures/tables from the main manuscript into files uploaded as **Supporting Information**; retain a legend for each file after the references and repair cross-references. [Journal 5]
- [ ] Create a point-by-point **Response to Reviewers** covering Journal 1–6, the editor's preprocessing concern, R1.1–10, R1 sex/generalizability, and R2.1–3. For each give the action, result, final page/line or exhibit, and reason for any declined/infeasible analysis. [Decision letter]
- [ ] Produce both **Revised Manuscript with Track Changes** and clean **Manuscript**, with regenerated figures and complete supporting files. Preserve the submitted baseline for a reliable LaTeX diff. [Decision letter]
- [ ] Compile the final LaTeX/BibTeX package and visually check tables, equations, captions, citations, figure callouts, and supplement references. Remove the duplicate trailing `\end{document}`. Reconcile all counts/numbers with final CSVs. [Additional audit finding]
- [ ] Verify submission metadata and data/code availability links; update the cover letter if financial disclosure changes. Coauthors should verify their own ORCIDs if desired. Check the submission portal for any additional reviewer attachments before finalizing the response. [Decision letter]

## Recommended execution order and completion gate

1. Recover provenance and inputs; resolve fALFF validity and the identical raw/ComBat exports.
2. Lock cohort definitions, preprocessing, harmonization, covariates, and validation design.
3. Run the main analyses and requested sensitivity checks; document infeasible requests.
4. Regenerate results, figures, supplements, code/data release, and manuscript narrative together.
5. Complete the point-by-point response, clean/marked manuscripts, and submission checks.

**Ready for revision submission when:** every editorial/reviewer point has a documented response; every numerical claim traces to reproducible final outputs; no exhibit contradicts the text; methodological limitations are explicit; and the submission package is complete. This checklist is a work plan, not a claim that the current results are verified or that acceptance is assured.
