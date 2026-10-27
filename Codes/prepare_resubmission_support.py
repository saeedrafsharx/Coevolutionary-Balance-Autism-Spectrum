from pathlib import Path
import re,shutil,zipfile,hashlib,json,csv
import pandas as pd
R=Path('revision/resubmission'); O=Path('revision/revised_paper')
# Preserve detailed verified supporting narrative and tables; replace all old plot floats.
s=(O/'supporting_information.tex').read_text()
s=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '',s,flags=re.S)
s=s[:s.index('\\section*{Additional corrected figures}')]
s=s.replace('Working revision draft, October 26, 2026','Revision, October 27, 2026')
s=s.replace('It does not\nresolve historical fALFF provenance or replace review of the final data-sharing\nstatement. S2 Supporting Information contains the full machine-readable\nresults;', 'Historical fALFF provenance remains limited as described in the main Methods.\nS2 Data contains the full machine-readable results;')
s=s.replace('S1 Supporting Information','S1 Appendix').replace('S2 Supporting Information','S2 Data')
for match in re.findall(r'\\input\{tables/([^}]+)\}',s):
 shutil.copy(O/'tables'/match,R/'tables'/match)
# Remove stale old-figure cross-references, which belonged to replaced supplement.
s=re.sub(r'Fig\.~\\ref\{[^}]+\}', 'the supporting figures below',s)
s=re.sub(r'Figs\.~\\ref\{[^}]+\}.*?\\ref\{[^}]+\}', 'the supporting figures below',s)
s+='\n\\clearpage\n\\section*{Supporting figures}\n'
caps=[('Composition of the primary node set','The common 106-node graph contains 91 cortical-assigned and 15 unassigned nodes. The seven network counts are shown. A single limbic node cannot support within-network edges.'),('Raw H estimates across node-set designs','Points show adjusted diagnosis coefficients and HC3 95\\% intervals. Node and participant sets differ across designs; these comparisons cannot isolate a node-count effect.'),('Individual male-control empirical and null energies','Each line connects the empirical H of one of 71 controls to that participant\'s mean of 1,000 randomizations. Pipelines and null constraints are separated. No topology rewiring is performed.'),('Participant-level null z-scores','Both groups are shown under unrestricted and blocked state permutations. Negative z denotes H below the individual null mean. These are alignment comparisons, not diagnosis tests.'),('Within-network block-energy distributions','All six available within-network energy blocks are shown in both pipelines. ASD is magenta and controls dark green. Descriptive distributions use all 140 eligible participants, whereas adjusted comparisons use 136.'),('Baseline classifier confusion matrices','Each matrix contains all 140 complete out-of-fold predictions. Counts and true-class-normalized percentages are shown for every model and pipeline. No model is selected from outer-fold performance.'),('Descriptive principal-component projection','PCA uses the standardized raw common 88-feature matrix and all 140 participants separately per pipeline. ASD is magenta and controls dark green. This full-sample descriptive projection is not used in cross-validation, feature selection or inferential testing. Projection coordinates are in S2 Data.'),('Exploratory feature-selection performance comparison','Balanced accuracy and ROC AUC are shown for every baseline and follow-up model/pipeline pair. The follow-up was specified after baseline outer performance was known and is exploratory.'),('Exploratory feature-selection ROC curves','All models use complete out-of-fold scores and the same known outer site folds. A better point estimate does not constitute independent confirmation of improvement.'),('Exploratory feature-selection confusion matrices','Counts and row-normalized percentages use all 140 out-of-fold predictions per model/pipeline pair. Selection and fitting occur inside training folds.'),('Baseline feature-selection frequencies','The 25 most frequently retained features per pipeline are shown across five overlapping outer training sets for each model. Frequencies describe selection, not significance, coefficient magnitude or causal relevance. Full selected names are in S2 Data.'),('Follow-up feature-selection frequencies','The top 25 features are ranked by total retained frequency across model fits. Training-only redundancy pruning precedes the chosen selector. Overlapping folds and multiple procedures preclude interpreting frequencies as independent biomarker evidence.'),('PLSC latent covariance summaries','Squared singular-value fractions and corrected rank-wise permutation probabilities are shown for ADOS and five-score complete-case blocks. Sample sizes are 43 and 21. No latent-variable test survives BH across the 14 tests; covariance fractions are not out-of-sample predictive accuracy.')]
for i,(title,cap) in enumerate(caps,1):
 s+=f'\nFigure~\\ref{{sfig:{i}}} presents {title.lower()}.\n\\begin{{figure}}[p]\n\\centering\n\\includegraphics[width=.98\\textwidth,height=.72\\textheight,keepaspectratio]{{figures/SFig{i}.pdf}}\n\\caption{{\\textbf{{{title}.}} {cap}}}\\label{{sfig:{i}}}\n\\end{{figure}}\n\\clearpage\n'
s+='\\end{document}\n'
(R/'supporting_information.tex').write_text(s)
# Data archive: retain complete audited tables, add the actual minimal score table.
with zipfile.ZipFile(O/'S2_supporting_tables_DRAFT.zip') as z: files={n:z.read(n) for n in z.namelist() if not n.endswith('/') and not n.lower().endswith('index.csv')}
m=pd.read_csv('results/abide_stage2_repaired/audit/demographic_manifest.csv')
cols=['SUB_ID','FILE_ID','SITE_ID','DX_GROUP','AGE_AT_SCAN','SEX','FIQ','func_mean_fd']+[c for c in m if c in ['ADI_R_SOCIAL_TOTAL_A','ADI_R_VERBAL_TOTAL_BV','ADI_R_ONSET_TOTAL_D','ADOS_TOTAL','ADOS_COMM']]
files['features/analysis_covariates_and_clinical_scores.csv']=m[cols].to_csv(index=False).encode()
for p in (R/'source_support').glob('*'):files['figure_source/'+p.name]=p.read_bytes()
files['README.md']=b'''# S2 Data
Complete revision results and analysis tables. No original MRI images are redistributed.
features/ contains the common raw feature table and analysis covariates/clinical scores.
statistics/ contains final group, clinical, null and prediction results.
harmonized_features/ contains checked ComBat sensitivity features.
continuity/ preserves original-export reproduction separately from revision findings.
ml_feature_selection/ contains the explicitly exploratory post-baseline extension,
including all inner candidates, selected names and aligned out-of-fold predictions.
control_nulls/ contains the separate healthy-male baseline (71 controls).
provenance/, diagnostics/, frozen_design/ document inputs, scope and model checks.
figure_source/ contains additional descriptive projection coordinates.
Control-only q values and all-participant q values have different declared families;
the revised main text uses the all-560 participant/pipeline/null family.
S3 Code supplies execution instructions. The main Methods describe unresolved
historical fALFF denominator/filtering provenance; completeness is not validation
of that historical branch. INDEX_SHA256.csv records every payload file.
'''
index='file,bytes,sha256\n'+''.join(f'{n},{len(b)},{hashlib.sha256(b).hexdigest()}\n' for n,b in sorted(files.items()))
files['INDEX_SHA256.csv']=index.encode()
with zipfile.ZipFile(R/'S2_Data.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n,b in sorted(files.items()):z.writestr(n,b)
# Self-contained source tree; original notebooks explicit historical audit only.
code={}
for p in Path('Codes').iterdir():
 if p.is_file() and p.suffix in {'.py','.sh','.ipynb','.md','.txt'}:code['Codes/'+p.name]=p.read_bytes()
for p in Path('revision/analysis_plan').glob('*'):
 if p.is_file():code['revision/analysis_plan/'+p.name]=p.read_bytes()
for p in R.glob('*.tex'):code['revision/resubmission/'+p.name]=p.read_bytes()
code['README.md']=b'''# S3 Code: revision analysis
The Codes directory contains executable analysis, repair, audit and reporting scripts.
Original acquisition/analysis notebooks are historical audit records, not instructions
to rerun the original preprocessing assumptions as the revised analysis.

Execution order: Stage 1 and Stage 1b QC -> frozen revision/analysis_plan definitions
-> abide_stage2_run.py / associated Stage 2 notebook -> repair_stage2_results.py
(if recovering the documented interrupted statistics run) -> complete_original_classifiers.py
-> run_ml_feature_selection.py -> report_ml_feature_selection.py.
Read each script's --help or top configuration before execution; input roots and output
roots are explicit. Stage 1 needs the ABIDE/PCP derivative dataset and metadata patch.
The two author Kaggle inputs are abide-full-derivatives-dataset/abide_revision and
abide-metadata-patch/abide_metadata_patch_local_v2.
MRI preprocessing and voxelwise fALFF computation are upstream, not performed here.
The historical fALFF denominator/filter order is not independently verified.

Original-algorithm classification is the baseline; expanded feature selection is a
separate post-baseline exploratory follow-up and must not overwrite it.
Rebuild the final main figures using build_resubmission_exhibits.py and manuscript
using assemble_resubmission_manuscript.py from the repository root with the verified
results directories available. Relative source templates are retained in the full
repository; S3 alone does not contain MRI images or the historical manuscript.
S2 supplies complete statistical inputs/outputs for audit, not the source imaging.

Software used: Python 3, NumPy 1.26.4, pandas 2.1.4, SciPy 1.11.4,
statsmodels 0.15.0, neuroCombat 0.2.12, scikit-learn 1.9.1,
XGBoost CPU 3.4.1, matplotlib 3.6.3. See machine-readable audit version records in S2.
Figure builds use Arial (installed locally); LaTeX uses pdflatex/bibtex and plos2025.bst.
'''
with zipfile.ZipFile(R/'S3_Code.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n,b in sorted(code.items()):z.writestr(n,b)
print('Packaged',len(files),'data files and',len(code),'code/source files; clinical columns',cols)
