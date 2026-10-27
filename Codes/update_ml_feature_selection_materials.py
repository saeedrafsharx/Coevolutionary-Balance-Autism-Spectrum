"""Add explicitly exploratory selection follow-up without replacing baseline ML."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'revision/revised_paper'
SRC=ROOT/'results/abide_ml_feature_selection_review'

def region(text,tag,body,anchor):
 begin=f'% {tag}_BEGIN';end=f'% {tag}_END';replacement=begin+'\n'+body.strip()+'\n'+end+'\n'
 if begin in text:return re.sub(re.escape(begin)+r'.*?'+re.escape(end)+'\n?',lambda _:replacement,text,count=1,flags=re.S)
 if anchor not in text:raise ValueError(anchor)
 return text.replace(anchor,replacement+'\n'+anchor,1)

def main():
 subprocess.run([sys.executable,str(ROOT/'Codes/report_ml_feature_selection.py')],cwd=ROOT,check=True)
 for source,target in [('feature_selection_ROC','ml_selection_ROC'),('feature_selection_confusion_heatmaps','ml_selection_confusion'),('performance_heatmaps','ml_selection_performance'),('feature_selection_selection_frequency','ml_selection_frequency'),('baseline_selection_frequency','ml_baseline_selection_frequency')]:
  for extension in ['pdf','png']:shutil.copy2(SRC/f'figures/{source}.{extension}',PAPER/f'figures/{target}.{extension}')
 p=PAPER/'manuscript_revision.tex';s=p.read_text()
 results=r'''\paragraph*{Feature-selection audit and exploratory extension.}
Feature selection was already nested in the five-model analysis above. Reconstructing
all 50 outer-training fits reproduces every saved prediction: ANOVA retained 22
features in 11 fits, 44 in 17, and all 88 in 22. We subsequently compared broader
selection methods in a separate exploratory analysis, retaining the same known
outer site folds rather than treating them as a fresh independent test.
Training-fold near-exact redundancy pruning left 59 distinct columns, and inner
CV chose all 59 in seven fits, ANOVA top-K in 16, mutual-information top-K in 17,
and L1-ranked top-K in ten. Additional selection retained 5, 10 or 20 features.

Four of the ten model/pipeline point estimates increased and six decreased
(Table~\ref{tab:FScomparison}). No-GSR logistic regression changed from .485 to
.544, no-GSR SVM from .485 to .536, and no-GSR KNN from .542 to .557. GSR XGBoost
changed from .457 to .492. Every paired conditional interval for change includes
zero, and all follow-up balanced-accuracy intervals include .5. The follow-up
therefore does not establish a reliable performance gain or justify selecting a
new final classifier from the outer results. Full ROC, confusion, performance
and selected-feature frequency heatmaps are in S1 Supporting Information; all
feature names, inner candidates, and test predictions are in S2.

\input{tables/feature_selection_comparison.tex}'''
 s=region(s,'ML_SELECTION_RESULTS',results,'\\subsection*{Sensitivity analysis: results without global signal regression}')
 methods=r'''\paragraph*{Exploratory comparison of feature-selection methods.}
After inspecting the baseline outer results, we specified a separate follow-up
using the same 88-feature inputs, participants, and five/three outer/inner site
folds. Imputation, variance filtering, redundancy pruning, scaling, selection,
and classifier fitting are estimated only from the corresponding training data
\cite{scikit-learn}. Redundancy pruning retains the first feature in fixed schema
order when absolute training correlation is at least $1-10^{-10}$; proportional
EnergyBlock/AvgFalffW terms and complementary proportions motivate this step.
The graph, participants and primary group tests are unchanged.

Inner CV jointly chooses all remaining features, or ANOVA, mutual information,
or L1-logistic coefficient-ranked top 5/10/20 features, with K capped at the
available count. Mutual information uses three neighbors and seed 101. The L1
ranker uses balanced liblinear logistic regression, C=1 and seed 101; coefficient
ties are broken by training ANOVA rank and fixed feature order. Zero-coefficient
selections are explicitly recorded; none occurred in the final outer fits.
Non-XGBoost classifier grids match the baseline. The compact follow-up XGBoost
grid uses depth 2/3, estimators 100/200 and learning rates .05/.1, with the same
fixed CPU histogram settings. Thus changes cannot be attributed to selector type
alone. All candidates are selected by inner balanced accuracy, and all ten
model/pipeline results are reported without choosing an outer-test winner.

Paired 1,000-replicate site bootstrap (seed 303) estimates conditional intervals
for the balanced-accuracy change on aligned OOF predictions. These intervals
omit full training uncertainty and do not adjust for repeated exploration.
Because baseline outer results had already been inspected, this follow-up is
exploratory rather than independent confirmation of improved generalization.
External evaluation would be needed to substantiate a predictive gain.'''
 s=region(s,'ML_SELECTION_METHODS',methods,'\\section*{Data and code availability}')
 sentence='A broader exploratory selection-method search did not establish a reliable performance gain.'
 if sentence not in s:
  prefix,rest=s.split('\\section*{Introduction}',1)
  prefix=prefix.replace('all conditional confidence intervals including 0.5.','all conditional confidence intervals including 0.5.\n'+sentence,1)
  s=prefix+'\\section*{Introduction}'+rest
 p.write_text(s)
 p=PAPER/'supporting_information.tex';s=p.read_text()
 body=r'''\section*{Feature-selection audit and exploratory extension}
The completed baseline contains all five original algorithm types. All 50 saved
outer-training models were reconstructed using their saved settings without new
hyperparameter search; all predicted labels match. ANOVA selection chose 25\%
(22 features) in 11 fits, 50\% (44 features) in 17, and 100\% in 22. Per-fold
selected feature names and reconstruction score differences are supplied in S2.
The baseline no-reduction option is part of its inner selection search, not an
absence of feature selection implementation.

The follow-up's training-only redundancy pruning reduced 88 to 59 columns in
every outer fit, followed by all-features, ANOVA, mutual-information or L1-ranked
selection chosen jointly with classifier parameters in the inner folds. All
50 follow-up outer fits have complete predictions, with 140 unique participants
for every algorithm/pipeline. Selection counts are 5/10/20 or all 59. The global
and cortical group models and the 59 reported H associations are unchanged.

Fig.~\ref{fig:SFperformance} reports all model performance estimates before/after
the follow-up. Four balanced-accuracy point estimates rise and six fall; every
paired change interval includes zero. The largest positive change is no-GSR
logistic regression, .059 (conditional 95\% interval $-.0148$ to .1088).
The no-GSR KNN follow-up has BA=.557 and descriptive pooled AUC=.606.
These estimates do not establish a generalizable improvement or justify selecting
a model after examining test performance. The feature representation and compact
XGBoost grid also changed, precluding attribution of a difference to selector
method alone. The prespecified follow-up document explicitly records that the
baseline outer results had already been viewed.

Figs.~\ref{fig:SFroc} and \ref{fig:SFconfusion} give follow-up ROC curves and
row-normalized confusion heatmaps with exact counts. Fig.~\ref{fig:SFfrequency}
shows 25 highest mean selected-feature frequencies, while all 88 features and
all fold selections are provided as CSV. Frequencies are descriptive across
five overlapping training folds, not independent stability tests, biomarker
significance, or permutation importance. Corresponding baseline feature-selection
frequency and normalized confusion heatmaps are supplied as standalone figures.
No held-out test observation is used to train a selector or choose its parameters.'''
 s=region(s,'ML_SELECTION_SUPPLEMENT',body,'\\section*{Additional corrected figures}')
 figures=r'''\clearpage
\begin{figure}[htbp]
  \centering
  \includegraphics[width=.96\textwidth]{figures/ml_selection_performance.pdf}
  \caption{All five classifiers before and after the exploratory selection-method
  extension. Left: balanced accuracy; right: pooled OOF AUC. Every cell uses all
  140 participants. Colors and point differences are descriptive; the change
  intervals in the main table all include zero. These known outer folds do not
  provide independent confirmation of improvement.}\label{fig:SFperformance}
\end{figure}
\clearpage
\begin{figure}[htbp]
  \centering
  \includegraphics[width=.96\textwidth]{figures/ml_selection_ROC.pdf}
  \caption{ROC curves for all five follow-up classifiers in both pipelines,
  using complete pooled OOF scores. AUC is descriptive. Feature method, K and
  classifier parameters were chosen within inner site-disjoint folds; the
  already inspected outer folds limit independent interpretation.}\label{fig:SFroc}
\end{figure}
\clearpage
\begin{figure}[htbp]
  \centering
  \includegraphics[width=.96\textwidth,height=.80\textheight,keepaspectratio]{figures/ml_selection_confusion.pdf}
  \caption{Follow-up confusion heatmaps for every classifier/pipeline. Rows are
  observed diagnoses; columns are predicted diagnoses. Cell annotations give
  counts and row percentages. Each panel contains 71 controls and 69 ASD
  participants. No model is selected by these outer-test displays.}\label{fig:SFconfusion}
\end{figure}
\clearpage
\begin{figure}[htbp]
  \centering
  \includegraphics[width=.96\textwidth,height=.78\textheight,keepaspectratio]{figures/ml_selection_frequency.pdf}
  \caption{Selected-feature frequencies in the follow-up. Color is the fraction
  of five outer-training fits selecting a feature for each model/pipeline.
  The 25 highest average frequencies are shown; all 88 are provided in S2.
  Labels use the network and metric abbreviations of the main H-correlation
  figure; LR, logistic regression; GNB, Gaussian naive Bayes; XGB, XGBoost.
  Repeated training samples overlap, and frequency is not statistical feature
  significance, causal importance or independent validation.}\label{fig:SFfrequency}
\end{figure}
'''
 s=region(s,'ML_SELECTION_SUPPLEMENT_FIGURES',figures,'\\end{document}')
 p.write_text(s)
 p=PAPER/'response_to_reviewers.md';s=p.read_text();tag='## Feature selection: additional exploratory analysis'
 if tag in s:s=s[:s.index(tag)].rstrip()+'\n'
 s+='\n'+tag+'''\n\nFeature selection was already nested inside every classifier: ANOVA top\n25/50/100 percent was chosen in inner site folds, and all 50 reconstructed\nouter fits reproduce the saved predictions. Selected features are now exported\nby name. A separate post-baseline exploratory extension adds training-only\nredundancy pruning, mutual information and L1-ranked selection alongside ANOVA\nand the no-reduction option. All methods and classifier parameters remain\ninner-fold choices. All ten results and every inner candidate are provided.\nROC, confusion, performance and selection-frequency heatmaps are included.\nFour point estimates improve and six decline; every paired conditional change\ninterval includes zero. We do not claim independent confirmation of improvement,\nselect an outer-test winner, or replace the original five-model analysis.\n'''
 p.write_text(s)
 p=PAPER/'README.md';s=p.read_text();tag='## Feature-selection extension'
 if tag not in s:s+='\n'+tag+'''\n\nAll five original classifiers are complete. Feature selection was already applied\nwithin training folds; its reconstructed selected feature names and score checks\nare now saved. An additional explicitly exploratory comparison evaluates ANOVA,\nmutual information, L1 ranking and no further selection after training-only\nredundancy pruning. No clear gain is established; baseline results are preserved.\n\nRead [the feature-selection report](../stage2_verified/ml_feature_selection/FEATURE_SELECTION_AND_RESULTS.md).\nPlots (PDF/PNG) are in results/abide_ml_feature_selection_review/figures/.\nThe main paper retains eight figures; S1 now has ten, including follow-up ROC,\nconfusion, performance and feature-selection frequency heatmaps.\n\nTo reproduce this optional follow-up and refresh its materials:\n\n```sh\nMPLCONFIGDIR=/tmp/coev_matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv_stage2/bin/python Codes/run_ml_feature_selection.py\nMPLCONFIGDIR=/tmp/coev_matplotlib .venv_stage2/bin/python Codes/update_ml_feature_selection_materials.py\n```\n\nThe runner reuses complete fold checkpoints. The main five-model pipeline is\nunchanged; follow-up output has its own plan, source hashes, versions and results.\n'''
 p.write_text(s)
 subprocess.run([sys.executable,str(ROOT/'Codes/package_revision_support.py')],cwd=ROOT,check=True)
 for name in ['manuscript_revision','supporting_information']:
  with (PAPER/f'{name}_build.txt').open('w') as handle:
   subprocess.run(['latexmk','-cd','-pdf','-interaction=nonstopmode','-halt-on-error',str(PAPER/f'{name}.tex')],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
  log=(PAPER/f'{name}.log').read_text()
  if any(term in log for term in ['Overfull','undefined','Float too large']):raise RuntimeError(f'Inspect {name} layout/references')
 p=PAPER/'draft_validation.json';validation=json.loads(p.read_text())
 for name,old in validation['preserved_source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==old
 validation['feature_selection_followup_source']=str(SRC.relative_to(ROOT))
 validation['baseline_feature_selection_reconstruction_folds']=50
 validation['feature_selection_followup_pairs']=10
 validation['main_figure_count']=8;validation['supplementary_figure_count']=10
 validation['compiled_pdf_sha256']={name:hashlib.sha256((PAPER/name).read_bytes()).hexdigest() for name in ['manuscript_revision.pdf','supporting_information.pdf']}
 p.write_text(json.dumps(validation,indent=2)+'\n')
 print('Feature selection follow-up added; baseline retained; main and S1 PDFs compiled.')

if __name__=='__main__':main()
