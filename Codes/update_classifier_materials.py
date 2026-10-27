"""Refresh paper exhibits and completion wording from actual classifier outputs.

Called after a resumable extension; handles four or all five original model types.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys
import pandas as pd
from classification_output_source import classification_statistics_directory
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'revision/revised_paper'
STATS=classification_statistics_directory()
ALL_MODELS=['LogisticRegression','SVM_RBF','GaussianNB','KNN','XGBoost']
NAMES={'LogisticRegression':'logistic regression','SVM_RBF':'RBF SVM','GaussianNB':'Gaussian naive Bayes','KNN':'KNN','XGBoost':'XGBoost'}

def replace_region(text,tag,replacement,old):
    begin=f'% {tag}_BEGIN';end=f'% {tag}_END'
    if tag.endswith('README'):
        text=text.replace(begin,f'<!-- {tag}_BEGIN -->').replace(end,f'<!-- {tag}_END -->')
        begin=f'<!-- {tag}_BEGIN -->';end=f'<!-- {tag}_END -->'
    block=begin+'\n'+replacement.strip()+'\n'+end
    if begin in text:
        return re.sub(re.escape(begin)+r'.*?'+re.escape(end),lambda _:block,text,count=1,flags=re.S)
    if old not in text:raise ValueError(f'Cannot find original region {tag}')
    return text.replace(old,block,1)

def main():
    table=pd.read_csv(STATS/'classification_conditional_intervals.csv')
    scope=json.loads((STATS/'classification_scope.json').read_text())
    models=[m for m in ALL_MODELS if m in set(table.model)]
    pending=[m for m in ALL_MODELS if m not in models]
    pairs=len(table)
    assert pairs==len(models)*2 and table.n_OOF_predictions.eq(140).all()
    low=float(table.balanced_accuracy.min());high=float(table.balanced_accuracy.max())
    inside=((table.conditional_site_bootstrap_CI95_low<=.5)&(table.conditional_site_bootstrap_CI95_high>=.5))
    ci_text=('Every conditional interval includes .5.' if inside.all() else
             f'{int(inside.sum())} of {pairs} conditional intervals include .5; these are conditional estimation intervals, not a classifier permutation test.')
    for script in ['summarize_stage2_revision.py','build_revision_exhibits.py','restore_revision_figures.py','build_continuity_materials.py']:
        print('Refresh:',script,flush=True)
        subprocess.run([sys.executable,str(ROOT/'Codes'/script)],cwd=ROOT,check=True)
    p=PAPER/'manuscript_revision.tex';text=p.read_text()
    old='''Every classifier/pipeline pair has 140 unique OOF predictions, replacing the
incomplete aggregates in the initial Stage 2 execution. Balanced accuracy ranges
from .485 to .558 (Table~\\ref{tab:classification}, Fig.~\\ref{fig:primary}); every
conditional interval includes .5. These estimates provide no demonstrated
out-of-site diagnostic utility under the specified workflow.'''
    result=f'''The {pairs} completed classifier/pipeline pairs have 140 unique OOF predictions
each, using the same frozen site-disjoint splits. Completed algorithms are
{', '.join(NAMES[m] for m in models)}. Balanced accuracy ranges from
{low:.3f} to {high:.3f} (Table~\\ref{{tab:classification}}, Fig.~\\ref{{fig:primary}}).
{ci_text} No final classifier is selected from outer-fold performance.
The conditional intervals do not include all training or QC-selection uncertainty.'''
    if pending:result+='\nThe remaining original algorithm type, '+', '.join(pending)+', is pending; no performance is inferred for it.'
    text=replace_region(text,'CLASSIFICATION_RESULTS',result,old)
    old='''The original classification analysis compared five algorithms, whereas the
revised site-held-out run currently evaluates logistic regression and RBF SVM.
Gaussian naive Bayes, KNN, and XGBoost have not been rerun under this validation
scheme; the two-model results must not be represented as a five-model replication.
This is a model-scope difference, not evidence that the original three additional
methods were invalid or that their revised performance is known.'''
    status=(f'The revised site-held-out analysis now restores {len(models)} of the five original\nalgorithm types, using the same revision participants, features, folds and scoring.\n'
        +('All five original algorithm types have complete OOF predictions in both pipelines.' if not pending else 'XGBoost remains pending. The completed four-model results are not represented\nas a five-model replication.')
        +'\nMatching the algorithm types does not restore identical original inputs or validation:\nsite-held-out evaluation replaces the original random train/test procedure.')
    text=replace_region(text,'CLASSIFICATION_CONTINUITY',status,old)
    old='''Classification compares balanced logistic regression and RBF SVM on all 140 primary
participants without ComBat. Deterministic size-balanced site-disjoint folds use
five outer and three inner splits, each required to contain both classes.
Median imputation, variance filtering, standardization, ANOVA feature selection
(25/50/100 percent), and hyperparameter tuning are nested. Logistic $C$ values are
0.01/0.1/1/10; SVM $C$ values are 0.1/1/10 with scale-based gamma. Inner scoring
uses balanced accuracy. No model is selected by outer-test accuracy. Pooled OOF
balanced accuracy is accompanied by a 1,000-replicate site bootstrap conditional
on fixed predictions. This omits some training uncertainty. The node mask and
available feature schema use cohort-wide QC, so prediction is exploratory
conditional on this mask, not prospective validation of the entire workflow.'''
    method=f'''Classification compares {', '.join(NAMES[m] for m in models)} on all 140 primary
participants without ComBat. Deterministic size-balanced site-disjoint folds use
five outer and three inner splits, each required to contain both classes.
Median imputation, variance filtering, standardization, ANOVA feature selection
(25/50/100 percent), and hyperparameter tuning are nested. Logistic and SVM
class weights are balanced. Logistic $C$ values are 0.01/0.1/1/10; SVM $C$ values
are 0.1/1/10 with scale-based gamma. Gaussian naive Bayes uses its default
variance smoothing and training-estimated priors, with feature-selection percentile
tuned inside folds. KNN evaluates 3/5/7/9/11 neighbors and uniform/distance
weights. Positive-class probabilities supply ROC scores for these added models;
logistic/SVM decision-function scores and predictions are reused unchanged.'''
    if 'XGBoost' in models:
        method+='''\nXGBoost uses CPU histogram trees, seed 42, one thread, log-loss evaluation,
max depth 2/3/4, 100/200/300 estimators, and learning rates .05/.1. The same
nested feature-selection percentiles apply; default scale-pos-weight is retained.'''
    method+='''\nInner scoring uses balanced accuracy. No model is selected by outer-test accuracy.
Pooled OOF balanced accuracy is accompanied by a 1,000-replicate site bootstrap
conditional on fixed predictions, using the same seed 303 and site order as the
completed Stage 2 models. This omits some training uncertainty. The node mask and
available feature schema use cohort-wide QC, so prediction is exploratory
conditional on this mask, not prospective validation of the entire workflow.'''
    text=replace_region(text,'CLASSIFICATION_METHODS',method,old)
    # Abstract range stays tied to all completed models, without inferring a p-value.
    pattern=r'site-held-out classification yielded balanced accuracies of .*?0\.5\.'
    abstract_result=f'site-held-out classification yielded balanced accuracies of {low:.3f}--{high:.3f}, with\n'+('all conditional confidence intervals including 0.5.' if inside.all() else f'{int(inside.sum())}/{pairs} conditional intervals including 0.5.')
    text=re.sub(pattern,lambda _:abstract_result,text,count=1,flags=re.S)
    p.write_text(text)
    p=PAPER/'supporting_information.tex';text=p.read_text()
    text=re.sub(r'All (?:four|eight|ten|\d+)(?: completed)? classifier/pipeline pairs have predictions for all 140',f'All {pairs} completed classifier/pipeline pairs have predictions for all 140',text)
    text=text.replace('both classifiers and pipelines','all completed classifiers under both pipelines').replace('for logistic regression and RBF SVM under both pipelines','for all completed classifiers under both pipelines')
    note=f'''\nThe original-algorithm extension currently covers {len(models)} model types
({', '.join(NAMES[m] for m in models)}), with {pairs} complete pipeline/model pairs.
'''
    note+=('All five original model types are now represented.' if not pending else 'XGBoost is pending; no surrogate boosting estimator or incomplete OOF score is substituted.')
    note+='\nThe prior logistic/SVM records are unchanged. The extension plan, dependency versions,\nfold checkpoints, and combined outputs are supplied in S2.\n'
    text=replace_region(text,'CLASSIFICATION_SUPPLEMENT',note,'The final repair reuses the 626 feature rows with SHA-256') if '% CLASSIFICATION_SUPPLEMENT_BEGIN' in text else text.replace('The final repair reuses the 626 feature rows with SHA-256','% CLASSIFICATION_SUPPLEMENT_BEGIN\n'+note+'% CLASSIFICATION_SUPPLEMENT_END\n\nThe final repair reuses the 626 feature rows with SHA-256',1)
    p.write_text(text)
    p=PAPER/'README.md';text=p.read_text()
    old='''Original five-model classification is only partly matched: GaussianNB, KNN and
XGBoost have not been rerun. The original spectral descriptor is separately
reported as exploratory, outside the frozen 88-feature schema. These differences
are explicit rather than represented as completed identical replications.'''
    readme=f'''Classification now includes {len(models)} original algorithm types and {pairs} complete
pipeline/model pairs: {', '.join(models)}. '''
    readme+=('All original algorithm types are complete.\n' if not pending else '''Only XGBoost remains pending because the sandbox cannot install its dependency.
From a normal terminal at the repository root, run:

```sh
bash Codes/finish_original_classifiers.sh
```

This installs the official CPU package, resumes only missing model folds, then
refreshes the paper, figures, supporting archive, and PDFs. Existing logistic/SVM,
GaussianNB and KNN predictions are reused. No imaging or feature rerun is needed.
''')
    readme+='\nThe original spectral descriptor remains separately reported as exploratory,\noutside the frozen 88-feature schema. Matching algorithms does not equate different\ncohorts/features/validation. See results/abide_stage2_classification_extension/.\n'
    text=replace_region(text,'CLASSIFICATION_README',readme,old)
    text=text.replace('.venv_stage2/bin/python Codes/package_revision_support.py','.venv_stage2/bin/python Codes/package_revision_support.py')
    p.write_text(text)
    report=ROOT/'revision/stage2_verified/CLASSIFIER_COMPLETION.md'
    markdown='| Pipeline | Model | Balanced accuracy | Conditional 95% CI |\n|---|---|---:|---|\n'
    for row in table.itertuples():
        markdown+=f'| {row.pipeline} | {row.model} | {row.balanced_accuracy:.3f} | {row.conditional_site_bootstrap_CI95_low:.3f}–{row.conditional_site_bootstrap_CI95_high:.3f} |\n'
    report.write_text('# Original-classifier completion\n\n'+readme.replace('%','')+'\n'+markdown+'\nAll intervals are conditional on fixed OOF predictions; none is a formal model permutation test. The 59 reported raw H correlations and group-model tables are unchanged.\n')
    p=PAPER/'response_to_reviewers.md';text=p.read_text()
    marker='## Original-algorithm completion status'
    if marker in text:text=text[:text.index(marker)].rstrip()+'\n'
    text+='\n'+marker+'\n\n'+f'The revised validation now includes {len(models)} of the five original classifier\nalgorithm types, with {pairs} complete model/pipeline OOF sets of 140 participants.\n'+('All five algorithms have been completed.\n' if not pending else 'Gaussian naive Bayes and KNN are now complete; only XGBoost remains pending.\n')+'The logistic/SVM predictions are preserved. All algorithms use the same\nsite-disjoint folds and nested preprocessing/selection. Model probabilities\nare not interpreted as new clinical evidence or independent diagnostic validation.\n'
    p.write_text(text)
    subprocess.run([sys.executable,str(ROOT/'Codes/package_revision_support.py')],cwd=ROOT,check=True)
    for name in ['manuscript_revision','supporting_information']:
        logfile=ROOT/f'revision/revised_paper/{name}_build.txt'
        with logfile.open('w') as handle:
            subprocess.run(['latexmk','-cd','-pdf','-interaction=nonstopmode','-halt-on-error',str(PAPER/f'{name}.tex')],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
        log=(PAPER/f'{name}.log').read_text()
        if 'undefined' in log.lower() or 'Overfull' in log or 'Float too large' in log:raise RuntimeError(f'Inspect {name} layout/references')
    validation_path=PAPER/'draft_validation.json';validation=json.loads(validation_path.read_text())
    for filename,digest in validation['preserved_source_sha256'].items():assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==digest
    validation['classification_source']=str(STATS.relative_to(ROOT))
    validation['complete_classifier_pipeline_pairs']=pairs
    validation['completed_classifier_models']=models
    validation['pending_classifier_models']=pending
    validation['compiled_pdf_sha256']={name:hashlib.sha256((PAPER/name).read_bytes()).hexdigest() for name in ['manuscript_revision.pdf','supporting_information.pdf']}
    validation['validation'] += [f'{pairs} complete classifier/pipeline pairs; original logistic/SVM predictions and bootstrap intervals reproduced unchanged']
    validation_path.write_text(json.dumps(validation,indent=2)+'\n')
    print('Updated classifier materials:',pairs,'complete pairs; pending:',pending,flush=True)

if __name__=='__main__':main()
