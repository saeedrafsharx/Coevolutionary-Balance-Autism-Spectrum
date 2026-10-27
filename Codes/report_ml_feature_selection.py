"""Figures, feature-name audits and manuscript supplement for exploratory selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score
from classification_output_source import classification_statistics_directory
from plot_heatmap_cells import heatmap_cells
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/abide_ml_feature_selection_review'
OUT=ROOT/'revision/stage2_verified/ml_feature_selection'
PAPER=ROOT/'revision/revised_paper'
FIG=SRC/'figures'
for p in [OUT,FIG]:p.mkdir(parents=True,exist_ok=True)
MODELS=['LogisticRegression','SVM_RBF','GaussianNB','KNN','XGBoost']
SHORT={'LogisticRegression':'LR','SVM_RBF':'SVM','GaussianNB':'GNB','KNN':'KNN','XGBoost':'XGB'}
plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(FIG/f'{name}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
def latex(frame,name,caption,label):
 (PAPER/'tables'/name).write_text('\\begin{table}[htbp]\n\\centering\\small\n'+f'\\caption{{{caption}}}\\label{{{label}}}\n'+frame.to_latex(index=False,escape=True,float_format=lambda x:f'{x:.3f}')+'\\end{table}\n')
summary=json.loads((SRC/'completion_summary.json').read_text());assert summary['complete']
base_stats=classification_statistics_directory()
baseline=pd.read_csv(base_stats/'classification_OOF_predictions.csv')
follow=pd.read_csv(SRC/'statistics/classification_OOF_predictions.csv')
results=pd.read_csv(SRC/'statistics/classification_conditional_intervals.csv')
assert len(results)==10 and len(follow)==1400
results.to_csv(OUT/'comparison_complete.csv',index=False)
# ROC curves: every completed original model, separately by pipeline.
for label,pred in [('baseline',baseline),('feature_selection',follow)]:
 fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
 for ax,pipeline in zip(axes,['GSR','noGSR']):
  for model in MODELS:
   block=pred[pred.pipeline.eq(pipeline)&pred.model.eq(model)];assert len(block)==140
   fpr,tpr,_=roc_curve(block.y_true,block.decision_score)
   ax.plot(fpr,tpr,label=f'{SHORT[model]}: AUC {roc_auc_score(block.y_true,block.decision_score):.3f}')
  ax.plot([0,1],[0,1],'--',color='gray');ax.set(title=f'{pipeline}: {label.replace("_"," ")}',xlabel='False positive rate',ylabel='True positive rate');ax.legend(fontsize=8)
 save(fig,f'{label}_ROC')
 # Normalized confusion heatmaps retain exact counts in annotations.
 fig,axes=plt.subplots(5,2,figsize=(8,13),constrained_layout=True)
 for row,model in enumerate(MODELS):
  for col,pipeline in enumerate(['GSR','noGSR']):
   block=pred[pred.pipeline.eq(pipeline)&pred.model.eq(model)]
   matrix=confusion_matrix(block.y_true,block.y_pred,labels=[0,1]);rate=matrix/matrix.sum(1,keepdims=True)
   ax=axes[row,col];heatmap_cells(ax,rate,cmap='Blues',vmin=0,vmax=1,aspect='equal')
   for i in range(2):
    for j in range(2):ax.text(j,i,f'{matrix[i,j]}\n{rate[i,j]:.0%}',ha='center',va='center',color='white' if rate[i,j]>.6 else 'black')
   ax.set(title=f'{pipeline}: {SHORT[model]}',xticks=[0,1],xticklabels=['Control','ASD'],yticks=[0,1],yticklabels=['Control','ASD'],xlabel='Predicted',ylabel='Observed')
 save(fig,f'{label}_confusion_heatmaps')
# Performance heatmaps report every model, not just a best outer score.
fig,axes=plt.subplots(1,2,figsize=(10,4.6),constrained_layout=True)
for ax,metric in zip(axes,['balanced_accuracy','AUC']):
 matrix=np.empty((5,4))
 for i,model in enumerate(MODELS):
  for j,(kind,pipeline) in enumerate([(k,p) for k in ['Baseline','Selection'] for p in ['GSR','noGSR']]):
   if kind=='Selection':value=results[results.pipeline.eq(pipeline)&results.model.eq(model)][metric].iloc[0]
   else:
    block=baseline[baseline.pipeline.eq(pipeline)&baseline.model.eq(model)]
    if metric=='AUC':value=roc_auc_score(block.y_true,block.decision_score)
    else:value=results[results.pipeline.eq(pipeline)&results.model.eq(model)].baseline_balanced_accuracy.iloc[0]
   matrix[i,j]=value
 im=heatmap_cells(ax,matrix,cmap='RdBu_r',vmin=.3,vmax=.7)
 for i in range(5):
  for j in range(4):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center')
 ax.set(xticks=range(4),xticklabels=['Base\nGSR','Base\nnoGSR','Selection\nGSR','Selection\nnoGSR'],yticks=range(5),yticklabels=[SHORT[m] for m in MODELS],title='AUC' if metric=='AUC' else 'Balanced accuracy');ax.axvline(1.5,color='black',lw=.7)
 fig.colorbar(im,ax=ax,shrink=.75)
save(fig,'performance_heatmaps')
# Feature selection is frequency over five overlapping training folds, not importance.
b=pd.read_csv(SRC/'audit/baseline_feature_selection.csv')
bfreq=b.groupby(['pipeline','model','feature']).selected.mean().rename('selection_frequency').reset_index()
bfreq.to_csv(OUT/'baseline_feature_selection_frequencies.csv',index=False)
freq=pd.read_csv(SRC/'statistics/feature_selection_frequencies.csv').rename(columns={'mean':'selection_frequency'})
renames={'Inter_':'Between ','Intra_':'Within ','Salience_VentAttn':'SV','Frontoparietal':'FP','DorsalAttn':'DA','SomatoMotor':'SM','Visual':'VIS','Default':'DMN','Limbic':'LIM','AvgWeight':'W','AvgFalffW':'F','EnergyBlock':'E','Whole_GlobalEfficiency':'Global efficiency','Whole_LocalEfficiency':'Local efficiency','Whole_Prop_':'Prop. ','imbalanced_same':'imbalanced same','imbalanced_opp':'imbalanced opposite','Whole_Energy_global':'Global H'}
def feature_label(name):
 for a,b in renames.items():name=name.replace(a,b)
 return name.replace('__','–').replace('_',' ')
for label,frame in [('baseline',bfreq),('feature_selection',freq)]:
 ranking=frame.groupby('feature').selection_frequency.mean().sort_values(ascending=False,kind='stable').head(25).index
 columns=pd.MultiIndex.from_product([['GSR','noGSR'],MODELS])
 matrix=frame.pivot(index='feature',columns=['pipeline','model'],values='selection_frequency').loc[ranking].reindex(columns=columns)
 fig,ax=plt.subplots(figsize=(8.5,8.8),constrained_layout=True)
 im=heatmap_cells(ax,matrix.to_numpy(),cmap='viridis',vmin=0,vmax=1)
 ax.set(xticks=range(10),xticklabels=[f'{p}\n{SHORT[m]}' for p,m in columns],yticks=range(25),yticklabels=[feature_label(f) for f in ranking],title=f'{label.replace("_"," ").title()}: 25 highest mean selection frequencies')
 ax.tick_params(labelsize=8);ax.axvline(4.5,color='white',lw=1);fig.colorbar(im,ax=ax,label='Selected in fraction of five outer-training fits',shrink=.75)
 save(fig,f'{label}_selection_frequency')
# Names, not just a plotted subset, are retained in the standalone audit.
freq.to_csv(OUT/'followup_feature_selection_frequencies_all88.csv',index=False)
pd.read_csv(SRC/'statistics/feature_selection_by_fold.csv').to_csv(OUT/'selected_features_by_fold.csv',index=False)
chosen=pd.read_csv(SRC/'statistics/selector_method_counts.csv');chosen.to_csv(OUT/'selector_method_counts.csv',index=False)
t=results[['pipeline','model','baseline_balanced_accuracy','balanced_accuracy','delta_balanced_accuracy','paired_delta_CI95_low','paired_delta_CI95_high']].copy()
t['model']=t.model.map(SHORT);t['Change 95% CI']=[f'{a:.3f}, {b:.3f}' for a,b in zip(t.paired_delta_CI95_low,t.paired_delta_CI95_high)]
t=t.drop(columns=['paired_delta_CI95_low','paired_delta_CI95_high']);t.columns=['Pipeline','Model','Base BA','Selection BA','Change','Change 95% CI']
latex(t,'feature_selection_comparison.tex','Exploratory feature-selection follow-up, using the same known site folds. BA is pooled OOF balanced accuracy. Change intervals use paired site bootstrap conditional on fixed predictions; they do not include all training uncertainty or correct for repeated exploration. No model is promoted by outer-fold rank.','tab:FScomparison')
checks=pd.read_csv(SRC/'audit/baseline_selector_reconstruction_checks.csv')
assert checks.prediction_matches.all()
method_counts=chosen.groupby('selection_method').outer_folds.sum()
text='''# Feature selection and exploratory ML follow-up — October 27, 2026

## Did the original revised classifiers use feature selection?

Yes. All five classifier types used nested training-fold imputation, variance
filtering, scaling and ANOVA ranking. Inner CV chose 25%, 50% or 100% of features.
Reconstruction of all 50 outer fits reproduces their saved predictions. The
selected counts were 22 features in 11 fits, 44 in 17 fits, and 88 in 22 fits.
Choosing 100% is the no-reduction option within that trained selection search.
The baseline_feature_selection.csv audit contains every selected feature name.

## What was added?

A separate exploratory run removes near-exact redundant columns within training
folds and jointly compares all remaining features, ANOVA, mutual information and
L1-coefficient-ranked top 5/10/20 selection. All five algorithms use the original
five outer/three inner site-disjoint folds. No diagnosis-based trimming or ComBat
is used. All inner CV candidates and selected features are exported. L1 ties use
training ANOVA rank; zero-coefficient selections are counted, not hidden.

This follow-up was designed after seeing the original outer results and therefore
is not an independent validation of improvement. Its narrower XGBoost grid and
changed representation mean any gain is not attributable to selector method alone.
No classifier is selected from outer accuracy and no positive result is guaranteed.

## All results

| Pipeline | Model | Baseline BA | Selection BA | Change | Paired conditional 95% CI for change |
|---|---|---:|---:|---:|---|
'''
for r in results.itertuples():
 text+=f'| {r.pipeline} | {r.model} | {r.baseline_balanced_accuracy:.3f} | {r.balanced_accuracy:.3f} | {r.delta_balanced_accuracy:+.3f} | {r.paired_delta_CI95_low:.3f}, {r.paired_delta_CI95_high:.3f} |\n'
text+='\n## Selection-method choices\n\n'
for method,count in method_counts.items():text+=f'- {method}: {count} of 50 outer-training fits.\n'
text+='''
## Figures and files

All five-model ROC curves, normalized confusion heatmaps, baseline-versus-selection
BA/AUC heatmaps, and selected-feature frequency heatmaps are in
results/abide_ml_feature_selection_review/figures/. Both PDF and PNG are supplied.
Heatmaps show all algorithms; frequency panels show the top 25 descriptive
frequencies with all 88 features provided as CSV. Frequencies are not biomarker
significance, model permutation importance or independent replication.
The original study group tests, nulls, and 59 H associations are unchanged.
'''
(OUT/'FEATURE_SELECTION_AND_RESULTS.md').write_text(text)
print(results[['pipeline','model','baseline_balanced_accuracy','balanced_accuracy','delta_balanced_accuracy','paired_delta_CI95_low','paired_delta_CI95_high','AUC']].to_string(index=False))
print('Saved baseline/follow-up ROC, confusion, performance and selection-frequency heatmaps.')
