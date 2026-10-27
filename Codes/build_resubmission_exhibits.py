"""Publication figures/tables from audited outputs, preserving original group colors."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle
from PIL import Image
from sklearn.metrics import roc_curve,roc_auc_score,balanced_accuracy_score,f1_score,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from abide_stage2 import model_design
from classification_output_source import classification_statistics_directory
from plot_heatmap_cells import heatmap_cells
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'revision/resubmission';FIG=OUT/'figures';UP=OUT/'upload_figures';TAB=OUT/'tables'
for p in [FIG,UP,TAB]:p.mkdir(parents=True,exist_ok=True)
SRC=ROOT/'results/abide_stage2_repaired';STAT=SRC/'statistics';FS=ROOT/'results/abide_ml_feature_selection_review';CONT=ROOT/'revision/stage2_verified/continuity'
META=pd.read_csv(SRC/'audit/demographic_manifest.csv')
FEATURES=pd.read_csv(SRC/'features/all_designs_raw_untrimmed.csv')
PRIMARY=FEATURES[FEATURES.design.eq('primary_common_106')].merge(META,on='SUB_ID',validate='many_to_one')
GROUP={1:('ASD','#DA005D'),2:('Control','#00473A')}
MC={'GaussianNB':'#1f77b4','SVM_RBF':'#ff7f0e','LogisticRegression':'#2ca02c','KNN':'#d62728','XGBoost':'#9467bd'}
MODELS=['GaussianNB','SVM_RBF','LogisticRegression','KNN','XGBoost'];SHORT={'GaussianNB':'GNB','SVM_RBF':'SVM','LogisticRegression':'LR','KNN':'KNN','XGBoost':'XGB'}
NETS=['Visual','SomatoMotor','DorsalAttn','Salience_VentAttn','Limbic','Frontoparietal','Default']
ABBR=dict(zip(NETS,['VIS','SM','DA','SV','LIM','FP','DMN']))
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
manifest=[]
def save(fig,number,name,main=True):
 key=f'Fig{number}' if main else f'SFig{number}'
 fig.savefig(FIG/f'{key}.pdf',facecolor='white')
 fig.savefig(FIG/f'{key}.png',dpi=200,facecolor='white')
 if main:
  temp=UP/f'{key}.tif';fig.savefig(temp,dpi=600,format='tiff',facecolor='white')
  with Image.open(temp) as im:rgb=im.convert('RGB')
  rgb.save(temp,compression='tiff_lzw',dpi=(600,600))
  with Image.open(temp) as im:
   assert im.mode=='RGB' and im.info.get('compression')=='tiff_lzw'
   manifest.append({'file':temp.name,'description':name,'width_pixels':im.width,'height_pixels':im.height,'DPI':600,'RGB':True,'bytes':temp.stat().st_size,'sha256':hashlib.sha256(temp.read_bytes()).hexdigest()})
 plt.close(fig)
def table(frame,name,title,legend,label):
 text='\\begin{table}[H]\n\\centering\\small\n'+f'\\caption{{\\textbf{{{title}}}}}\\label{{{label}}}\n'
 text+=frame.to_latex(index=False,escape=True,float_format=lambda x:f'{x:.3f}')
 text+='\\begin{flushleft}\\footnotesize\n'+legend+'\n\\end{flushleft}\n\\end{table}\n'
 (TAB/name).write_text(text)
def fmt(x,digits=3):return '--' if not np.isfinite(x) else f'{x:.{digits}f}'
def ci(row):return f'{row.CI95_low:.2f}, {row.CI95_high:.2f}'
def dots(ax,field,title,data=PRIMARY):
 for j,pipeline in enumerate(['GSR','noGSR']):
  for k,(dx,(group,color)) in enumerate(GROUP.items()):
   values=data.loc[data.pipeline.eq(pipeline)&data.DX_GROUP.eq(dx),field].to_numpy();pos=j*3+k
   ax.boxplot([values],positions=[pos],widths=.65,showfliers=False,medianprops={'color':'black'},boxprops={'color':color})
   ax.scatter(pos+np.random.default_rng(101).uniform(-.18,.18,len(values)),values,s=7,alpha=.65,color=color,linewidths=0)
 ax.set(xticks=[.5,3.5],xticklabels=['GSR','No-GSR'],title=title)
 ax.tick_params(axis='y',labelsize=8)
def featlabel(value):
 replacements={'Whole_Energy_global':'H','Whole_GlobalEfficiency':'Global efficiency','Whole_LocalEfficiency':'Local efficiency','Whole_Prop_':'Prop. ','Inter_':'','Intra_':'Within ','AvgWeight':'W','AvgFalffW':'F','EnergyBlock':'E','imbalanced_same':'imbalanced same','imbalanced_opp':'imbalanced opposite',**ABBR}
 for a,b in replacements.items():value=value.replace(a,b)
 return value.replace('__','–').replace('_',' ')
# Fig1: cohort and exact separation of upstream derivatives/local work.
fig,ax=plt.subplots(figsize=(7.5,6.2),constrained_layout=True);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
boxes=[(.5,.92,.88,.10,'Existing ABIDE I source pool: 182 (90 ASD / 92 controls)'),(.5,.76,.88,.12,'Age 18–30; observed FIQ >80 → 140 (69 ASD / 71 controls)\nFixed coverage-qualified CC200 network: 106 nodes'),(.25,.55,.43,.15,'Downloaded filtered ROI series\nGSR and no-GSR\nLocal Fisher-transformed FC'),(.75,.55,.43,.15,'Downloaded fALFF maps/masks\nUpstream computation\nLocal ROI means / relative states'),(.5,.34,.88,.12,'Static H; pairwise configurations; global/local efficiency\nWithin-network and between-network descriptors'),(.2,.11,.29,.15,'Null configurations\nAll 140 participants\nBoth diagnoses'),(.51,.11,.29,.15,'Adjusted group models\n136 (67 ASD / 69 controls)\nSite, age, FIQ and FD'),(.82,.11,.29,.15,'Clinical / prediction\nScore-wise missingness\nSite-held-out models')]
for x,y,w,h,label in boxes:
 ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.012',facecolor='#f3f5f7',edgecolor='#405465',lw=1));ax.text(x,y,label,ha='center',va='center',fontsize=9)
for a,b in [((.5,.865),(.5,.83)),((.35,.695),(.25,.635)),((.65,.695),(.75,.635)),((.25,.47),(.4,.41)),((.75,.47),(.6,.41)),((.35,.27),(.2,.195)),((.5,.27),(.51,.195)),((.65,.27),(.82,.195))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#405465'})
save(fig,1,'Workflow and participant flow')
# Fig2: both-diagnosis participant differences against own null expectation.
null=pd.read_csv(CONT/'both_groups_null_comparisons.csv');null['delta']=null.observed_H-null.null_mean
schemes=['node_permutation','network_hemisphere_blocks'];scheme_names={'node_permutation':'Unrestricted states','network_hemisphere_blocks':'Network/hemisphere blocks'}
fig,axes=plt.subplots(2,2,figsize=(7.5,6.4),constrained_layout=True)
for row,scheme in enumerate(schemes):
 for col,pipeline in enumerate(['GSR','noGSR']):
  ax=axes[row,col]
  for i,(dx,(group,color)) in enumerate(GROUP.items()):
   values=null.loc[null.pipeline.eq(pipeline)&null['null'].eq(scheme)&null.DX_GROUP.eq(dx),'delta'];ax.boxplot([values],positions=[i],widths=.55,showfliers=False,medianprops={'color':'black'});ax.scatter(i+np.random.default_rng(101).uniform(-.12,.12,len(values)),values,s=12,color=color,alpha=.7,linewidths=0)
  ax.axhline(0,color='gray',ls='--',lw=.8);ax.set(xticks=[0,1],xticklabels=['ASD','Control'],ylabel='Empirical H − own null mean',title=f'{pipeline}: {scheme_names[scheme]}')
save(fig,2,'Empirical versus null alignment in both groups')
# Fig3: all nonlegacy global features; group colors retained.
fields={'Whole_Energy_global':'(A) Hamiltonian H','Whole_GlobalEfficiency':'(B) Global efficiency','Whole_LocalEfficiency':'(C) Local efficiency','Whole_Prop_agreement':'(D) Agreement','Whole_Prop_disagreement':'(E) Disagreement','Whole_Prop_imbalanced_same':'(F) Imbalanced-same','Whole_Prop_imbalanced_opp':'(G) Imbalanced-opposite'}
fig,axes=plt.subplots(2,4,figsize=(7.5,6.3),constrained_layout=True)
for ax,(field,title) in zip(axes.flat,fields.items()):dots(ax,field,title)
ax=axes.flat[-1];ax.axis('off');ax.text(.05,.8,'ASD: magenta (n=69)',color=GROUP[1][1],fontsize=10);ax.text(.05,.64,'Control: dark green (n=71)',color=GROUP[2][1],fontsize=10);ax.text(.05,.41,'Raw, untrimmed features\nSame 106 nodes in both pipelines\nAdjusted group models: n=136',fontsize=9,linespacing=1.5)
save(fig,3,'Global distributions')
# Figs4/5: all block comparisons, avoiding selected nominal results.
groups=pd.read_csv(STAT/'group_models.csv');g=groups[groups.design.eq('primary_common_106')&groups.model.eq('raw_adjusted')]
for family,number in [('Intra',4),('Inter',5)]:
 blocks=[(n,n) for n in NETS] if family=='Intra' else [(a,b) for i,a in enumerate(NETS) for b in NETS[i+1:]]
 matrix=np.full((len(blocks),6),np.nan)
 for i,(a,b) in enumerate(blocks):
  for j,(pipeline,suffix) in enumerate([(p,s) for p in ['GSR','noGSR'] for s in ['AvgWeight','AvgFalffW','EnergyBlock']]):
   key=f'Intra_{a}_{suffix}' if a==b else f'Inter_{a}__{b}_{suffix}';found=g[g.pipeline.eq(pipeline)&g.Feature.eq(key)&g.status.eq('ok')]
   if len(found):matrix[i,j]=found.beta.iloc[0]/found.SE_HC3.iloc[0]
 fig,ax=plt.subplots(figsize=(7.5,4.5 if family=='Intra' else 7.8),constrained_layout=True);cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#d9d9d9');im=heatmap_cells(ax,matrix,cmap=cmap,vmin=-3,vmax=3)
 ax.set(xticks=range(6),xticklabels=['GSR\nMean FC','GSR\nState–FC','GSR\nEnergy','No-GSR\nMean FC','No-GSR\nState–FC','No-GSR\nEnergy'],yticks=range(len(blocks)),yticklabels=[ABBR[a] if a==b else ABBR[a]+'–'+ABBR[b] for a,b in blocks]);ax.axvline(2.5,color='black',lw=.8);fig.colorbar(im,ax=ax,label='Diagnosis coefficient / HC3 SE',shrink=.8);save(fig,number,f'{family}-network comparisons')
# Fig6: original top-20 H relationships, with sign colors and group-colored titles.
h=pd.read_csv(CONT/'H_vs_all_metrics_complete.csv');raw=h[h.version.eq('raw_eligible')]
fig,axes=plt.subplots(2,2,figsize=(7.5,7.8),constrained_layout=True)
for row,(dx,(group,color)) in enumerate(GROUP.items()):
 for col,pipeline in enumerate(['GSR','noGSR']):
  frame=raw[raw.pipeline.eq(pipeline)&raw.group.eq(group)];frame=frame.loc[frame.Pearson_r.abs().nlargest(20).index].sort_values('Pearson_r');ax=axes[row,col]
  ax.barh(range(20),frame.Pearson_r,color=np.where(frame.Pearson_r>0,'#d62728','#1f77b4'));ax.set(yticks=range(20),yticklabels=[featlabel(r.feature)+(' *' if r.Pearson_q_per_group87<.05 else '') for r in frame.itertuples()],xlabel='Pearson r with H',xlim=(-.65,.65));ax.set_title(f'{pipeline}: {group}',color=color);ax.tick_params(labelsize=8);ax.axvline(0,color='gray',lw=.7)
save(fig,6,'Within-group H correlations')
# Fig7: exact adjusted efficiency analysis.
eff=pd.read_csv(STAT/'efficiency_partial_correlations.csv')
fig,axes=plt.subplots(2,2,figsize=(7.5,6.2),constrained_layout=True)
for col,pipeline in enumerate(['GSR','noGSR']):
 d=PRIMARY[PRIMARY.pipeline.eq(pipeline)];d=d[d.SITE_ID.map(d.SITE_ID.value_counts()).ge(3)]
 for row,(dx,(group,color)) in enumerate(GROUP.items()):
  b=d[d.DX_GROUP.eq(dx)];b=b[b.SITE_ID.map(b.SITE_ID.value_counts()).ge(2)].reset_index(drop=True);x=model_design(b,include_group=False);x['mean_FC']=b.QC_MeanFisherZ_FC
  residual=lambda y:np.asarray(y)-x.to_numpy()@np.linalg.lstsq(x,y,rcond=None)[0]
  a=residual(b.Whole_GlobalEfficiency);z=residual(b.Whole_Energy_global);expected=eff[eff.pipeline.eq(pipeline)&eff.DX_GROUP.eq(dx)&eff.feature.eq('Whole_GlobalEfficiency')].iloc[0];assert np.isclose(np.corrcoef(a,z)[0,1],expected.partial_r)
  ax=axes[row,col];ax.scatter(a,z,s=17,color=color,alpha=.75,linewidths=0);line=np.linspace(a.min(),a.max(),100);ax.plot(line,np.polyval(np.polyfit(a,z,1),line),color='gray');ax.set(xlabel='Global-efficiency residual',ylabel='H residual');ax.set_title(f'{pipeline}: {group}; n={len(b)}\nr={expected.partial_r:.3f}; q={expected.p_BH:.3f}',color=color)
save(fig,7,'Adjusted efficiency associations')
# Fig8: clinical results are ASD-only; pipeline encoded by marker/hatching, not group color.
clinical=pd.read_csv(STAT/'clinical_H_partial_correlations.csv');plsr=pd.read_csv(STAT/'clinical_PLSR_site_heldout.csv');scores=list(plsr.score.unique())
fig,axes=plt.subplots(1,2,figsize=(7.5,4.7),constrained_layout=True)
for offset,pipeline,marker in [(-.15,'GSR','o'),(.15,'noGSR','^')]:
 c=clinical[clinical.pipeline.eq(pipeline)].set_index('score').loc[scores];p=plsr[plsr.pipeline.eq(pipeline)].set_index('score').loc[scores]
 axes[0].scatter(c.partial_r,np.arange(5)+offset,label=pipeline,marker=marker,s=35,facecolors=GROUP[1][1] if pipeline=='GSR' else 'white',edgecolors=GROUP[1][1])
 axes[1].barh(np.arange(5)+offset,p.OOF_R2,height=.28,label=pipeline,facecolor=GROUP[1][1] if pipeline=='GSR' else 'white',edgecolor=GROUP[1][1],hatch=None if pipeline=='GSR' else '///')
for ax in axes:ax.set(yticks=range(5),yticklabels=[x.replace('_',' ') for x in scores]);ax.axvline(0,color='gray',ls='--',lw=.8);ax.legend(fontsize=8)
axes[0].set(xlabel='Adjusted H–clinical partial r',title='(A) Clinical associations');axes[1].set(xlabel='Site-held-out OOF R²',title='(B) PLS regression')
save(fig,8,'Clinical associations and PLS regression')
# Figs9/10: complete five-model OOF results, original algorithm colors.
classstats=classification_statistics_directory();pred=pd.read_csv(classstats/'classification_OOF_predictions.csv');ints=pd.read_csv(classstats/'classification_conditional_intervals.csv')
fig,axes=plt.subplots(1,2,figsize=(7.5,4.2),constrained_layout=True)
for ax,pipeline in zip(axes,['GSR','noGSR']):
 for model in MODELS:
  b=pred[pred.pipeline.eq(pipeline)&pred.model.eq(model)];assert len(b)==140;fpr,tpr,_=roc_curve(b.y_true,b.decision_score);ax.plot(fpr,tpr,color=MC[model],label=f'{SHORT[model]}: {roc_auc_score(b.y_true,b.decision_score):.3f}',lw=1.5)
 ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='False positive rate',ylabel='True positive rate',title=pipeline);ax.legend(title='Model: AUC',fontsize=8,title_fontsize=9)
save(fig,9,'Complete site-held-out ROC curves')
fig,ax=plt.subplots(figsize=(7.5,4.4),constrained_layout=True)
for i,model in enumerate(MODELS):
 for shift,pipeline,marker in [(-.12,'GSR','o'),(.12,'noGSR','s')]:
  r=ints[ints.pipeline.eq(pipeline)&ints.model.eq(model)].iloc[0];ax.errorbar(r.balanced_accuracy,i+shift,xerr=[[r.balanced_accuracy-r.conditional_site_bootstrap_CI95_low],[r.conditional_site_bootstrap_CI95_high-r.balanced_accuracy]],fmt=marker,color=MC[model],mfc=MC[model] if pipeline=='GSR' else 'white',capsize=3)
ax.axvline(.5,color='gray',ls='--');ax.set(yticks=range(5),yticklabels=[SHORT[m] for m in MODELS],xlabel='Pooled OOF balanced accuracy (conditional 95% CI)');ax.plot([],[],'ok',label='GSR');ax.plot([],[],'sk',mfc='white',label='No-GSR');ax.legend()
save(fig,10,'Classifier comparison')
# Fig11: redraw conceptual configurations at publication resolution, same sign colors.
fig,ax=plt.subplots(figsize=(7.5,3.0),constrained_layout=True);ax.set(xlim=(0,4),ylim=(0,1));ax.axis('off')
for i,(title,other,edge,contribution) in enumerate([('Agreement',1,1,'−|z|'),('Imbalanced-same',1,-1,'+|z|'),('Imbalanced-opposite',-1,1,'+|z|'),('Disagreement',-1,-1,'−|z|')]):
 ax.text(i+.5,.94,title,ha='center',fontsize=9);ax.plot([i+.25,i+.75],[.57,.57],color=GROUP[2][1] if edge==1 else GROUP[1][1],lw=4,zorder=1)
 for x,state in [(i+.25,1),(i+.75,other)]:ax.add_patch(Circle((x,.57),.13,color=GROUP[1][1] if state==1 else GROUP[2][1],zorder=2));ax.text(x,.57,'+1' if state==1 else '−1',ha='center',va='center',color='white',weight='bold',fontsize=12)
 ax.text(i+.5,.29,'FC sign: '+('+' if edge==1 else '−'),ha='center',fontsize=10);ax.text(i+.5,.12,'H contribution: '+contribution,ha='center',fontsize=10)
save(fig,11,'Pairwise state-sign configurations')
# Main tables.
ids=pd.read_csv(STAT/'model_participants_primary_common_106_GSR.csv');model=META[META.SUB_ID.isin(ids.SUB_ID)]
rows=[]
for sample,d in [('Eligible',META),('Adjusted model',model)]:
 for dx,b in d.groupby('DX_GROUP'):
  rows.append({'Sample':sample,'Group':GROUP[dx][0],'N':len(b),'Age, years':f'{b.AGE_AT_SCAN.mean():.2f} ({b.AGE_AT_SCAN.std():.2f})','FIQ':f'{b.FIQ.mean():.2f} ({b.FIQ.std():.2f})','Mean FD, mm':f'{b.func_mean_fd.mean():.3f} ({b.func_mean_fd.std():.3f})'})
table(pd.DataFrame(rows),'demographics.tex','Participant characteristics in the two primary analysis samples.','Continuous values are mean (SD). All participants were recorded male. Eligibility is age 18--30 inclusive and observed FIQ $>80$. No handedness restriction or missing-IQ imputation is applied. The adjusted-model subset excludes four participants at sites with fewer than three eligible participants.','tab:demographics')
n=null.groupby(['group','pipeline','null']).agg(N=('SUB_ID','size'),H=('observed_H','mean'),Null_H=('null_mean','mean'),Mean_z=('z','mean'),Significant=('q_BH_all560_null_comparisons',lambda p:(p<.05).sum())).reset_index();n['null']=n['null'].map({'node_permutation':'U','network_hemisphere_blocks':'B'});n.columns=['Group','Pipeline','Null','N','Mean H','Mean null H','Mean z','q<.05 count']
table(n,'nulls.tex','Empirical alignment against state-permutation null configurations.','U: unrestricted state permutations; B: network/hemisphere-blocked permutations. Each person has 1,000 permutations under each scheme. BH correction here spans all 560 person/pipeline/null comparisons; the counts do not test an ASD--control difference.','tab:nulls')
energy=groups[groups.design.eq('primary_common_106')&groups.Feature.eq('Whole_Energy_global')&groups.status.eq('ok')];t=[]
for r in energy.itertuples():t.append({'Pipeline':r.pipeline,'Model':{'raw_adjusted':'Raw adjusted','ComBat_sensitivity_adjusted':'ComBat','energy_trim_sensitivity':'Pooled trim'}[r.model],'N':int(r.n),'Coefficient':r.beta,'95% CI':ci(r),'p':r.p,'Holm p':r.p_adjusted})
table(pd.DataFrame(t),'primary_H.tex','Adjusted diagnosis coefficients for primary-network H.','Coefficients compare ASD with controls, adjusting for site, age, FIQ and mean FD. HC3 intervals use a t reference. Holm correction covers the two pipelines within each model version. ComBat and pooled trimming are exploratory sensitivities.','tab:primary')
rows=[]
for net in NETS:
 row={'Network':ABBR[net]}
 for pipe,head in [('GSR','GSR'),('noGSR','No-GSR')]:
  d=g[g.pipeline.eq(pipe)&g.Feature.eq(f'Intra_{net}_EnergyBlock')&g.status.eq('ok')]
  row[head+' beta']=fmt(d.beta.iloc[0]) if len(d) else '--';row[head+' 95% CI']=ci(d.iloc[0]) if len(d) else '--';row[head+' q']=fmt(d.p_adjusted.iloc[0]) if len(d) else '--'
 rows.append(row)
table(pd.DataFrame(rows),'intra_energy.tex','Within-network energy diagnosis coefficients.','Raw adjusted models use 136 participants. BH spans both pipelines and all available within-network measures (36 tests). Limbic energy is unavailable because only one node is assigned. VIS: visual; SM: somatomotor; DA: dorsal attention; SV: salience/ventral attention; LIM: limbic; FP: frontoparietal; DMN: default mode. Other block measures are shown in Fig.~\\ref{fig:intra} and S2 Data.','tab:intra')
rows=[]
for i,a in enumerate(NETS):
 for b in NETS[i+1:]:
  row={'Pair':ABBR[a]+'--'+ABBR[b]}
  for pipeline,head in [('GSR','GSR'),('noGSR','No-GSR')]:
   d=g[g.pipeline.eq(pipeline)&g.Feature.eq(f'Inter_{a}__{b}_AvgWeight')&g.status.eq('ok')].iloc[0];row[head+' beta']=f'{d.beta:.4f}';row[head+' p']=d.p;row[head+' q']=d.p_adjusted
  rows.append(row)
table(pd.DataFrame(rows),'inter_FC.tex','Between-network mean-FC diagnosis coefficients.','Raw adjusted models use 136 participants. BH spans all three between-network measures in both pipelines (126 tests); q values therefore refer to the full family, not just this mean-FC table. Network abbreviations follow Table~\\ref{tab:intra}. State-weighted FC and energy are included in Fig.~\\ref{fig:inter} and S2 Data.','tab:inter')
count=pd.read_csv(CONT/'H_correlation_significance_counts.csv');t=count[count.version.eq('raw_eligible')][['pipeline','group','n','tests','Pearson_q87_below05','Spearman_q87_below05']];t.columns=['Pipeline','Group','N','Tests','Pearson q<.05','Spearman q<.05']
table(t,'H_counts.tex','Within-group associations between H and the other network metrics.','BH is applied separately over 87 features per pipeline/diagnosis/estimator, following the original descriptive Pearson convention. These counts include algebraically related features and are not independent biological discoveries or group contrasts. All coefficients and alternative pooled corrections are in S2 Data.','tab:Hcounts')
t=clinical.merge(plsr,on=['pipeline','score','n']);t['Score']=t.score.str.replace('_',' ');t['MSE ratio']=t.OOF_MSE/t.training_mean_baseline_MSE;t=t[['pipeline','Score','n','partial_r','p_BH','OOF_R2','MSE ratio']];t.columns=['Pipeline','Score','N','Partial r','BH q','OOF R2','MSE ratio']
table(t,'clinical.tex','Adjusted H associations and clinical prediction.','The partial correlation controls age, FIQ, mean FD and site; BH covers the ten score/pipeline tests. PLSR uses the common 88-feature set and nested site-held-out validation. MSE ratio compares PLSR with the corresponding outer-training-mean predictor; ratios above one favor that baseline.','tab:clinical')
rows=[]
for r in ints.itertuples():
 d=pred[pred.pipeline.eq(r.pipeline)&pred.model.eq(r.model)];rows.append({'Pipeline':r.pipeline,'Model':SHORT[r.model],'Accuracy':(d.y_true==d.y_pred).mean(),'BA':r.balanced_accuracy,'BA 95% CI':f'{r.conditional_site_bootstrap_CI95_low:.3f}, {r.conditional_site_bootstrap_CI95_high:.3f}','F1':f1_score(d.y_true,d.y_pred),'AUC':roc_auc_score(d.y_true,d.decision_score)})
table(pd.DataFrame(rows),'classification.tex','Complete site-held-out classification results.','Every row uses 140 OOF predictions (69 ASD/71 controls). BA: balanced accuracy; F1 treats ASD as positive. Intervals bootstrap sites conditional on fixed predictions and omit part of training uncertainty. LR: logistic regression; SVM: radial-basis support vector machine; GNB: Gaussian naive Bayes; KNN: k-nearest neighbors; XGB: XGBoost. All models are reported without an outer-test winner.','tab:classification')
f=pd.read_csv(FS/'statistics/classification_conditional_intervals.csv');f['Model']=f.model.map(SHORT);f['Change CI']=[f'{a:.3f}, {b:.3f}' for a,b in zip(f.paired_delta_CI95_low,f.paired_delta_CI95_high)];t=f[['pipeline','Model','baseline_balanced_accuracy','balanced_accuracy','delta_balanced_accuracy','Change CI']];t.columns=['Pipeline','Model','Baseline BA','Follow-up BA','Change','Change 95% CI']
table(t,'selection.tex','Exploratory feature-selection comparison on the same known outer folds.','Paired change intervals resample sites conditional on fixed OOF predictions. The follow-up was specified after baseline results had been examined and is not independent confirmation of improvement. Representation and XGBoost search grids also differ. All ten estimates are shown; no classifier is selected by outer performance.','tab:selection')
# Supporting plots regenerated with consistent group/model colors.
# S1 coverage-defined network composition.
mapping=pd.read_csv(ROOT/'results/abide_stage2_review/audit/anatomical_mapping_majority_cortex.csv');roi=pd.read_csv(ROOT/'revision/analysis_plan/primary_common_106_ROIs.csv');composition=mapping[mapping.ROI_label.isin(roi.ROI_label)].network_name.value_counts();composition=composition.reindex(NETS+['Unassigned'],fill_value=0)
fig,ax=plt.subplots(figsize=(7.5,3.6),constrained_layout=True);ax.bar(range(8),composition,color='#536878');ax.set(xticks=range(8),xticklabels=[ABBR.get(x,'Unassigned') for x in composition.index],ylabel='Retained primary nodes');[ax.text(i,n+.4,str(n),ha='center') for i,n in enumerate(composition)];save(fig,1,'Primary network composition',False)
# S2 raw adjusted node sensitivity coefficients.
e=groups[groups.model.eq('raw_adjusted')&groups.Feature.eq('Whole_Energy_global')&groups.status.eq('ok')];fig,ax=plt.subplots(figsize=(7.5,4.3),constrained_layout=True)
for i,r in enumerate(e.itertuples()):ax.errorbar(r.beta,i,xerr=[[r.beta-r.CI95_low],[r.CI95_high-r.beta]],fmt='o',color='#405465',capsize=3)
ax.axvline(0,color='gray',ls='--');ax.set(yticks=range(6),yticklabels=[f"{r.design.replace('primary_common_','').replace('sensitivity_broader_','').replace('sensitivity_full','')} nodes; {r.pipeline}; n={int(r.n)}" for r in e.itertuples()],xlabel='Adjusted ASD coefficient (HC3 95% CI)');save(fig,2,'Node-design sensitivity',False)
# S3 dedicated control empirical-v-own-null means.
fig,axes=plt.subplots(2,2,figsize=(7.5,6.2),constrained_layout=True)
for row,scheme in enumerate(schemes):
 for col,pipeline in enumerate(['GSR','noGSR']):
  b=null[null.DX_GROUP.eq(2)&null.pipeline.eq(pipeline)&null['null'].eq(scheme)];ax=axes[row,col];ax.plot([0,1],np.vstack([b.observed_H,b.null_mean]),color='#9bac9f',lw=.6,alpha=.5);ax.scatter(np.zeros(71),b.observed_H,color=GROUP[2][1],s=10);ax.scatter(np.ones(71),b.null_mean,color='gray',s=10);ax.set(xticks=[0,1],xticklabels=['Empirical H','Own null mean'],ylabel='H',title=f'{pipeline}: {scheme_names[scheme]}')
save(fig,3,'Male-control empirical versus null means',False)
# S4 individual standardized null scores.
fig,axes=plt.subplots(2,2,figsize=(7.5,6.2),constrained_layout=True)
for row,scheme in enumerate(schemes):
 for col,pipeline in enumerate(['GSR','noGSR']):
  ax=axes[row,col]
  for i,(dx,(group,color)) in enumerate(GROUP.items()):
   b=null[null.DX_GROUP.eq(dx)&null.pipeline.eq(pipeline)&null['null'].eq(scheme)];ax.scatter(i+np.random.default_rng(101).uniform(-.15,.15,len(b)),b.z,s=13,color=color,alpha=.7);ax.boxplot([b.z],positions=[i],widths=.5,showfliers=False,medianprops={'color':'black'})
  ax.axhline(0,color='gray',ls='--');ax.set(xticks=[0,1],xticklabels=['ASD','Control'],ylabel='Individual null z',title=f'{pipeline}: {scheme_names[scheme]}')
save(fig,4,'Individual null standardized scores',False)
# S5 all valid intra-network energies.
fig,axes=plt.subplots(2,3,figsize=(7.5,5.5),constrained_layout=True)
for ax,net in zip(axes.flat,[n for n in NETS if n!='Limbic']):dots(ax,f'Intra_{net}_EnergyBlock',ABBR[net])
save(fig,5,'Within-network energy distributions',False)
def confusion_plot(frame,num,label):
 fig,axes=plt.subplots(5,2,figsize=(7.5,8.5),constrained_layout=True)
 for i,model in enumerate(MODELS):
  for j,pipeline in enumerate(['GSR','noGSR']):
   d=frame[frame.pipeline.eq(pipeline)&frame.model.eq(model)];matrix=confusion_matrix(d.y_true,d.y_pred,labels=[0,1]);rate=matrix/matrix.sum(1,keepdims=True);ax=axes[i,j];heatmap_cells(ax,rate,cmap='Blues',vmin=0,vmax=1,aspect='equal')
   for a in range(2):
    for b in range(2):ax.text(b,a,str(matrix[a,b]),ha='center',va='center',color='white' if rate[a,b]>.6 else 'black',fontsize=10)
   ax.set(xticks=[0,1],xticklabels=['Control','ASD'],yticks=[0,1],yticklabels=['Control','ASD']);ax.set_title(f'{pipeline}: {SHORT[model]}',fontsize=9);ax.tick_params(labelsize=8)
 save(fig,num,label,False)
confusion_plot(pred,6,'Baseline confusion matrices')
# S7 descriptive PCA: original figure type; not used for model evaluation.
fig,axes=plt.subplots(1,2,figsize=(7.5,4.2),constrained_layout=True);pca_rows=[]
for ax,pipeline in zip(axes,['GSR','noGSR']):
 d=PRIMARY[PRIMARY.pipeline.eq(pipeline)];cols=[c for c in d if c.startswith(('Whole_','Intra_','Inter_')) and 'legacy' not in c and np.isfinite(d[c]).all() and d[c].std()>1e-12];assert len(cols)==88;pc=PCA(n_components=2,svd_solver='full');projection=pc.fit_transform(StandardScaler().fit_transform(d[cols]))
 for dx,(label,color) in GROUP.items():mask=d.DX_GROUP.eq(dx);ax.scatter(projection[mask,0],projection[mask,1],color=color,s=18,alpha=.75,label=label,linewidths=0)
 ax.set(title=pipeline,xlabel=f'PC1 ({pc.explained_variance_ratio_[0]:.1%})',ylabel=f'PC2 ({pc.explained_variance_ratio_[1]:.1%})');ax.legend()
 pca_rows.extend([{'pipeline':pipeline,'SUB_ID':int(sid),'PC1':x,'PC2':y,'scope':'descriptive full-sample standardized PCA; not part of validation'} for sid,(x,y) in zip(d.SUB_ID,projection)])
pd.DataFrame(pca_rows).to_csv(OUT/'source_support/descriptive_PCA.csv',index=False);save(fig,7,'Descriptive PCA',False)
# S8/S9/S10 follow-up comparison, ROC, confusion.
follow=pd.read_csv(FS/'statistics/classification_OOF_predictions.csv');fig,axes=plt.subplots(1,2,figsize=(7.5,4.1),constrained_layout=True)
for ax,metric in zip(axes,['BA','AUC']):
 values=np.empty((5,4))
 for i,model in enumerate(MODELS):
  for j,(version,pipeline) in enumerate([(v,p) for v in ['Baseline','Follow-up'] for p in ['GSR','noGSR']]):
   d=(pred if version=='Baseline' else follow);d=d[d.pipeline.eq(pipeline)&d.model.eq(model)];values[i,j]=balanced_accuracy_score(d.y_true,d.y_pred) if metric=='BA' else roc_auc_score(d.y_true,d.decision_score)
 im=heatmap_cells(ax,values,cmap='RdBu_r',vmin=.3,vmax=.7)
 for i in range(5):
  for j in range(4):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=9)
 ax.set(title=metric,xticks=range(4),xticklabels=['Base\nGSR','Base\nNo-GSR','Follow\nGSR','Follow\nNo-GSR'],yticks=range(5),yticklabels=[SHORT[m] for m in MODELS]);ax.tick_params(labelsize=8);fig.colorbar(im,ax=ax,shrink=.7)
save(fig,8,'Exploratory model-performance comparison',False)
fig,axes=plt.subplots(1,2,figsize=(7.5,4.2),constrained_layout=True)
for ax,pipeline in zip(axes,['GSR','noGSR']):
 for model in MODELS:
  d=follow[follow.pipeline.eq(pipeline)&follow.model.eq(model)];fpr,tpr,_=roc_curve(d.y_true,d.decision_score);ax.plot(fpr,tpr,color=MC[model],label=f'{SHORT[model]}: {roc_auc_score(d.y_true,d.decision_score):.3f}')
 ax.plot([0,1],[0,1],'--',color='gray');ax.set(title=pipeline,xlabel='False positive rate',ylabel='True positive rate');ax.legend(title='Model: AUC',fontsize=8,title_fontsize=9)
save(fig,9,'Follow-up ROC curves',False);confusion_plot(follow,10,'Follow-up confusion matrices')
# S11/S12 selection frequency maps (top 25; all 88 in data).
base_selected=pd.read_csv(FS/'audit/baseline_feature_selection.csv');bf=base_selected.groupby(['pipeline','model','feature']).selected.mean().rename('frequency').reset_index();sf=pd.read_csv(FS/'statistics/feature_selection_frequencies.csv').rename(columns={'mean':'frequency'})
for num,frame,name in [(11,bf,'Baseline selected-feature frequency'),(12,sf,'Follow-up selected-feature frequency')]:
 order=frame.groupby('feature').frequency.mean().sort_values(ascending=False,kind='stable').head(25).index;cols=pd.MultiIndex.from_product([['GSR','noGSR'],MODELS]);matrix=frame.pivot(index='feature',columns=['pipeline','model'],values='frequency').loc[order].reindex(columns=cols)
 fig,ax=plt.subplots(figsize=(7.5,7.8),constrained_layout=True);im=heatmap_cells(ax,matrix.to_numpy(),cmap='viridis',vmin=0,vmax=1);ax.set(xticks=range(10),xticklabels=[('G' if p=='GSR' else 'N')+'\n'+SHORT[m] for p,m in cols],yticks=range(25),yticklabels=[featlabel(x) for x in order]);ax.tick_params(labelsize=8);ax.axvline(4.5,color='white',lw=.8);fig.colorbar(im,ax=ax,label='Selection frequency over five training fits',shrink=.8);save(fig,num,name,False)
# S13 PLSC squared covariance fractions, with corrected probabilities visible.
plsc=pd.read_csv(STAT/'clinical_PLSC.csv');fig,axes=plt.subplots(2,2,figsize=(7.5,5.7),constrained_layout=True)
for row,block in enumerate(['ADOS_block','all_five_scores']):
 for col,pipeline in enumerate(['GSR','noGSR']):
  d=plsc[plsc.pipeline.eq(pipeline)&plsc.block.eq(block)];ax=axes[row,col];ax.bar(d.LV,d.squared_covariance_fraction,color=GROUP[1][1],alpha=.8)
  for r in d.itertuples():ax.text(r.LV,r.squared_covariance_fraction+.03,f'q={r.p_BH_exploratory:.3f}',ha='center',fontsize=8)
  ax.set(title=f'{pipeline}: '+('ADOS, n=43' if row==0 else 'Five scores, n=21'),xticks=d.LV,xlabel='Latent variable',ylabel='Squared covariance fraction',ylim=(0,1.05))
save(fig,13,'Exploratory PLSC diagnostics',False)
pd.DataFrame(manifest).to_csv(OUT/'FIGURE_MANIFEST.csv',index=False)
(OUT/'COLOR_CONVENTIONS.json').write_text(json.dumps({'ASD':'#DA005D','Control':'#00473A','positive_H_correlation':'#d62728','negative_H_correlation':'#1f77b4','model_colors':MC,'font':'Arial','main_figure_dpi':600,'note':'Group colors match original ml-gsr-nogsr-rerun.ipynb. State colors in the conceptual diagram denote mathematical labels, not diagnoses.'},indent=2)+'\n')
print('Built 11 main figures with RGB LZW TIFFs, 13 supporting figures, and 9 main tables.')
