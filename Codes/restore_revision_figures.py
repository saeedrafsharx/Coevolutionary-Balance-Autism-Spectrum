"""Restore scientific figures using corrected outputs; original images stay intact."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from abide_stage2 import model_design
from classification_output_source import classification_statistics_directory
from plot_heatmap_cells import heatmap_cells
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'revision/revised_paper'
FIG=OUT/'figures'
STAT=ROOT/'results/abide_stage2_repaired/statistics'
META=pd.read_csv(ROOT/'results/abide_stage2_repaired/audit/demographic_manifest.csv')
FEATURES=pd.read_csv(ROOT/'results/abide_stage2_repaired/features/all_designs_raw_untrimmed.csv')
PRIMARY=FEATURES[FEATURES.design.eq('primary_common_106')].merge(META,on='SUB_ID',validate='many_to_one')
COLORS={'ASD':'#b4437a','Control':'#247c85'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
def latex_table(frame,name,caption,label):
    (OUT/'tables'/name).write_text('\\begin{table}[htbp]\n\\centering\\small\n'+f'\\caption{{{caption}}}\\label{{{label}}}\n'+frame.to_latex(index=False,escape=True,float_format=lambda x:f'{x:.2f}')+'\\end{table}\n')
shutil.copy2(ROOT/'Images/figure_configurations_v2.png',FIG/'pair_configurations.png')
controls=pd.read_csv(ROOT/'revision/stage2_verified/control_nulls/control_participant_null_comparisons.csv')
cs=pd.read_csv(ROOT/'revision/stage2_verified/control_nulls/control_only_null_summary.csv')
nulls=['node_permutation','network_hemisphere_blocks']
labels={'node_permutation':'Unrestricted states','network_hemisphere_blocks':'Network/hemisphere states'}
fig,axes=plt.subplots(2,2,figsize=(9,7.4),constrained_layout=True)
for col,pipeline in enumerate(['GSR','noGSR']):
    for row,scheme in enumerate(nulls):
        ax=axes[row,col];frame=controls[controls.pipeline.eq(pipeline)&controls['null'].eq(scheme)]
        ax.plot(np.array([0,1]),np.vstack([frame.observed_H,frame.null_mean]),color='#9bafb2',alpha=.3,lw=.65)
        for x,field in enumerate(['observed_H','null_mean']):
            ax.scatter(np.repeat(x,len(frame)),frame[field],s=12,alpha=.65,color=['#247c85','#999999'][x],zorder=3)
            ax.scatter(x,frame[field].mean(),s=65,marker='D',color='black',zorder=4)
        ax.set(xticks=[0,1],xticklabels=['Empirical H','Own null mean H'],ylabel='Hamiltonian H',title=f'{pipeline}: {labels[scheme]}\n71 male controls; 1,000 permutations each',xlim=(-.3,1.3))
save(fig,'control_null_empirical')
fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
for ax,scheme in zip(axes,nulls):
    for i,pipeline in enumerate(['GSR','noGSR']):
        frame=controls[controls.pipeline.eq(pipeline)&controls['null'].eq(scheme)]
        jitter=np.random.default_rng(101).uniform(-.12,.12,len(frame))
        ax.boxplot([frame.z],positions=[i],widths=.35,showfliers=False)
        ax.scatter(i+jitter,frame.z,s=13,alpha=.65,color='#247c85')
    ax.axhline(0,color='gray',ls='--');ax.set(xticks=[0,1],xticklabels=['GSR','noGSR'],ylabel='Participant null z-score',title=labels[scheme])
save(fig,'control_null_zscores')
t=cs.copy(); t['Null']=t['null'].map({'node_permutation':'Unrestricted','network_hemisphere_blocks':'Blocked'})
t=t[['pipeline','Null','n_controls','mean_observed_H','mean_null_H','mean_z','n_individual_BH_q_below_05']]
t.columns=['Pipeline','Null','N','Mean H','Mean null H','Mean z','q<.05 count']
latex_table(t,'control_null_summary.tex','Male-control baseline against state-permutation null configurations. All 71 eligible controls are included. Individual BH correction covers 284 control/pipeline/null tests; counts do not constitute a diagnosis comparison.','tab:controlnull')
# All seven global quantities: untrimmed raw distributions, no p-value selection.
fields={'Whole_Energy_global':'Hamiltonian H','Whole_GlobalEfficiency':'Global efficiency','Whole_LocalEfficiency':'Local efficiency','Whole_Prop_agreement':'Same states / positive FC','Whole_Prop_disagreement':'Opposite states / negative FC','Whole_Prop_imbalanced_same':'Same states / negative FC','Whole_Prop_imbalanced_opp':'Opposite states / positive FC'}
fig,axes=plt.subplots(2,4,figsize=(12,6.5),constrained_layout=True)
for ax,(field,title) in zip(axes.flat,fields.items()):
    for p,pipeline in enumerate(['GSR','noGSR']):
        for g,(code,group) in enumerate([(1,'ASD'),(2,'Control')]):
            values=PRIMARY.loc[PRIMARY.pipeline.eq(pipeline)&PRIMARY.DX_GROUP.eq(code),field]
            pos=p*3+g
            ax.boxplot([values],positions=[pos],widths=.6,showfliers=False,medianprops={'color':'black'})
            ax.scatter(pos+np.random.default_rng(101).uniform(-.2,.2,len(values)),values,s=5,alpha=.4,color=COLORS[group])
    ax.set(xticks=[.5,3.5],xticklabels=['GSR','noGSR'],title=title)
axes.flat[-1].axis('off');axes.flat[-1].text(.1,.6,'ASD: magenta (n=69)\nControls: teal (n=71)\nUntrimmed raw features\nIdentical 106-node graph',linespacing=1.8)
save(fig,'global_feature_distributions')
# Complete cortical-family regression heatmaps; no feature chosen by p-value.
groups=pd.read_csv(STAT/'group_models.csv')
groups=groups[groups.design.eq('primary_common_106')&groups.model.eq('raw_adjusted')]
nets=['Visual','SomatoMotor','DorsalAttn','Salience_VentAttn','Limbic','Frontoparietal','Default']
short={'Visual':'Visual','SomatoMotor':'Somatomotor','DorsalAttn':'Dorsal attention','Salience_VentAttn':'Salience/ventral attention','Limbic':'Limbic','Frontoparietal':'Frontoparietal','Default':'Default'}
for family,name in [('Intra','intra_network_effects'),('Inter','inter_network_effects')]:
    blocks=[(n,n) for n in nets] if family=='Intra' else [(a,b) for i,a in enumerate(nets) for b in nets[i+1:]]
    matrix=np.full((len(blocks),6),np.nan)
    for i,(a,b) in enumerate(blocks):
        for j,(pipeline,suffix) in enumerate([(p,s) for p in ['GSR','noGSR'] for s in ['AvgWeight','AvgFalffW','EnergyBlock']]):
            feature=f'Intra_{a}_{suffix}' if family=='Intra' else f'Inter_{a}__{b}_{suffix}'
            found=groups[groups.pipeline.eq(pipeline)&groups.Feature.eq(feature)&groups.status.eq('ok')]
            if len(found):matrix[i,j]=float(found.beta.iloc[0]/found.SE_HC3.iloc[0])
    fig,ax=plt.subplots(figsize=(9,4.5 if family=='Intra' else 9.5),constrained_layout=True)
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#d9d9d9')
    im=heatmap_cells(ax,matrix,cmap=cmap,vmin=-3,vmax=3)
    ax.set(xticks=range(6),xticklabels=['GSR\nMean FC','GSR\nState-FC','GSR\nEnergy','noGSR\nMean FC','noGSR\nState-FC','noGSR\nEnergy'],yticks=range(len(blocks)),yticklabels=[short[a] if a==b else short[a]+' — '+short[b] for a,b in blocks],title=f'{family}-network adjusted diagnosis effects\nAll specified tests; none survive family BH correction')
    ax.axvline(2.5,color='black',lw=.7);fig.colorbar(im,ax=ax,label='Diagnosis coefficient / HC3 SE (t statistic)',shrink=.8)
    save(fig,name)
# Residualized efficiency scatter: reproduces exact saved within-group analyses.
saved=pd.read_csv(STAT/'efficiency_partial_correlations.csv')
fig,axes=plt.subplots(2,2,figsize=(9,7),constrained_layout=True)
for col,pipeline in enumerate(['GSR','noGSR']):
    data=PRIMARY[PRIMARY.pipeline.eq(pipeline)].copy();data=data[data.SITE_ID.map(data.SITE_ID.value_counts()).ge(3)]
    for row,(code,group) in enumerate([(1,'ASD'),(2,'Control')]):
        block=data[data.DX_GROUP.eq(code)].copy();block=block[block.SITE_ID.map(block.SITE_ID.value_counts()).ge(2)].reset_index(drop=True)
        x=model_design(block,include_group=False);x['mean_FC']=block.QC_MeanFisherZ_FC
        residual=lambda y:np.asarray(y)-x.to_numpy()@np.linalg.lstsq(x,y,rcond=None)[0]
        a=residual(block.Whole_GlobalEfficiency);b=residual(block.Whole_Energy_global)
        r,p=stats.pearsonr(a,b)
        expected=saved[saved.pipeline.eq(pipeline)&saved.DX_GROUP.eq(code)&saved.feature.eq('Whole_GlobalEfficiency')].iloc[0]
        assert len(block)==expected.n and np.isclose(r,expected.partial_r)
        ax=axes[row,col];ax.scatter(a,b,s=18,alpha=.65,color=COLORS[group]);xx=np.linspace(a.min(),a.max(),100);ax.plot(xx,np.polyval(np.polyfit(a,b,1),xx),color='gray')
        ax.set(xlabel='Global efficiency residual',ylabel='H residual',title=f'{pipeline}: {group}; n={len(block)}\nPartial r={r:.3f}; BH q={expected.p_BH:.3f}')
save(fig,'adjusted_efficiency')
# Clinical partial H estimates and PLS regression validation include all outcomes.
clinical=pd.read_csv(STAT/'clinical_H_partial_correlations.csv');plsr=pd.read_csv(STAT/'clinical_PLSR_site_heldout.csv')
scores=list(plsr.score.unique())
fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
for offset,pipeline in [(-.15,'GSR'),(.15,'noGSR')]:
    a=clinical[clinical.pipeline.eq(pipeline)].set_index('score').loc[scores]
    b=plsr[plsr.pipeline.eq(pipeline)].set_index('score').loc[scores]
    axes[0].scatter(a.partial_r,np.arange(5)+offset,label=pipeline,s=35)
    axes[1].barh(np.arange(5)+offset,b.OOF_R2,height=.28,label=pipeline)
for ax in axes:ax.set(yticks=range(5),yticklabels=[s.replace('_',' ') for s in scores]);ax.axvline(0,color='gray',ls='--');ax.legend()
axes[0].set(xlabel='Adjusted H–clinical partial correlation',title='All five outcomes; no BH-significant association')
axes[1].set(xlabel='Site-held-out OOF R²',title='PLSR: all models worse than training-mean baseline')
save(fig,'clinical_validation')
# ROC and confusion summaries use identical complete OOF predictions, no best-model selection.
pred=pd.read_csv(classification_statistics_directory()/'classification_OOF_predictions.csv')
model_names=[m for m in ['LogisticRegression','SVM_RBF','GaussianNB','KNN','XGBoost'] if m in set(pred.model)]
fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
fig2,axs=plt.subplots(len(model_names),2,figsize=(8,2.8*len(model_names)),constrained_layout=True)
for col,pipeline in enumerate(['GSR','noGSR']):
    for row,model in enumerate(model_names):
        frame=pred[pred.pipeline.eq(pipeline)&pred.model.eq(model)]
        assert len(frame)==140 and frame.SUB_ID.is_unique
        fpr,tpr,_=roc_curve(frame.y_true,frame.decision_score)
        axes[col].plot(fpr,tpr,label=f'{model}: AUC={roc_auc_score(frame.y_true,frame.decision_score):.3f}')
        matrix=confusion_matrix(frame.y_true,frame.y_pred,labels=[0,1]);ax=axs[row,col];heatmap_cells(ax,matrix,cmap='Blues',vmin=0,vmax=71,aspect='equal')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>40 else 'black')
        ax.set(xticks=[0,1],xticklabels=['Control','ASD'],yticks=[0,1],yticklabels=['Control','ASD'],xlabel='Predicted',ylabel='Observed',title=f'{pipeline}: {model}')
    axes[col].plot([0,1],[0,1],'--',color='gray');axes[col].set(xlabel='False positive rate',ylabel='True positive rate',title=f'{pipeline}: pooled site-held-out OOF predictions');axes[col].legend(fontsize=8)
save(fig,'classification_ROC');save(fig2,'classification_confusion')
print('Restored control-null, conceptual, global, network, efficiency, clinical, and OOF prediction figures.')
