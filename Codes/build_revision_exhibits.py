"""Build revision exhibits from verified outputs; never modifies submitted paper."""
from pathlib import Path
import shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from classification_output_source import classification_statistics_directory
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'revision/revised_paper'
VER = ROOT/'revision/stage2_verified'
STAT = ROOT/'results/abide_stage2_repaired/statistics'

def table(frame, filename, caption, label):
    text = frame.to_latex(index=False, escape=True, float_format=lambda x: f'{x:.3f}')
    (OUT/'tables'/filename).write_text('\\begin{table}[htbp]\n\\centering\n\\small\n'+f'\\caption{{{caption}}}\\label{{{label}}}\n'+text+'\\end{table}\n')

demo = pd.read_csv(VER/'primary_model_demographics.csv')
rows = []
for r in demo.to_dict('records'):
    rows.append({'Group':r['group'],'N':r['n'],'Age (years)':f"{r['AGE_AT_SCAN_mean']:.2f} ({r['AGE_AT_SCAN_SD']:.2f})",'FIQ':f"{r['FIQ_mean']:.2f} ({r['FIQ_SD']:.2f})",'Mean FD (mm)':f"{r['func_mean_fd_mean']:.3f} ({r['func_mean_fd_SD']:.3f})"})
table(pd.DataFrame(rows),'demographics.tex','Primary adjusted-model participants. Continuous variables are mean (SD). All participants were recorded male.','tab:demographics')
h = pd.read_csv(VER/'Hamiltonian_all_sensitivities.csv')
rows = []
for r in h[h.design.eq('primary_common_106')].to_dict('records'):
    rows.append({'Pipeline':r['pipeline'],'Model':{'raw_adjusted':'Raw adjusted','ComBat_sensitivity_adjusted':'ComBat sensitivity','energy_trim_sensitivity':'Pooled trim'}[r['model']],'N':int(r['n']),'ASD coefficient':r['beta'],'95% CI':f"{r['CI95_low']:.2f}, {r['CI95_high']:.2f}",'Holm p':r['p_adjusted']})
# Percent signs are escaped by pandas; use a plain header here.
frame=pd.DataFrame(rows).rename(columns={'95\\% CI':'95% CI'})
table(frame,'primary_H.tex','Adjusted diagnosis coefficients for the 106-region Hamiltonian, with HC3 confidence intervals. Holm correction covers both pipelines within each model version.','tab:primary')
c=pd.read_csv(classification_statistics_directory()/'classification_conditional_intervals.csv')
rows=[]
for r in c.to_dict('records'):
    rows.append({'Pipeline':r['pipeline'],'Classifier':{'SVM_RBF':'RBF SVM','LogisticRegression':'Logistic regression','GaussianNB':'Gaussian naive Bayes','KNN':'K-nearest neighbors','XGBoost':'XGBoost'}[r['model']],'BA':r['balanced_accuracy'],'95% CI':f"{r['conditional_site_bootstrap_CI95_low']:.3f}, {r['conditional_site_bootstrap_CI95_high']:.3f}"})
table(pd.DataFrame(rows),'classification.tex','Pooled out-of-fold balanced accuracy (BA) for all 140 participants. Site-bootstrap intervals condition on fixed predictions and omit some training uncertainty.','tab:classification')
site=pd.read_csv(VER/'primary_model_site_counts.csv')
table(site,'site_counts.tex','Primary adjusted-model site counts. CALTECH contributes only ASD participants.','tab:Ssites')
q=pd.read_csv(VER/'diagnostics/site_diagnostics.csv')
q=q[q.design.eq('primary_common_106')][['pipeline','version','site_partial_R2_descriptive','max_leverage','max_Cooks_distance']]
q.columns=['Pipeline','Version','Site partial R2','Max leverage',"Max Cook's D"]
table(q,'site_diagnostics.tex','Primary-model diagnostics. Site partial $R^2$ is descriptive and conditional on diagnosis, age, FIQ and mean FD. The design is identical before and after ComBat.','tab:Sdiag')
p=pd.read_csv(STAT/'clinical_PLSR_site_heldout.csv')[['pipeline','score','n','OOF_R2','OOF_MSE','training_mean_baseline_MSE']]
p.columns=['Pipeline','Score','N','OOF R2','OOF MSE','Baseline MSE']
p.Score=p.Score.str.replace('_',' ')
table(p,'clinical_plsr.tex','Nested site-held-out PLS regression. Baselines predict the outer-fold training mean. Negative OOF $R^2$ does not imply evidence of an inverse biological effect.','tab:Splsr')
null=pd.read_csv(STAT/'participant_null_scores.csv')
ns=null.groupby(['pipeline','null']).agg(N=('SUB_ID','count'),mean_z=('z','mean'),SD_z=('z','std')).reset_index()
ns['null']=ns['null'].replace({'node_permutation':'Unrestricted','network_hemisphere_blocks':'Network/hemisphere'})
ns.columns=['Pipeline','Null','N','Mean z','SD z']
table(ns,'null_summary.tex','Descriptive participant null-score summaries. These are alignment randomizations, not diagnostic tests or full spatial-autocorrelation nulls.','tab:Snull')
fig, ax=plt.subplots(figsize=(7,5.3))
ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
boxes=[(.5,.89,'Original requested pool: 182\n90 ASD / 92 controls'),(.5,.65,'Age 18–30; observed FIQ >80: 140\n69 ASD / 71 controls; 12 sites'),(.5,.41,'Fixed 106 common ROIs: features for all 140\nNonconstant series; ≥50% support in each retained ROI'),(.5,.15,'Adjusted model: 136 (67 ASD / 69 controls); 9 sites\n4 excluded by site N<3 rule\nExploratory classification retains all 140')]
for x,y,label in boxes:
    ax.add_patch(FancyBboxPatch((x-.46,y-.085),.92,.17,boxstyle='round,pad=0.01',facecolor='#edf3f8',edgecolor='#36556b'))
    ax.text(x,y,label,ha='center',va='center',fontsize=10)
for y1,y2 in [(.80,.745),(.56,.505),(.32,.245)]: ax.annotate('',xy=(.5,y2),xytext=(.5,y1),arrowprops={'arrowstyle':'->'})
fig.savefig(OUT/'figures/participant_flow.pdf',bbox_inches='tight');fig.savefig(OUT/'figures/participant_flow.png',dpi=180,bbox_inches='tight');plt.close(fig)
shutil.copy2(VER/'primary_results.pdf',OUT/'figures/primary_results.pdf')
# Supplementary forest plot reports each design's own sample; raw estimates only.
raw=h[h.model.eq('raw_adjusted')].copy()
fig,ax=plt.subplots(figsize=(7,4.3),constrained_layout=True)
for i,r in enumerate(raw.itertuples()):
    ax.errorbar(r.beta,i,xerr=[[r.beta-r.CI95_low],[r.CI95_high-r.beta]],fmt='o',capsize=3)
ax.axvline(0,color='gray',ls='--');ax.set(yticks=range(len(raw)),yticklabels=[f"{r.design.replace('primary_common_','').replace('sensitivity_broader_','').replace('sensitivity_full','')} ROIs; {r.pipeline}; n={int(r.n)}" for r in raw.itertuples()],xlabel='Adjusted ASD coefficient (HC3 95% CI)')
fig.savefig(OUT/'figures/node_sensitivities.pdf');plt.close(fig)
print('Built revision exhibits:',OUT)
