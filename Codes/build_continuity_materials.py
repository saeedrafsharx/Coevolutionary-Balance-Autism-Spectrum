"""Render original-question continuity records and restored descriptive H plots."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from classification_output_source import classification_statistics_directory
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'revision/stage2_verified/continuity'
PAPER=ROOT/'revision/revised_paper'
STAT=ROOT/'results/abide_stage2_repaired/statistics'
h=pd.read_csv(DATA/'H_vs_all_metrics_complete.csv')
raw=h[h.version.eq('raw_eligible')]
raw[raw.Pearson_q_per_group87.lt(.05)].sort_values(['pipeline','group','Pearson_p']).to_csv(DATA/'SIGNIFICANT_REVISED_RAW_H_ASSOCIATIONS_59.csv',index=False)
# Compact labels preserve the original top-20 descriptive presentation.
rename={'Salience_VentAttn':'SV','Frontoparietal':'FP','SomatoMotor':'SM','DorsalAttn':'DA','Default':'DMN','Visual':'VIS','Limbic':'LIM','AvgFalffW':'F','EnergyBlock':'E','AvgWeight':'W','Whole_GlobalEfficiency':'Global efficiency','Whole_LocalEfficiency':'Local efficiency','Whole_Prop_':'Prop. ','imbalanced_same':'imbalanced same','imbalanced_opp':'imbalanced opposite','Intra_':'Within ','Inter_':'Between '}
def label(feature):
    for a,b in rename.items():feature=feature.replace(a,b)
    return feature.replace('__','–').replace('_',' ')
plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(10,10),constrained_layout=True)
for col,pipeline in enumerate(['GSR','noGSR']):
 for row,group in enumerate(['ASD','Control']):
    frame=raw[raw.pipeline.eq(pipeline)&raw.group.eq(group)].copy();frame=frame.loc[frame.Pearson_r.abs().nlargest(20).index].sort_values('Pearson_r')
    ax=axes[row,col];colors=np.where(frame.Pearson_r>0,'#bb4e4e','#3b78a0')
    ax.barh(range(len(frame)),frame.Pearson_r,color=colors)
    ax.set(yticks=range(len(frame)),yticklabels=[label(r.feature)+(' *' if r.Pearson_q_per_group87<.05 else '') for r in frame.itertuples()],xlabel='Pearson r with H',xlim=(-.65,.65),title=f'{pipeline}: {group}; n={int(frame.n.iloc[0])}')
    ax.tick_params(axis='y',labelsize=7);ax.axvline(0,color='gray',lw=.7)
fig.savefig(PAPER/'figures/H_correlates_raw.pdf',bbox_inches='tight');fig.savefig(PAPER/'figures/H_correlates_raw.png',dpi=180,bbox_inches='tight');plt.close(fig)
# Complete tables are readable CSVs; manuscript summaries retain fixed families.
def latex(frame,name,caption,tag):
 (PAPER/'tables'/name).write_text('\\begin{table}[htbp]\n\\centering\\small\n'+f'\\caption{{{caption}}}\\label{{{tag}}}\n'+frame.to_latex(index=False,escape=True,float_format=lambda x:f'{x:.3f}')+'\\end{table}\n')
counts=pd.read_csv(DATA/'H_correlation_significance_counts.csv');t=counts[counts.version.eq('raw_eligible')][['pipeline','group','n','tests','Pearson_q87_below05','Spearman_q87_below05']];t.columns=['Pipeline','Group','N','Tests','Pearson q<.05','Spearman q<.05']
latex(t,'H_correlation_counts.tex','Within-group H associations in all eligible participants. BH correction is applied over 87 features separately per pipeline/group and estimator, following the original descriptive Pearson family convention. These counts are not independent biological findings or diagnosis comparisons.','tab:Hcorrelations')
legacy=pd.read_csv(DATA/'legacy_bipolarity_exploratory_models.csv');t=legacy[['pipeline','n','beta','CI95_low','CI95_high','p_Holm_exploratory']].copy();t.columns=['Pipeline','N','Coefficient','CI low','CI high','Holm p']
for column in ['Coefficient','CI low','CI high']:t[column]=t[column].map(lambda value:f'{value:.4f}')
latex(t,'legacy_bipolarity.tex','Original spectral bipolarity descriptor: additional exploratory raw adjusted diagnosis models. Holm covers the two pipeline tests, separately from the frozen inferential families. The descriptor is not asserted to optimize signed structural balance.','tab:Slegacy')
# Full original-to-revision question mapping; include genuine differences explicitly.
maprows=[
('Empirical H versus null configurations','Results: Null-model validation','Present for all participants; control-only baseline added','Same question; changed randomization: fixed FC, states only. Original text also described shuffled weights.','main null results; continuity/both_groups_null_summary.csv'),
('Global H, four motif proportions and two efficiencies','Results: Whole-brain coevolutionary energy and motif composition','Present: 7 global quantities; raw adjusted and ComBat/trim sensitivities','Same quantities; 106-node coverage scope and sample differ.','statistics/group_models.csv'),
('Spectral bipolarity','Whole-brain metrics; no-GSR sensitivity','Restored as separate exploratory adjusted comparison','Raw descriptor was retained; newly reported tests remain outside frozen families.','continuity/legacy_bipolarity_exploratory_models.csv'),
('Within-network group comparisons','Results: Network-specific coevolutionary metrics','Present: 3 measures across seven named networks; 18 valid tests per pipeline','One primary limbic node makes 3 within-limbic features unavailable.','main network-specific subsection; intra-network effect figure'),
('Between-network group comparisons','Results: Inter-network connectivity and energy','Present: 21 network pairs x3 measures per pipeline','Same measures; anatomical assignments, coverage and covariate models revised.','main inter-network subsection; inter-network effect figure'),
('Clinical Pearson and Spearman correlations','Results: Correlation analysis with behavioral scores','Present for all five scores; common 88 features','BH pooled over pipelines; samples/outcomes and missingness explicitly recorded.','statistics/clinical_all_feature_correlations.csv'),
('PLS regression','Results: Multivariate analysis','Present with nested site-held-out predictions','Up to three components; new validation scheme; old component-loading plots not reused.','statistics/clinical_PLSR_site_heldout.csv'),
('PLS correlation','Results: Multivariate analysis','Present: ADOS and five-score complete-case blocks','Nuisance residualization and within-site permutations; 1000 versus original 5000 permutations.','statistics/clinical_PLSC.csv'),
('Within-group H versus all network metrics','No-GSR sensitivity; supporting H-correlation figures','Restored: raw eligible, matched raw and ComBat versions; all 87 other available features','Previously compressed to adjusted efficiency; restored as separate descriptive complement.','continuity/H_vs_all_metrics_complete.csv'),
('H versus global/local efficiency and mean connectivity','No-GSR sensitivity; within-group framework discussion','Present unadjusted; adjusted efficiency and diagnosis interaction also reported','Raw associations and mean-FC/FD-adjusted tests answer different questions.','continuity/H_vs_all_metrics_complete.csv; statistics/efficiency_partial_correlations.csv'),
('Supervised ASD-control classification','Results: Machine learning; no-GSR sensitivity','Present, but only logistic regression and RBF SVM are rerun','Original also used GaussianNB, KNN, XGBoost; those three not rerun. No identical-model claim.','statistics/classification_scope.json'),
('Feature importance and PCA','Classification and Supporting information','Not current inferential analyses; old plots preserved only','Original fitted-model importance is not a new nested-model importance estimate; PCA is descriptive.','FIGURE_INVENTORY.csv'),
('GSR versus no-GSR sensitivity','Results: Sensitivity analysis without GSR','Present for all primary and complementary feature analyses','Same primary people/nodes permit paired pipeline interpretation; no direct pipeline-by-diagnosis test is claimed.','all tables have pipeline field'),
('Raw versus harmonized and trimming sensitivity','Discussion: Effect of harmonization; Methods: IQR removal','Present with checked ComBat and untrimmed primary models','Harmonization alone cannot explain changes when cohort/nodes/fALFF also change.','statistics/ComBat_diagnostics.csv; group_models.csv'),
('Signed modularity; sample matching','Notebook add-ons, not a reported manuscript result','Signed modularity not rerun; primary pipelines already share 140 IDs','Distinguish notebook-only experiments from manuscript findings; local FC arrays unavailable.','original notebook add-on cells; frozen design manifests')]
mapping=pd.DataFrame(maprows,columns=['original_question','original_location','revision_status','comparability_limit','evidence'])
classification=pd.read_csv(classification_statistics_directory()/'classification_conditional_intervals.csv')
completed_models=sorted(classification.model.unique())
pending_models=sorted(set(['LogisticRegression','SVM_RBF','GaussianNB','KNN','XGBoost'])-set(completed_models))
mask=mapping.original_question.eq('Supervised ASD-control classification')
mapping.loc[mask,'revision_status']='Completed algorithm types: '+', '.join(completed_models)
mapping.loc[mask,'comparability_limit']=('All five original algorithm types restored; new features, folds and validation remain different.' if not pending_models else 'Pending algorithm types: '+', '.join(pending_models)+'; not substituted or claimed complete.')
mapping.to_csv(DATA/'ORIGINAL_TO_REVISION_ANALYSIS_MAP.csv',index=False)
# Human-readable audit with full significant revised raw list, not selected examples.
text='''# Original-paper continuity and significance audit

The study's central questions are preserved: empirical-versus-null organization,
ASD-control contrasts at global, intra-network and inter-network levels,
within-group H relationships, clinical associations, PLS and prediction, with
GSR/no-GSR sensitivity. This is not a declaration that the original findings were
false. A reproducible association in one feature/cohort definition can differ in
another. Changed estimates must be described with their actual inputs.

## Original findings reproduced from the supplied exports

All 364 exported original H-versus-feature Pearson correlations, raw probabilities,
and per-group BH probabilities were numerically reproduced from the original
feature tables to absolute tolerance 1e-10. In particular, original no-GSR ASD
H-global efficiency is r=-0.324338, p=.003334, q=.005917 (n=80), and local
efficiency is r=-0.324319, q=.005917. These associations exist in the supplied
original feature tables. The submitted manuscript's stated sample and some values
differ from those exports; reproduction of an export does not establish that it
is the exact submitted run or establish an ASD-specific group interaction.

The original manuscript itself reported no corrected global, within-network or
between-network diagnosis effects. Its significant claims primarily concerned
null comparisons and within-group H correlations. The original GSR group export
separately contains three BH-significant inter-network mean-FC effects that the
submitted text reports differently; they are preserved in
`original_export_significant_group_comparisons.csv`, not relabeled as revision results.

## Revised significant comparisons: keep questions separate

- **Global, intra-network and inter-network ASD-control contrasts:** zero corrected
  effects in the frozen group tables (raw, ComBat and node/trimming sensitivities).
  The restored exploratory spectral-bipolarity contrast also does not survive
  Holm (both adjusted probabilities .916).
- **Empirical versus null:** participant-level lower-tail findings occur in both
  diagnoses. `both_groups_null_summary.csv` corrects all 560 participant/pipeline/
  null comparisons together. The previously reported control-only table keeps
  its separate explicitly labeled 284-test correction. Neither tests diagnosis.
- **H versus other metrics:** the raw-eligible descriptive complement has 59
  Pearson associations surviving original-style BH over 87 features per group/
  pipeline: 26 GSR-ASD, 21 GSR-control, 12 no-GSR-ASD, zero no-GSR-control.
  Full details follow below. These include mathematically linked energy/state-FC
  terms and are not 59 independent biological effects. Matched raw/ComBat and
  Spearman results are provided without choosing whichever version is significant.
- **Adjusted H-efficiency, efficiency interactions, clinical correlations and
  PLSC:** no corrected significant tests. Negative PLSR R2 and classifier intervals
  including .5 are predictive outcomes, not formal tests proving chance performance.

## Nominal group differences are not corrected findings

The primary raw model has two nominal inter-network terms in GSR and two nominal
within-network terms without GSR. They are listed with exact beta, intervals,
p and q in `revised_nominal_group_comparisons_NOT_corrected_significance.csv`.
None survives the frozen correction. Reporting them as corrected discoveries
would change the interpretation; omitting their existence would also be incomplete.

## Completeness and comparability

`ORIGINAL_TO_REVISION_ANALYSIS_MAP.csv` inventories every original analysis type.
Intra/inter tests already existed computationally but their dedicated main Results
sections were compressed too much. H-versus-all-feature correlation was genuinely
missing from the earlier revision presentation and is now restored. Original
spectral bipolarity is reported as a separate exploratory descriptor. Original
five-model classification is only partially matched (two models rerun); old
importance/loadings cannot be reused as current fitted-model results. Null schemes,
participant eligibility, node coverage, anatomical assignments, fALFF inputs,
covariate adjustment, correction scopes, and validation differ and remain explicit.
Pooled versus per-group BH can change individual discoveries in either direction;
the pooled alternative is not uniformly stricter in every subgroup.

## Full significant revised raw H associations

Scope: all 140 eligible participants, raw 106-node features, Pearson BH separately
over 87 tests per diagnosis/pipeline. These are descriptive associations, not
ASD-control differences. CSV: `SIGNIFICANT_REVISED_RAW_H_ASSOCIATIONS_59.csv`.

| Pipeline | Group | N | Feature correlated with H | r | BH q |
|---|---|---:|---|---:|---:|
'''
for r in raw[raw.Pearson_q_per_group87.lt(.05)].sort_values(['pipeline','group','Pearson_p']).itertuples():
 text+=f'| {r.pipeline} | {r.group} | {r.n} | {r.feature} | {r.Pearson_r:.4f} | {r.Pearson_q_per_group87:.6g} |\n'
claims=pd.DataFrame([
 {'original_claim':'ASD empirical H versus null mean','manuscript_probability':'t-test p=7.5e-5; Wilcoxon p=9.1e-5','verification_scope':'Submitted claim; original subject-level null ensemble unavailable here'},
 {'original_claim':'Control empirical H versus null mean','manuscript_probability':'t-test p=1.2e-4; Wilcoxon p=1.6e-4','verification_scope':'Submitted claim; original subject-level null ensemble unavailable here'},
 {'original_claim':'Within-ASD no-GSR H versus global/local efficiency','manuscript_probability':'FDR p approximately .005','verification_scope':'Original supplied export reproduced: r approximately -.324; q=.005917; n=80'},
 {'original_claim':'Within-ASD H versus block state-weighted FC and energy','manuscript_probability':'Strongest FDR p below 1e-7','verification_scope':'All original exported H-feature r, p and q reproduced; scope differs from revision'},
 {'original_claim':'Within-ASD no-GSR H versus selected block mean FC','manuscript_probability':'FDR p approximately .009','verification_scope':'Original-export full significant list retained; do not equate to a group interaction'},
 {'original_claim':'Earlier pre-harmonization GSR diagnosis-energy contrast','manuscript_probability':'FDR p approximately .002','verification_scope':'Historical earlier-run claim in Discussion; exact run not identified by supplied current exports'}])
claims.to_csv(DATA/'ORIGINAL_MANUSCRIPT_SIGNIFICANT_CLAIMS.csv',index=False)
text+='\n## Original manuscript positive claims and verification scope\n\n| Original claim | Reported probability | Verification scope |\n|---|---|---|\n'
for row in claims.itertuples():
 text+=f'| {row.original_claim} | {row.manuscript_probability} | {row.verification_scope} |\n'
text+='\n## Significant historical group-export comparisons (separate source scope)\n\nThese three rows are significant in the supplied original GSR export, not in the revised analysis. The submitted prose reports different corrected probabilities. Their source run must remain explicit.\n\n| Pipeline | Original exported feature | Raw p | BH q |\n|---|---|---:|---:|\n'
for row in pd.read_csv(DATA/'original_export_significant_group_comparisons.csv').itertuples():
 text+=f'| {row.pipeline} | {row.Feature} | {row.p_value:.6g} | {row.p_corrected:.6g} |\n'
text=text.replace('Original\nfive-model classification is only partially matched (two models rerun); old', 'Original\nfive-model classification now has '+str(len(completed_models))+' completed algorithm types ('+', '.join(completed_models)+'). '+('Pending: '+', '.join(pending_models)+'. ' if pending_models else 'All five are now complete. ')+'Old')
(DATA/'CONTINUITY_AND_SIGNIFICANT_COMPARISONS.md').write_text(text)
print('Wrote analysis map, complete significance report, and restored H-correlation figure.')
