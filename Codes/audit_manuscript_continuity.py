"""Restore original descriptive questions and export explicit significance ledgers."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from abide_stage2 import adjusted_pvalues, inferential_features, model_design, hc3_fit
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/abide_stage2_repaired'
STAT=SRC/'statistics'
OUT=ROOT/'revision/stage2_verified/continuity'
OUT.mkdir(parents=True,exist_ok=True)
metadata=pd.read_csv(SRC/'audit/demographic_manifest.csv')
features=pd.read_csv(SRC/'features/all_designs_raw_untrimmed.csv')
rows=[]
for pipeline,frame in features[features.design.eq('primary_common_106')].groupby('pipeline'):
    raw=frame.merge(metadata,on='SUB_ID',validate='one_to_one')
    ids=pd.read_csv(STAT/f'model_participants_primary_common_106_{pipeline}.csv')
    matched=raw[raw.SUB_ID.isin(ids.SUB_ID)]
    combat=pd.read_csv(STAT/f'ComBat_sensitivity_features_primary_common_106_{pipeline}.csv')
    assert set(matched.SUB_ID)==set(combat.SUB_ID)
    cols=[c for c in inferential_features(raw) if c!='Whole_Energy_global' and np.isfinite(raw[c]).all() and raw[c].std()>1e-12]
    assert len(cols)==87
    for version,table in [('raw_eligible',raw),('raw_model_subset',matched),('ComBat_model_subset',combat)]:
        for diagnosis,block in table.groupby('DX_GROUP'):
            for feature in cols:
                r,p=stats.pearsonr(block.Whole_Energy_global,block[feature])
                rho,sp=stats.spearmanr(block.Whole_Energy_global,block[feature])
                coupled=feature.endswith(('_AvgFalffW','_EnergyBlock')) or feature.startswith('Whole_Prop_')
                rows.append({'version':version,'pipeline':pipeline,'group':'ASD' if diagnosis==1 else 'Control',
                             'feature':feature,'n':len(block),'Pearson_r':r,'Pearson_p':p,'Spearman_rho':rho,'Spearman_p':sp,
                             'shares_state_weight_or_configuration_terms_with_H':coupled,
                             'interpretation':'descriptive within-group association; not independent validation or diagnosis specificity'})
table=pd.DataFrame(rows)
for _,block in table.groupby(['version','pipeline','group']):
    for estimator in ['Pearson','Spearman']:
        table.loc[block.index,estimator+'_q_per_group87']=adjusted_pvalues(block[estimator+'_p'])
for _,block in table.groupby('version'):
    for estimator in ['Pearson','Spearman']:
        table.loc[block.index,estimator+'_q_across348']=adjusted_pvalues(block[estimator+'_p'])
table.to_csv(OUT/'H_vs_all_metrics_complete.csv',index=False)
table[table.Pearson_q_per_group87.lt(.05)].to_csv(OUT/'H_vs_metrics_Pearson_significant_per_group.csv',index=False)
table[table.Pearson_q_across348.lt(.05)].to_csv(OUT/'H_vs_metrics_Pearson_significant_across_groups_pipelines.csv',index=False)
counts=table.groupby(['version','pipeline','group']).agg(n=('n','first'),tests=('feature','size'),
    Pearson_q87_below05=('Pearson_q_per_group87',lambda x:x.lt(.05).sum()),
    Pearson_q348_below05=('Pearson_q_across348',lambda x:x.lt(.05).sum()),
    Spearman_q87_below05=('Spearman_q_per_group87',lambda x:x.lt(.05).sum())).reset_index()
counts.to_csv(OUT/'H_correlation_significance_counts.csv',index=False)
# The original saved exports are retained separately from the submitted claims.
old=[]
for pipeline in ['GSR','noGSR']:
    frame=pd.read_csv(ROOT/f'Data/{pipeline}_results/statistical_comparison_{pipeline}.csv')
    for row in frame[frame.p_corrected.lt(.05)].to_dict('records'):
        old.append({'source_scope':'original supplied export; not necessarily submitted-manuscript run','pipeline':pipeline,**row})
pd.DataFrame(old).to_csv(OUT/'original_export_significant_group_comparisons.csv',index=False)
oldh=[]
for pipeline in ['GSR','noGSR']:
    frame=pd.read_csv(ROOT/f'Data/{pipeline}_results/hamiltonian_vs_metrics_{pipeline}.csv')
    frame.insert(0,'pipeline',pipeline);oldh.append(frame)
oldh=pd.concat(oldh,ignore_index=True)
oldh[oldh.Pearson_p_FDR.lt(.05)].to_csv(OUT/'original_export_significant_H_correlations.csv',index=False)
# All corrected group families, nominal rows separately from corrected rows.
g=pd.read_csv(STAT/'group_models.csv');valid=g[g.status.eq('ok')]
valid[valid.p_adjusted.lt(.05)].to_csv(OUT/'revised_significant_group_comparisons.csv',index=False)
valid[valid.p.lt(.05)].sort_values(['design','model','family','pipeline','p']).to_csv(OUT/'revised_nominal_group_comparisons_NOT_corrected_significance.csv',index=False)
summary=valid.groupby(['design','model','family','pipeline']).agg(tests=('Feature','size'),n=('n','first'),minimum_p=('p','min'),minimum_adjusted_p=('p_adjusted','min'),nominal_p_below05=('p',lambda x:x.lt(.05).sum()),adjusted_p_below05=('p_adjusted',lambda x:x.lt(.05).sum())).reset_index()
summary.to_csv(OUT/'all_group_family_significance_summary.csv',index=False)
# Preserve all corrected secondary-outcome significance lists, even if empty.
for filename,columns in [('clinical_all_feature_correlations.csv',['Pearson_p_BH','Spearman_p_BH']),('clinical_H_partial_correlations.csv',['p_BH']),('clinical_PLSC.csv',['p_BH_exploratory']),('efficiency_partial_correlations.csv',['p_BH']),('group_by_efficiency_interactions.csv',['p_Holm'])]:
    frame=pd.read_csv(STAT/filename)
    for column in columns:
        frame[frame[column].lt(.05)].to_csv(OUT/f'significant_{Path(filename).stem}_{column}.csv',index=False)
# Both-diagnosis null summary, corrected over all participants/pipelines/nulls.
null=pd.read_csv(STAT/'participant_null_scores.csv').merge(metadata[['SUB_ID','DX_GROUP']],on='SUB_ID',validate='many_to_one')
null['group']=null.DX_GROUP.map({1:'ASD',2:'Control'})
null['q_BH_all560_null_comparisons']=adjusted_pvalues(null.p_lower_energy)
null.to_csv(OUT/'both_groups_null_comparisons.csv',index=False)
null[null.q_BH_all560_null_comparisons.lt(.05)].to_csv(OUT/'SIGNIFICANT_NULL_COMPARISONS_q560.csv',index=False)
ns=null.groupby(['group','pipeline','null']).agg(n=('SUB_ID','size'),mean_z=('z','mean'),nominal_p_below05=('p_lower_energy',lambda x:x.lt(.05).sum()),q560_below05=('q_BH_all560_null_comparisons',lambda x:x.lt(.05).sum())).reset_index()
ns.to_csv(OUT/'both_groups_null_summary.csv',index=False)
print(counts.to_string(index=False))
print('\nCorrected primary raw group family summaries:')
print(summary[summary.design.eq('primary_common_106')&summary.model.eq('raw_adjusted')].to_string(index=False))
print('\nWithin-group efficiency, unadjusted descriptive correlations:')
print(table[table.feature.str.contains('Efficiency')&table.version.eq('raw_eligible')][['pipeline','group','feature','n','Pearson_r','Pearson_p','Pearson_q_per_group87']].to_string(index=False))

# Numerically reproduce all original exported H correlations, including their FDR.
checks=[]
for pipeline in ['GSR','noGSR']:
    original=pd.read_csv(ROOT/f'Data/{pipeline}_results/features_ComBat_{pipeline}.csv')
    saved=pd.read_csv(ROOT/f'Data/{pipeline}_results/hamiltonian_vs_metrics_{pipeline}.csv')
    for group,block in saved.groupby('Group'):
        data=original[original.Group.eq('Control' if group=='TD' else group)]
        recalculated=[]
        for record in block.itertuples():
            r,p=stats.pearsonr(data.Whole_Energy_global,data[record.Network_Metric])
            recalculated.append({'feature':record.Network_Metric,'r':r,'p':p,'saved_r':record.Pearson_r,'saved_p':record.Pearson_p,'saved_q':record.Pearson_p_FDR})
        frame=pd.DataFrame(recalculated);frame['recomputed_q']=adjusted_pvalues(frame.p)
        assert np.allclose(frame.r,frame.saved_r,atol=1e-10)
        assert np.allclose(frame.p,frame.saved_p,atol=1e-10)
        assert np.allclose(frame.recomputed_q,frame.saved_q,atol=1e-10)
        frame.insert(0,'group',group);frame.insert(0,'pipeline',pipeline);frame['n_original_export']=len(data)
        checks.append(frame)
pd.concat(checks,ignore_index=True).to_csv(OUT/'original_H_correlations_numeric_reproduction.csv',index=False)
print('All original exported H correlations and per-group FDR numerically reproduced.')

legacy=[]
for pipeline,block in features[features.design.eq('primary_common_106')].groupby('pipeline'):
    ids=pd.read_csv(STAT/f'model_participants_primary_common_106_{pipeline}.csv')
    data=block[block.SUB_ID.isin(ids.SUB_ID)].merge(metadata,on='SUB_ID',validate='one_to_one')
    row,_=hc3_fit(data.Whole_Bipolarity_legacy_spectral,model_design(data))
    legacy.append({'pipeline':pipeline,'feature':'Whole_Bipolarity_legacy_spectral','n':len(data),'scope':'exploratory original descriptor; separate from frozen families',**row})
legacy=pd.DataFrame(legacy);legacy['p_Holm_exploratory']=adjusted_pvalues(legacy.p,'Holm')
legacy.to_csv(OUT/'legacy_bipolarity_exploratory_models.csv',index=False)
print(legacy[['pipeline','beta','CI95_low','CI95_high','p','p_Holm_exploratory']].to_string(index=False))
