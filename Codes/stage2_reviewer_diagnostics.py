"""Reviewer-requested descriptive diagnostics; preserves frozen primary outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from abide_stage2 import model_design, hc3_fit, adjusted_pvalues

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'results/abide_stage2_repaired'
OUT = ROOT / 'revision/stage2_verified/diagnostics'
OUT.mkdir(parents=True, exist_ok=True)
metadata = pd.read_csv(SRC / 'audit/demographic_manifest.csv')
features = pd.read_csv(SRC / 'features/all_designs_raw_untrimmed.csv')
summary, sites, influence, restricted, excluded = [], [], [], [], []
for (design, pipeline), raw in features.groupby(['design', 'pipeline']):
    ids = pd.read_csv(SRC / f'statistics/model_participants_{design}_{pipeline}.csv')
    raw = ids[['SUB_ID']].merge(raw, validate='one_to_one').merge(metadata, on='SUB_ID', validate='one_to_one')
    combat = pd.read_csv(SRC / f'statistics/ComBat_sensitivity_features_{design}_{pipeline}.csv')
    assert set(raw.SUB_ID) == set(combat.SUB_ID)
    for version, data in [('raw', raw), ('ComBat', combat)]:
        x = model_design(data)
        y = data.Whole_Energy_global.to_numpy()
        fit = sm.OLS(y, x).fit()
        reduced = x.loc[:, ~x.columns.str.startswith('site_')]
        biofit = sm.OLS(y, reduced).fit()
        residual = biofit.resid.to_numpy()
        partial_r2 = (biofit.ssr-fit.ssr)/biofit.ssr
        residual_groups = [residual[data.SITE_ID.to_numpy() == site] for site in sorted(data.SITE_ID.unique())]
        levene = stats.levene(*residual_groups, center='median')
        diagnostic = fit.get_influence()
        summary.append({'design':design, 'pipeline':pipeline, 'version':version, 'n':len(data),
                        'sites':data.SITE_ID.nunique(), 'site_partial_R2_descriptive':partial_r2,
                        'Levene_statistic_descriptive':levene.statistic, 'Levene_p_descriptive':levene.pvalue,
                        'design_rank':np.linalg.matrix_rank(x), 'design_columns':len(x.columns),
                        'max_leverage':diagnostic.hat_matrix_diag.max(), 'max_Cooks_distance':diagnostic.cooks_distance[0].max()})
        for (site, group), block in data.groupby(['SITE_ID','DX_GROUP']):
            sites.append({'design':design,'pipeline':pipeline,'version':version,'SITE_ID':site,
                          'group':'ASD' if group==1 else 'Control','n':len(block),
                          'H_mean':block.Whole_Energy_global.mean(),'H_SD':block.Whole_Energy_global.std()})
        for i, subject in enumerate(data.SUB_ID):
            influence.append({'design':design,'pipeline':pipeline,'version':version,'SUB_ID':subject,
                              'residual':float(fit.resid.iloc[i]),'fitted_H':float(fit.fittedvalues.iloc[i]),
                              'leverage':float(diagnostic.hat_matrix_diag[i]),
                              'Cooks_distance':float(diagnostic.cooks_distance[0][i])})
    counts = raw.groupby('SITE_ID').DX_GROUP.nunique()
    keep = raw.SITE_ID.isin(counts[counts.eq(2)].index)
    for row in raw.loc[~keep,['SUB_ID','SITE_ID','DX_GROUP']].to_dict('records'):
        excluded.append({'design':design,'pipeline':pipeline,'reason':'single_diagnosis_site',**row})
    data = raw.loc[keep].reset_index(drop=True)
    row, _ = hc3_fit(data.Whole_Energy_global, model_design(data))
    restricted.append({'design':design,'pipeline':pipeline,'model':'both_diagnosis_sites_raw_adjusted',
                       'n':len(data),'ASD_n':int(data.DX_GROUP.eq(1).sum()),
                       'Control_n':int(data.DX_GROUP.eq(2).sum()), 'sites':data.SITE_ID.nunique(),**row})
    data[['SUB_ID','SITE_ID','DX_GROUP']].to_csv(OUT/f'both_diagnosis_sites_{design}_{pipeline}.csv',index=False)
table = pd.DataFrame(restricted)
for _, block in table.groupby('design'):
    table.loc[block.index,'p_Holm_exploratory'] = adjusted_pvalues(block.p,'Holm')
for name, rows in [('site_diagnostics',summary),('site_by_group_H',sites),('influence',influence),('both_diagnosis_site_exclusions',excluded)]:
    pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
table.to_csv(OUT/'both_diagnosis_sites_models.csv',index=False)
print(table[['design','pipeline','n','beta','CI95_low','CI95_high','p_Holm_exploratory']].to_string(index=False))
print(pd.DataFrame(summary)[['design','pipeline','version','site_partial_R2_descriptive','max_leverage','max_Cooks_distance']].to_string(index=False))
