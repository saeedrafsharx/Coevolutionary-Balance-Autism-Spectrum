"""Expose saved healthy-control null comparisons without rerunning simulations."""
from pathlib import Path
import numpy as np
import pandas as pd
from abide_stage2 import adjusted_pvalues
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/abide_stage2_repaired'
OUT=ROOT/'revision/stage2_verified/control_nulls'
OUT.mkdir(parents=True,exist_ok=True)
null=pd.read_csv(SRC/'statistics/participant_null_scores.csv')
metadata=pd.read_csv(SRC/'audit/demographic_manifest.csv')
controls=metadata.loc[metadata.DX_GROUP.eq(2)&metadata.SEX.eq(1),['SUB_ID','SITE_ID','DX_GROUP','SEX']]
assert len(controls)==71 and controls.SUB_ID.is_unique
rows=null.merge(controls,on='SUB_ID',validate='many_to_one')
assert len(rows)==284 and not rows.duplicated(['pipeline','null','SUB_ID']).any()
assert rows.n_permutations.eq(1000).all() and not rows.zero_null_variance.any()
rows['delta_H']=rows.observed_H-rows.null_mean
assert np.allclose(rows.delta_H/rows.null_SD,rows.z)
rows['q_BH_all284_control_tests']=adjusted_pvalues(rows.p_lower_energy)
rows.to_csv(OUT/'control_participant_null_comparisons.csv',index=False)
summaries=[]
for (pipeline,scheme),frame in rows.groupby(['pipeline','null']):
    assert len(frame)==71 and frame.SUB_ID.nunique()==71
    rng=np.random.default_rng(20261026)
    sites=sorted(frame.SITE_ID.unique())
    blocks=[frame.loc[frame.SITE_ID.eq(site),['delta_H','z']].to_numpy() for site in sites]
    boot=np.array([np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))]).mean(0) for _ in range(5000)])
    intervals=np.percentile(boot,[2.5,97.5],axis=0)
    summaries.append({'pipeline':pipeline,'null':scheme,'n_controls':len(frame),'n_sites':len(sites),
                      'mean_observed_H':frame.observed_H.mean(),'mean_null_H':frame.null_mean.mean(),
                      'mean_delta_H':frame.delta_H.mean(),'delta_H_CI95_low':intervals[0,0],'delta_H_CI95_high':intervals[1,0],
                      'mean_z':frame.z.mean(),'z_CI95_low':intervals[0,1],'z_CI95_high':intervals[1,1],
                      'median_z':frame.z.median(),'n_observed_below_own_null_mean':int(frame.delta_H.lt(0).sum()),
                      'n_individual_p_below_05':int(frame.p_lower_energy.lt(.05).sum()),
                      'n_individual_BH_q_below_05':int(frame.q_BH_all284_control_tests.lt(.05).sum()),
                      'interval_scope':'5000 site-cluster bootstrap replicates; conditional on saved null summaries',
                      'null_scope':'fixed signed weighted FC; permute nodal states only'})
summary=pd.DataFrame(summaries)
summary.to_csv(OUT/'control_only_null_summary.csv',index=False)
print(summary[['pipeline','null','n_controls','mean_delta_H','mean_z','n_individual_p_below_05','n_individual_BH_q_below_05']].to_string(index=False))
