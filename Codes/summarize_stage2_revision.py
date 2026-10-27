"""Export manuscript-facing tables from the completed Stage 2 repair."""
from pathlib import Path
import json
import hashlib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from classification_output_source import classification_statistics_directory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'results/abide_stage2_repaired'
OUT = ROOT / 'revision/stage2_verified'
OUT.mkdir(parents=True, exist_ok=True)
summary = json.loads((SRC / 'repair_summary.json').read_text())
assert summary['full_repair_complete']
assert hashlib.sha256((SRC / 'features/all_designs_raw_untrimmed.csv').read_bytes()).hexdigest() == summary['feature_table_sha256']
groups = pd.read_csv(SRC / 'statistics/group_models.csv')
combat = pd.read_csv(SRC / 'statistics/ComBat_diagnostics.csv')
assert len(combat) == 6 and combat.status.eq('ok').all()
classifier_stats = classification_statistics_directory()
classification = pd.read_csv(classifier_stats / 'classification_conditional_intervals.csv')
predictions = pd.read_csv(classifier_stats / 'classification_OOF_predictions.csv')
assert len(classification) in (4, 8, 10) and classification.n_OOF_predictions.eq(140).all()
for _, frame in predictions.groupby(['pipeline', 'model']):
    assert len(frame) == 140 and frame.SUB_ID.nunique() == 140
    assert frame.groupby('site').fold.nunique().eq(1).all()
energy = groups.loc[groups.family.eq('primary_H')].copy()
assert len(energy) == 18, energy.Feature.unique()
energy.to_csv(OUT / 'Hamiltonian_all_sensitivities.csv', index=False)
classification.to_csv(OUT / 'classification_complete_OOF.csv', index=False)
ids = pd.read_csv(SRC / 'statistics/model_participants_primary_common_106_GSR.csv')
metadata = pd.read_csv(SRC / 'audit/demographic_manifest.csv')
model = ids.merge(metadata, on=['SUB_ID', 'SITE_ID', 'DX_GROUP'], validate='one_to_one')
rows = []
for group, frame in model.groupby('DX_GROUP'):
    row = {'group': 'ASD' if group == 1 else 'Control', 'n': len(frame), 'sites': frame.SITE_ID.nunique()}
    for variable in ['AGE_AT_SCAN', 'FIQ', 'func_mean_fd']:
        row[variable + '_mean'] = frame[variable].mean()
        row[variable + '_SD'] = frame[variable].std()
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT / 'primary_model_demographics.csv', index=False)
pd.crosstab(model.SITE_ID, model.DX_GROUP).rename(columns={1:'ASD',2:'Control'}).to_csv(OUT / 'primary_model_site_counts.csv')
primary = energy.loc[energy.design.eq('primary_common_106') & energy.model.eq('raw_adjusted')]
fig, axes = plt.subplots(1, 2, figsize=(10, max(3.7, .55*len(classification))), constrained_layout=True)
for i, row in enumerate(primary.itertuples()):
    axes[0].errorbar(row.beta, i, xerr=[[row.beta-row.CI95_low], [row.CI95_high-row.beta]], fmt='o', capsize=4)
axes[0].set(yticks=range(len(primary)), yticklabels=primary.pipeline, xlabel='Adjusted ASD coefficient (HC3 95% CI)', title='Primary 106-ROI Hamiltonian; n=136')
axes[0].axvline(0, color='gray', linestyle='--')
for i, row in enumerate(classification.itertuples()):
    axes[1].errorbar(row.balanced_accuracy, i, xerr=[[row.balanced_accuracy-row.conditional_site_bootstrap_CI95_low], [row.conditional_site_bootstrap_CI95_high-row.balanced_accuracy]], fmt='o', capsize=4)
axes[1].set(yticks=range(len(classification)), yticklabels=[r.pipeline+': '+{'LogisticRegression':'Logistic','SVM_RBF':'RBF SVM','GaussianNB':'Gaussian NB','KNN':'KNN','XGBoost':'XGBoost'}[r.model] for r in classification.itertuples()], xlabel='Balanced accuracy (conditional 95% CI)', title='Site-held-out classification; n=140')
axes[1].axvline(.5, color='gray', linestyle='--')
for extension in ['pdf', 'png']:
    fig.savefig(OUT / f'primary_results.{extension}', dpi=200)
plt.close(fig)
(OUT / 'verification.json').write_text(json.dumps({'source':str(SRC.relative_to(ROOT)), 'classification_source':str(classifier_stats.relative_to(ROOT)), 'feature_sha256':summary['feature_table_sha256'], 'ComBat_successful_design_pipeline_pairs':6, 'complete_classifier_pairs':len(classification), 'OOF_subjects_per_pair':140, 'group_model_n':len(model)}, indent=2)+'\n')
print('Verified and exported:', OUT)
