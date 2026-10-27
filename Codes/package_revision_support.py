"""Package local supporting tables with provenance; no remote publication."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import pandas as pd
from classification_output_source import classification_statistics_directory
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'revision/revised_paper'
SRC=ROOT/'results/abide_stage2_repaired'
files={}
for path in (SRC/'statistics').glob('*'):
    if path.suffix in {'.csv','.json'} and 'in_progress' not in path.name and not path.name.startswith('ComBat_sensitivity_features_'):
        files['statistics/'+path.name]=path.read_bytes()
classifier_stats=classification_statistics_directory()
if classifier_stats != SRC/'statistics':
    for path in classifier_stats.glob('classification*'):
        if path.suffix in {'.csv','.json'}:
            baseline=SRC/'statistics'/path.name
            if baseline.exists():files['classification_baseline/'+path.name]=baseline.read_bytes()
            files['statistics/'+path.name]=path.read_bytes()
    for path in (classifier_stats.parent/'audit').glob('*.json'):
        files['classification_extension_audit/'+path.name]=path.read_bytes()
    files['classification_extension_audit/completion_summary.json']=(classifier_stats.parent/'completion_summary.json').read_bytes()
# Harmonized features include only required analysis fields, not every original phenotype.
for path in (SRC/'statistics').glob('ComBat_sensitivity_features_*.csv'):
    data=pd.read_csv(path)
    cols=[c for c in data if c.startswith(('Whole_','Intra_','Inter_')) or c in ['SUB_ID','SITE_ID','DX_GROUP','AGE_AT_SCAN','FIQ','func_mean_fd','design','pipeline']]
    files['harmonized_features/'+path.name]=data[cols].to_csv(index=False).encode()
for path in (ROOT/'revision/stage2_verified/diagnostics').glob('*.csv'):
    files['diagnostics/'+path.name]=path.read_bytes()
for path in (ROOT/'revision/stage2_verified/control_nulls').glob('*.csv'):
    files['control_nulls/'+path.name]=path.read_bytes()
for path in (ROOT/'revision/stage2_verified/continuity').glob('*.csv'):
    files['continuity/'+path.name]=path.read_bytes()
selection=ROOT/'results/abide_ml_feature_selection_review'
if (selection/'completion_summary.json').exists():
    for folder in ['statistics','audit','inner_cv']:
        for path in (selection/folder).glob('*'):
            if path.suffix in {'.csv','.json'}:files['ml_feature_selection/'+folder+'/'+path.name]=path.read_bytes()
    files['ml_feature_selection/completion_summary.json']=(selection/'completion_summary.json').read_bytes()
    files['ml_feature_selection/specification.md']=(ROOT/'revision/analysis_plan/ML_FEATURE_SELECTION_EXTENSION.md').read_bytes()
for path in (ROOT/'revision/analysis_plan').glob('*.csv'):
    if 'participants' in path.name or 'ROIs' in path.name:
        files['frozen_design/'+path.name]=path.read_bytes()
for rel,path in {
    'features/all_designs_raw_untrimmed.csv': SRC/'features/all_designs_raw_untrimmed.csv',
    'provenance/repair_summary.json':SRC/'repair_summary.json',
    'provenance/frozen_design.json':SRC/'frozen_design.json',
    'provenance/input_hashes.csv':ROOT/'results/abide_stage2_review/audit/input_hashes.csv',
    'provenance/measurement_QC.csv':ROOT/'results/abide_stage2_review/audit/rebuilt_measurement_QC.csv',
    'provenance/anatomical_mapping.csv':ROOT/'results/abide_stage2_review/audit/anatomical_mapping_majority_cortex.csv',
    'provenance/TR_recovery.csv':ROOT/'Data/abide_metadata_patch_local_v2/TR_recovery.csv',
}.items(): files[rel]=path.read_bytes()
metadata=pd.read_csv(SRC/'audit/demographic_manifest.csv')
cols=['SUB_ID','FILE_ID','SITE_ID','DX_GROUP','AGE_AT_SCAN','SEX','HANDEDNESS_CATEGORY','FIQ','func_mean_fd']
files['features/primary_analysis_covariates.csv']=metadata[cols].to_csv(index=False).encode()
for name in ['original_dependency_versions.json','actual_data_HC3_crosscheck.csv']:
    files['provenance/'+name]=(SRC/'audit'/name).read_bytes()
files['README.md']=b'''# S2 Supporting Information: local draft archive

This archive accompanies the revised analysis, not the original submission.
No raw MRI data are included. Source derivatives remain with ABIDE/PCP and the
author's input datasets. Author review of source terms and sharing statements is
required before deposit; this script does not publish anything.

statistics/: complete final group, clinical, efficiency, null and prediction tables,
model manifests, trimming fences and model exclusions.
classification_baseline/: original completed logistic/SVM outputs, preserved when extended.
classification_extension_audit/: original-algorithm restoration plan and completion status.
harmonized_features/: saved harmonized features and analysis covariates only.
diagnostics/: reviewer follow-up site/influence and both-diagnosis-site analyses.
control_nulls/: explicit 71-male-control baseline and individual BH comparisons.
continuity/: full original-question map, significant and nominal-only ledgers,
original H-correlation reproduction, restored H complements and legacy descriptor.
ml_feature_selection/: separate exploratory selection-method comparison, all
inner candidates, per-fold selected names, bootstrap comparisons and audit plan.
frozen_design/: fixed ROI lists and participant sets.
features/: original 626 raw feature rows and minimal primary analysis covariates.
provenance/: repair status, source hashes, QC, atlas mapping, inherited TR records
and original Kaggle dependency versions (not the local repair environment).

No missing clinical score is imputed as an observed outcome. Unavailable limbic
features are not null tests. Harmonization, spatial-null and predictive caveats
are described in the manuscript and S1. Historical fALFF provenance remains open.
MANIFEST.csv gives checksums of all other archive members except itself.
'''
manifest=pd.DataFrame([{'file':name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()} for name,content in sorted(files.items())])
files['MANIFEST.csv']=manifest.to_csv(index=False).encode()
archive=OUT/'S2_supporting_tables_DRAFT.zip'
with ZipFile(archive,'w',ZIP_DEFLATED) as handle:
    for name,content in sorted(files.items()):handle.writestr(name,content)
with ZipFile(archive) as handle:
    assert handle.testzip() is None
    for row in manifest.itertuples():assert hashlib.sha256(handle.read(row.file)).hexdigest()==row.sha256
manifest.to_csv(OUT/'S2_file_index.csv',index=False)
print(f'Packaged and checked {len(files)} files: {archive}')
