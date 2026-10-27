"""Repair Stage 2 statistics from exported features; never rerun imaging/graphs."""
import argparse
import importlib.util
import shutil
from pathlib import Path

from abide_stage2_run import *


def repair_stage2_results(source, destination, complete=False):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Keep original review outputs unchanged; choose another output directory')
    missing = [name for name in ('statsmodels', 'neuroCombat', 'sklearn') if importlib.util.find_spec(name) is None]
    if complete and missing:
        raise RuntimeError('Missing packages: '+', '.join(missing)+'. Run bash Codes/finish_stage2.sh in a normal terminal.')
    destination.mkdir(parents=True, exist_ok=True)
    stats_out = destination / 'statistics'
    stats_out.mkdir(exist_ok=True)
    (destination / 'features').mkdir(exist_ok=True)
    (destination / 'audit').mkdir(exist_ok=True)
    features_path = source / 'features/all_designs_raw_untrimmed.csv'
    features_table = pd.read_csv(features_path)
    metadata = pd.read_csv(source / 'audit/demographic_manifest.csv')
    plan = json.loads((source / 'frozen_design.json').read_text())
    expected = 2*sum(d['n_subjects'] for d in plan['designs'])
    if len(features_table) != expected or features_table.duplicated(['design', 'pipeline', 'SUB_ID']).any():
        raise ValueError('Exported feature rows do not match the frozen design')
    shutil.copy2(features_path, destination / 'features' / features_path.name)
    shutil.copy2(source / 'audit/demographic_manifest.csv', destination / 'audit/demographic_manifest.csv')
    shutil.copy2(source / 'frozen_design.json', destination / 'frozen_design.json')
    for name in ('stage2_summary.json', 'analysis_job_status.json', 'dependency_versions.json', 'backend_validation.json'):
        if (source / name).exists():
            shutil.copy2(source / name, destination / 'audit' / ('original_'+name))
    for path in (source / 'statistics').glob('*'):
        if path.name.startswith(('clinical_', 'efficiency_', 'group_by_efficiency_', 'participant_null_')):
            shutil.copy2(path, stats_out / path.name)
    json_save(destination / 'backend_validation.json', validate_stats_backend())
    print('Repairing group models from', len(features_table), 'saved feature rows', flush=True)
    models = group_models(features_table, metadata, stats_out, run_combat=complete)
    primary_name = plan['designs'][0]['name']
    primary = features_table[features_table.design.eq(primary_name)].copy()
    classification_complete = False
    if complete:
        print('Rerunning only classification with complete site-disjoint folds', flush=True)
        classifier_analyses(primary, metadata, stats_out)
        scope = json.loads((stats_out / 'classification_scope.json').read_text())
        classification_complete = scope['complete_pipeline_model_pairs'] == 4
    combat_failures = None
    if complete:
        table = pd.read_csv(stats_out / 'ComBat_diagnostics.csv')
        combat_failures = int(table.status.ne('ok').sum())
    summary = {'finished_utc': datetime.now(timezone.utc).isoformat(), 'input_review': str(source),
        'feature_table_sha256': digest(features_path), 'feature_rows_reused': len(features_table),
        'imaging_or_graph_recalculation': False, 'raw_adjusted_models_recovered': True,
        'ComBat_sensitivity_requested': complete, 'ComBat_sensitivity_failures': combat_failures,
        'classification_full_cohort_recovered': classification_complete,
        'completed_original_clinical_efficiency_null_outputs_preserved': True,
        'full_repair_complete': complete and combat_failures == 0 and classification_complete,
        'ready_for_submission': False,
        'repair_changes': ['adaptive CDF-based power bracket; optional power diagnostics cannot abort group tables',
                          'size-balanced site-disjoint classification folds, verified to contain both classes',
                          'incomplete OOF predictions cannot produce a full-cohort aggregate']}
    json_save(destination / 'repair_summary.json', summary)
    print(models.loc[models.Feature.eq('Whole_Energy_global'),
        ['design', 'pipeline', 'model', 'n', 'beta', 'CI95_low', 'CI95_high', 'p', 'p_adjusted']].to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=repo / 'results/abide_stage2_review')
    parser.add_argument('--output', type=Path, default=repo / 'results/abide_stage2_repaired')
    parser.add_argument('--complete', action='store_true', help='Require dependencies and rerun ComBat sensitivities plus classification')
    args = parser.parse_args()
    repair_stage2_results(args.source, args.output, complete=args.complete)


if __name__ == '__main__':
    main()
