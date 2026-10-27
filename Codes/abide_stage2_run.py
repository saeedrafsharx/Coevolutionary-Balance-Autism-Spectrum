"""Orchestrate Stage 2 and export one compact review bundle."""
if 'classifier_analyses' not in globals():
    from abide_stage2_predictive import *


def validate_stats_backend():
    """Compare statsmodels HC3 covariance against its matrix formula before long runs."""
    rng = np.random.default_rng(404)
    x = pd.DataFrame({'intercept': np.ones(80), 'diagnosis_ASD': np.tile([0., 1.], 40),
                      'nuisance': rng.normal(size=80)})
    y = x.to_numpy()@np.array([1., 2., .4])+rng.normal(size=80)
    result, fit = hc3_fit(y, x)
    matrix = x.to_numpy()
    inv = np.linalg.inv(matrix.T@matrix)
    coef = inv@matrix.T@y
    residual = y-matrix@coef
    leverage = np.sum((matrix@inv)*matrix, axis=1)
    weighted = matrix*(residual/(1-leverage))[:, None]
    cov = inv@weighted.T@weighted@inv
    np.testing.assert_allclose(result['beta'], coef[1], rtol=1e-10)
    np.testing.assert_allclose(result['SE_HC3'], np.sqrt(cov[1, 1]), rtol=1e-10)
    return {'statsmodels_HC3_matrix_check': 'passed'}


def run_stage2(derivatives_root, patch_root, output_root, plan, n_perm=1000,
               run_combat=True, run_clinical=True, run_classification=True):
    import zipfile
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    json_save(out / 'run_settings.json', {'n_permutations': n_perm, 'run_ComBat_sensitivity': run_combat,
        'run_clinical': run_clinical, 'run_classification': run_classification, 'effect_size_bootstraps': 2000,
        'model_primary': 'raw site/age/IQ/FD-adjusted, HC3', 'prediction_harmonization': 'none'})
    json_save(out / 'backend_validation.json', validate_stats_backend())
    if run_classification:
        metadata_preview = pd.read_csv(Path(derivatives_root) / 'metadata/participants_requested.csv')
        metadata_preview = metadata_preview[metadata_preview.SUB_ID.isin(plan['designs'][0]['SUB_IDs'])]
        labels_preview = metadata_preview.DX_GROUP.eq(1).astype(int).to_numpy()
        sites_preview = metadata_preview.SITE_ID.to_numpy()
        fold_preview = []
        for fold, (train, test) in enumerate(site_group_splits(sites_preview, labels_preview, 5)):
            inner = site_group_splits(sites_preview[train], labels_preview[train], 3)
            fold_preview.append({'fold': fold, 'n_train': len(train), 'n_test': len(test),
                                 'ASD_test': int(labels_preview[test].sum()), 'inner_valid_folds': len(inner)})
        json_save(out / 'classification_fold_preflight.json', fold_preview)
    rows, metadata = rebuild_stage2(derivatives_root, patch_root, out, plan, n_null=n_perm)
    primary_name = plan['designs'][0]['name']
    primary = rows[rows.design.eq(primary_name)].copy()
    jobs = [('group_models', lambda: group_models(rows, metadata, out / 'statistics', run_combat=run_combat)),
            ('efficiency_relationships', lambda: efficiency_relationships(primary, metadata, out / 'statistics'))]
    if run_clinical:
        jobs.append(('clinical_analyses', lambda: clinical_analyses(primary, metadata, out / 'statistics', n_perm=n_perm)))
    if run_classification:
        jobs.append(('classification', lambda: classifier_analyses(primary, metadata, out / 'statistics')))
    status = []
    for name, job in jobs:
        print('START', name, flush=True)
        try:
            job()
            status.append({'analysis': name, 'status': 'completed'})
        except Exception as exc:
            status.append({'analysis': name, 'status': 'failed_no_fallback', 'error': str(exc), 'traceback': traceback.format_exc()})
            print('FAILED', name, exc, flush=True)
        json_save(out / 'analysis_job_status.json', status)
    # Descriptive plots only; all unavailable outcomes stay explicit in CSVs.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    data = primary.merge(metadata[['SUB_ID', 'DX_GROUP']], on='SUB_ID', validate='many_to_one')
    for ax, (pipeline, block) in zip(axes, data.groupby('pipeline')):
        samples = [block.loc[block.DX_GROUP.eq(g), 'Whole_Energy_global'] for g in (1, 2)]
        ax.boxplot(samples)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['ASD', 'Control'])
        ax.set_title(f'{pipeline}: common {plan["designs"][0]["n_ROIs"]}-ROI H (raw)')
        ax.set_ylabel('Hamiltonian')
    fig.tight_layout()
    fig.savefig(out / 'primary_raw_energy_distributions.png', dpi=150)
    plt.close(fig)
    summary = {'finished_utc': datetime.now(timezone.utc).isoformat(),
               'primary_demographic_n': len(metadata), 'rebuilt_feature_rows': len(rows),
               'analysis_jobs': status, 'all_jobs_completed': all(s['status'] == 'completed' for s in status),
               'statistical_results_reviewed': False, 'ready_for_submission': False,
               'remaining_scientific_checks': ['upstream fALFF band/input provenance',
                    'author review of changed anatomical scope and handedness description',
                    'QC/model diagnostics, sensitivity results, and model identifiability',
                    'rewritten manuscript, figures, supporting information, response letter'],
               'null_model_caveat': 'network/hemisphere blocks preserve coarse organization, not full spatial autocorrelation',
               'equivalence_caveat': 'exploratory conventional bounds; no validated clinical SESOI',
               'ComBat_caveat': 'exploratory sensitivity; uncertainty does not integrate harmonization parameter estimation'}
    summary['prediction_scope'] = 'site-held-out models conditional on cohort-wide QC mask; not prospective validation of mask selection'
    combat_file = out / 'statistics/ComBat_diagnostics.csv'
    if run_combat and combat_file.exists():
        combat = pd.read_csv(combat_file)
        summary['ComBat_sensitivity_failures'] = int(combat.status.ne('ok').sum())
    json_save(out / 'stage2_summary.json', summary)
    (out / 'README.md').write_text(
        '# Stage 2 revision outputs\n\nUse frozen_design.json for cohort/node definitions.\n'
        'Primary inference: raw site/age/IQ/FD-adjusted H, HC3 CI, Holm across two pipelines.\n'
        'Secondary families: BH across both pipelines within each design/model/family.\n'
        'ROI-policy, ComBat, and energy-trimming models are sensitivities, not additional primary claims.\n'
        'Clinical/PLS and classification analyses use the primary common-node design.\n'
        'Classification and PLSR hold out sites; no diagnosis-protected ComBat is used for prediction.\n'
        'Missing clinical outcomes are score-wise exclusions, not imaging exclusions.\n'
        'Uncomputed/structurally undefined features are not null results.\n'
        'Inspect analysis_job_status.json and ComBat_diagnostics.csv even if the notebook finishes.\n')
    archive = out.parent / 'abide_stage2_review.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in out.rglob('*'):
            if path.is_file() and path.suffix in ('.csv', '.json', '.png', '.md'):
                bundle.write(path, path.relative_to(out))
    print('Return:', archive, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary
