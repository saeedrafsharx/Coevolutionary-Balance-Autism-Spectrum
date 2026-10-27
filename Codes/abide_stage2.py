"""Frozen-design revision features and inference. Helpers from abide_stage1 precede this in the notebook."""
if 'spatial_image' not in globals():
    from abide_stage1 import *
from scipy import stats
from scipy.optimize import brentq
import importlib.metadata


def adjusted_pvalues(values, method='BH'):
    p = np.asarray(values, float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return result
    order = valid[np.argsort(p[valid])]
    m = len(order)
    if method == 'Holm':
        adjusted = np.maximum.accumulate(p[order] * np.arange(m, 0, -1))
    elif method == 'BH':
        adjusted = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    else:
        raise ValueError(method)
    result[order] = np.minimum(adjusted, 1)
    return result


def model_design(frame, include_group=True):
    columns = {}
    if include_group:
        columns['diagnosis_ASD'] = frame.DX_GROUP.eq(1).astype(float)
    for name in ('AGE_AT_SCAN', 'FIQ', 'func_mean_fd'):
        values = pd.to_numeric(frame[name], errors='coerce')
        if not np.isfinite(values).all() or values.eq(-9999).any():
            raise ValueError(f'Missing/sentinel model covariate: {name}')
        if values.std(ddof=0) > 0:
            columns[name] = (values - values.mean()) / values.std(ddof=0)
    design = pd.DataFrame(columns, index=frame.index)
    sites = pd.get_dummies(frame.SITE_ID.astype(str), prefix='site', drop_first=True, dtype=float)
    design = pd.concat([pd.Series(1., index=frame.index, name='intercept'), design, sites], axis=1)
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError('Rank-deficient covariate/site design; inspect model manifest')
    if len(design) <= design.shape[1] + 3:
        raise ValueError('Insufficient residual degrees of freedom')
    return design


def hc3_fit(y, design, term='diagnosis_ASD'):
    if np.linalg.matrix_rank(design.to_numpy(dtype=float)) != design.shape[1]:
        raise ValueError('Rank-deficient regression design')
    try:
        import statsmodels.api as sm
    except ImportError:
        # Same OLS/HC3 estimator as the statsmodels matrix check run on Kaggle.
        # This implementation is explicitly identified in the result, not a
        # replacement analysis or a different covariance estimator.
        matrix = design.to_numpy(dtype=float)
        values = np.asarray(y, float)
        inv = np.linalg.inv(matrix.T@matrix)
        beta = np.linalg.lstsq(matrix, values, rcond=None)[0]
        residual = values-matrix@beta
        leverage = np.sum((matrix@inv)*matrix, axis=1)
        if np.max(leverage) >= 1-1e-9:
            raise ValueError('Unit leverage makes HC3 uncertainty undefined')
        weighted = matrix*(residual/(1-leverage))[:, None]
        cov = inv@weighted.T@weighted@inv
        index = design.columns.get_loc(term)
        se = float(np.sqrt(cov[index, index]))
        df = len(values)-matrix.shape[1]
        critical = stats.t.ppf(.975, df)
        return {'beta': float(beta[index]), 'SE_HC3': se,
                'CI95_low': float(beta[index]-critical*se), 'CI95_high': float(beta[index]+critical*se),
                'p': float(2*stats.t.sf(abs(beta[index]/se), df)), 'df_residual': float(df),
                'implementation_backend': 'numpy_OLS_HC3_matrix'}, None
    fit = sm.OLS(np.asarray(y, float), design.astype(float)).fit(cov_type='HC3', use_t=True)
    if np.max(fit.get_influence().hat_matrix_diag) >= 1 - 1e-9:
        raise ValueError('Unit leverage makes HC3 uncertainty undefined')
    ci = fit.conf_int(alpha=.05).loc[term]
    return {'beta': float(fit.params[term]), 'SE_HC3': float(fit.bse[term]),
            'CI95_low': float(ci.iloc[0]), 'CI95_high': float(ci.iloc[1]),
            'p': float(fit.pvalues[term]), 'df_residual': float(fit.df_resid),
            'implementation_backend': 'statsmodels_OLS_HC3'}, fit


def hedge_g(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    if pooled <= 0:
        return np.nan
    return float((1-3/(4*(len(a)+len(b))-9)) * (a.mean()-b.mean()) / pooled)


def bootstrap_g(a, b, n_boot=2000, seed=101):
    rng = np.random.default_rng(seed)
    values = [hedge_g(rng.choice(a, len(a), replace=True), rng.choice(b, len(b), replace=True)) for _ in range(n_boot)]
    return np.nanpercentile(values, [2.5, 97.5])


def minimum_detectable_d(n1, n0, alpha=.05, power=.8):
    if min(n1, n0) < 2 or not 0 < alpha < power < 1:
        raise ValueError('Invalid sample sizes, alpha, or target power')
    df = n1+n0-2
    critical = stats.t.ppf(1-alpha/2, df)
    def detection(d):
        nc = d*np.sqrt(n1*n0/(n1+n0))
        # Some SciPy/Boost versions return NaN from nct.sf at large noncentrality.
        # CDF complements plus an adaptive small bracket avoid that unstable tail.
        value = 1-stats.nct.cdf(critical, df, nc)+stats.nct.cdf(-critical, df, nc)
        if not np.isfinite(value):
            raise ValueError(f'Nonfinite noncentral-t power at effect size {d}')
        return float(np.clip(value, 0, 1))
    upper = .1
    while detection(upper) < power:
        upper *= 2
        if upper > 10:
            raise ValueError('Could not bracket sensitivity-power solution')
    return float(brentq(lambda d: detection(d)-power, 0, upper))


def primary_effect_diagnostics(frame, fit_result, fit, seed=101):
    a = frame.loc[frame.DX_GROUP.eq(1), 'Whole_Energy_global'].to_numpy()
    b = frame.loc[frame.DX_GROUP.eq(2), 'Whole_Energy_global'].to_numpy()
    lo, hi = bootstrap_g(a, b, seed=seed)
    # Exploratory conventional small-effect bounds, not a validated clinical SESOI.
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    margin = .2*pooled
    beta, se, df = fit_result['beta'], fit_result['SE_HC3'], fit_result['df_residual']
    tost = max(stats.t.sf((beta+margin)/se, df), stats.t.cdf((beta-margin)/se, df))
    return {'ASD_n': len(a), 'Control_n': len(b), 'ASD_mean': float(a.mean()), 'Control_mean': float(b.mean()),
            'ASD_SD': float(a.std(ddof=1)), 'Control_SD': float(b.std(ddof=1)),
            'unadjusted_Hedges_g': hedge_g(a, b), 'g_bootstrap_CI95_low': float(lo), 'g_bootstrap_CI95_high': float(hi),
            'TOST_margin_raw_units': float(margin), 'TOST_exploratory_p': float(tost),
            'TOST_bounds_basis': 'plus/minus 0.2 observed pooled SD; exploratory, not clinical equivalence',
            'two_sample_MDE80_alpha05_approx': minimum_detectable_d(len(a), len(b)),
            'two_sample_MDE80_alpha025_approx': minimum_detectable_d(len(a), len(b), alpha=.025),
            'power_model_caveat': 'two-sample t approximation; not adjusted for sites/covariates'}


def feature_family(name):
    if name == 'Whole_Energy_global':
        return 'primary_H'
    if name.startswith('Whole_'):
        return 'secondary_Whole'
    return name.split('_', 1)[0]


def inferential_features(frame):
    return [c for c in frame if c.startswith(('Whole_', 'Intra_', 'Inter_')) and 'legacy' not in c]


def group_models(all_features, metadata, output, minimum_site_n=3, run_combat=True):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    results, diagnostics, exclusions, combat_logs = [], [], [], []
    for (name, pipeline), table in all_features.groupby(['design', 'pipeline'], sort=False):
        table = table.merge(metadata, on='SUB_ID', validate='one_to_one')
        counts = table.SITE_ID.value_counts()
        retained = table.SITE_ID.map(counts).ge(minimum_site_n)
        for row in table.loc[~retained, ['SUB_ID', 'SITE_ID', 'DX_GROUP']].to_dict('records'):
            exclusions.append({'design': name, 'pipeline': pipeline, 'reason': 'site_N_below_3_for_adjusted_model', **row})
        data = table.loc[retained].reset_index(drop=True)
        data[['SUB_ID', 'SITE_ID', 'DX_GROUP']].to_csv(out / f'model_participants_{name}_{pipeline}.csv', index=False)
        x = model_design(data)
        x.to_csv(out / f'covariate_design_{name}_{pipeline}.csv', index=False)
        columns = inferential_features(data)
        valid = [c for c in columns if np.isfinite(data[c]).all() and data[c].std() > 1e-12]
        for feature in sorted(set(columns)-set(valid)):
            results.append({'design': name, 'pipeline': pipeline, 'model': 'raw_adjusted', 'Feature': feature,
                            'family': feature_family(feature), 'status': 'unavailable_or_constant_not_tested'})
        versions = {'raw_adjusted': data}
        if run_combat:
            try:
                from neuroCombat import neuroCombat
                covars = pd.DataFrame({'SITE': data.SITE_ID.astype(str), 'Group': data.DX_GROUP.eq(1).astype(int),
                                      'Age': data.AGE_AT_SCAN, 'IQ': data.FIQ, 'FD': data.func_mean_fd})
                adjusted = neuroCombat(dat=data[valid].to_numpy().T, covars=covars, batch_col='SITE',
                                       categorical_cols=['Group'], continuous_cols=['Age', 'IQ', 'FD'])['data'].T
                if adjusted.shape != data[valid].shape or not np.isfinite(adjusted).all():
                    raise ValueError('Invalid ComBat result; no raw fallback allowed')
                modified = data.copy()
                modified[valid] = adjusted
                versions['ComBat_sensitivity_adjusted'] = modified
                modified.to_csv(out / f'ComBat_sensitivity_features_{name}_{pipeline}.csv', index=False)
                combat_logs.append({'design': name, 'pipeline': pipeline, 'status': 'ok',
                                    'max_abs_change': float(np.max(np.abs(adjusted-data[valid].to_numpy())))})
            except Exception as exc:
                combat_logs.append({'design': name, 'pipeline': pipeline, 'status': 'failed_no_fallback', 'error': str(exc)})
        for version, data_version in versions.items():
            for feature in valid:
                stats_row, fit = hc3_fit(data_version[feature], x)
                results.append({'design': name, 'pipeline': pipeline, 'model': version, 'Feature': feature,
                                'family': feature_family(feature), 'status': 'ok', 'n': len(data_version), **stats_row})
                if feature == 'Whole_Energy_global':
                    try:
                        diagnostics.append({'design': name, 'pipeline': pipeline, 'model': version,
                                            **primary_effect_diagnostics(data_version, stats_row, fit)})
                    except Exception as exc:
                        diagnostics.append({'design': name, 'pipeline': pipeline, 'model': version,
                                            'diagnostic_error': str(exc)})
            pd.DataFrame(results).to_csv(out / 'group_models_in_progress.csv', index=False)
        # Pooled-energy trimming is sensitivity only, recalculated within this complete design.
        flagged, fences = flag_iqr(table, 'Whole_Energy_global')
        fences['stage'] = 'all eligible design participants before small-site exclusion'
        trim = table.loc[~flagged].copy()
        # Sites can become singletons after trimming; apply the same minimum-size rule again.
        trim = trim.loc[trim.SITE_ID.map(trim.SITE_ID.value_counts()).ge(minimum_site_n)].reset_index(drop=True)
        trim[['SUB_ID', 'SITE_ID', 'DX_GROUP']].to_csv(out / f'trim_sensitivity_participants_{name}_{pipeline}.csv', index=False)
        if len(trim) >= 20 and trim.DX_GROUP.nunique() == 2:
            try:
                stats_row, fit = hc3_fit(trim.Whole_Energy_global, model_design(trim))
                results.append({'design': name, 'pipeline': pipeline, 'model': 'energy_trim_sensitivity',
                                'Feature': 'Whole_Energy_global', 'family': 'primary_H', 'status': 'ok', 'n': len(trim), **stats_row})
            except ValueError as exc:
                exclusions.append({'design': name, 'pipeline': pipeline, 'reason': 'trim_model_unavailable', 'error': str(exc)})
        json_save(out / f'trim_fences_{name}_{pipeline}.json', fences)
    table = pd.DataFrame(results)
    table['p_adjusted'] = np.nan
    table['correction'] = ''
    for key, subset in table.loc[table.status.eq('ok')].groupby(['design', 'model', 'family']):
        method = 'Holm' if key[2] == 'primary_H' else 'BH'
        table.loc[subset.index, 'p_adjusted'] = adjusted_pvalues(subset.p, method)
        table.loc[subset.index, 'correction'] = method + '_across_both_pipelines_within_family'
    table.to_csv(out / 'group_models.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(out / 'primary_effect_sizes_power_equivalence.csv', index=False)
    pd.DataFrame(exclusions).to_csv(out / 'model_exclusions.csv', index=False)
    pd.DataFrame(combat_logs).to_csv(out / 'ComBat_diagnostics.csv', index=False)
    return table


def residual_partial_correlation(y, z, controls):
    controls = np.asarray(controls, float)
    rank = np.linalg.matrix_rank(controls)
    if len(y) <= rank+2:
        raise ValueError('Insufficient partial-correlation degrees of freedom')
    a = np.asarray(y)-controls@np.linalg.lstsq(controls, y, rcond=None)[0]
    b = np.asarray(z)-controls@np.linalg.lstsq(controls, z, rcond=None)[0]
    r = float(np.corrcoef(a, b)[0, 1])
    df = len(y)-rank-1
    p = float(2*stats.t.sf(abs(r)*np.sqrt(df/max(1-r*r, 1e-15)), df))
    return r, p, df


def efficiency_relationships(primary, metadata, output):
    rows, interactions = [], []
    for pipeline, table in primary.groupby('pipeline'):
        data = table.merge(metadata, on='SUB_ID', validate='one_to_one')
        data = data.loc[data.SITE_ID.map(data.SITE_ID.value_counts()).ge(3)].reset_index(drop=True)
        for diagnosis, subset in data.groupby('DX_GROUP'):
            subset = subset.reset_index(drop=True)
            # Drop sites that become singleton within this group before nuisance residualization.
            subset = subset.loc[subset.SITE_ID.map(subset.SITE_ID.value_counts()).ge(2)].reset_index(drop=True)
            try:
                controls = model_design(subset, include_group=False)
                controls['mean_FC'] = subset.QC_MeanFisherZ_FC
                for feature in ('Whole_GlobalEfficiency', 'Whole_LocalEfficiency'):
                    r, p, df = residual_partial_correlation(subset.Whole_Energy_global, subset[feature], controls)
                    rows.append({'pipeline': pipeline, 'DX_GROUP': diagnosis, 'feature': feature,
                                 'n': len(subset), 'partial_r': r, 'p': p, 'df': df,
                                 'controls': 'mean Fisher-z FC, FD, age, IQ, site'})
            except ValueError as exc:
                rows.append({'pipeline': pipeline, 'DX_GROUP': diagnosis, 'error': str(exc)})
        x = model_design(data)
        eff = (data.Whole_GlobalEfficiency-data.Whole_GlobalEfficiency.mean())/data.Whole_GlobalEfficiency.std(ddof=0)
        x['global_efficiency'] = eff
        x['mean_FC'] = data.QC_MeanFisherZ_FC
        x['group_x_efficiency'] = x.diagnosis_ASD*eff
        row, _ = hc3_fit(data.Whole_Energy_global, x, term='group_x_efficiency')
        interactions.append({'pipeline': pipeline, 'n': len(data), **row})
    table = pd.DataFrame(rows)
    if 'p' in table:
        table['p_BH'] = adjusted_pvalues(table.p)
    table.to_csv(Path(output) / 'efficiency_partial_correlations.csv', index=False)
    table = pd.DataFrame(interactions)
    table['p_Holm'] = adjusted_pvalues(table.p, 'Holm')
    table.to_csv(Path(output) / 'group_by_efficiency_interactions.csv', index=False)


def null_energy_statistics(fc, states, blocks=None, permutations=1000, seed=101):
    rng = np.random.default_rng(seed)
    observed = energy(fc, states)
    null = np.empty(permutations)
    for start in range(0, permutations, 100):
        count = min(100, permutations-start)
        vectors = np.tile(states, (count, 1))
        if blocks is None:
            for row in vectors:
                rng.shuffle(row)
        else:
            for block in np.unique(blocks):
                indices = np.flatnonzero(blocks == block)
                for row in vectors:
                    row[indices] = rng.permutation(row[indices])
        null[start:start+count] = -.5*np.sum((vectors@fc)*vectors, axis=1)
    mean, sd = float(null.mean()), float(null.std(ddof=1))
    return {'observed_H': observed, 'null_mean': mean, 'null_SD': sd,
            'z': (observed-mean)/sd if sd > 1e-12 else None,
            'p_lower_energy': (1+int(np.sum(null <= observed)))/(permutations+1) if sd > 1e-12 else None,
            'n_permutations': permutations, 'zero_null_variance': sd <= 1e-12}


def rebuild_stage2(root, patch, out, plan, compute_local=True, n_null=1000):
    import nibabel as nib
    root, patch, out = Path(root), Path(patch), Path(out)
    for folder in ('features', 'audit', 'arrays', 'statistics'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    json_save(out / 'frozen_design.json', plan)
    json_save(out / 'dependency_versions.json', {name: importlib.metadata.version(name)
        for name in ('numpy', 'pandas', 'scipy', 'nibabel', 'statsmodels', 'scikit-learn', 'neuroCombat')})
    metadata = pd.read_csv(root / 'metadata/participants_requested.csv')
    metadata = metadata.loc[metadata.SUB_ID.isin(plan['demographic_eligible_SUB_IDs'])].copy()
    if len(metadata) != len(plan['demographic_eligible_SUB_IDs']) or metadata.SUB_ID.duplicated().any():
        raise ValueError('Frozen demographic manifest does not match inputs')
    if not (metadata.AGE_AT_SCAN.between(18, 30)&metadata.FIQ.gt(80)&metadata.FIQ.ne(-9999)).all():
        raise ValueError('Demographic inclusion criterion mismatch')
    metadata.to_csv(out / 'audit/demographic_manifest.csv', index=False)
    cc, cc_affine = spatial_image(resolve_nifti(root / 'atlases/CC200.nii.gz'))
    labels = integer_labels(cc)
    yeo_path = resolve_nifti(patch / 'atlases/Yeo7_mirror/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.normalized.nii.gz')
    yeo, yeo_affine = spatial_image(yeo_path)
    mapping = anatomical_mapping(cc, resample_labels(yeo, yeo_affine, cc.shape, cc_affine), cc_affine)
    mapping.loc[mapping.Yeo_covered_fraction < plan['cortical_assignment_minimum_fraction'], ['network_id', 'network_name']] = [0, 'Unassigned']
    mapping.to_csv(out / 'audit/anatomical_mapping_majority_cortex.csv', index=False)
    indices = {int(k): i for i, k in enumerate(labels)}
    mapping = mapping.set_index('ROI_label')
    output_rows, qc_rows, null_rows, failures, source_rows = [], [], [], [], []
    for pipeline, (filtered, unfiltered) in PIPELINES.items():
        for counter, person in enumerate(metadata.to_dict('records'), 1):
            sid, fid = int(person['SUB_ID']), person['FILE_ID']
            try:
                ts_path = derivative(root, fid, filtered, 'rois_cc200')
                map_path = resolve_nifti(derivative(root, fid, unfiltered, 'falff'))
                mask_path = resolve_nifti(derivative(root, fid, unfiltered, 'func_mask'))
                ts, _ = read_timeseries(ts_path, labels, allow_constant=True)
                raw_map, affine = spatial_image(map_path)
                raw_mask, mask_affine = spatial_image(mask_path)
                for kind, path in [('ROI_time_series', ts_path), ('fALFF', map_path), ('mask', mask_path)]:
                    source_rows.append({'pipeline': pipeline, 'SUB_ID': sid, 'kind': kind,
                                        'path': str(path), 'sha256': digest(path)})
                falff, coverage = roi_falff(raw_map, resample_labels(cc, cc_affine, raw_map.shape, affine),
                    resample_labels(raw_mask > .5, mask_affine, raw_map.shape, affine), labels, allow_missing=True)
                coverage_fraction = np.array([r['coverage_fraction'] for r in coverage])
                for design in plan['designs']:
                    if sid not in design['SUB_IDs']:
                        continue
                    selected = np.array([indices[k] for k in design['ROI_labels']])
                    if not np.isfinite(falff[selected]).all() or np.any(coverage_fraction[selected] < plan['coverage_threshold']):
                        raise ValueError(f"Frozen coverage rule not met: {design['name']}")
                    selected_ts = ts[:, selected]
                    if np.any(np.ptp(selected_ts, axis=0) == 0):
                        raise ValueError(f"Frozen node set has constant time series: {design['name']}")
                    states = median_states(falff[selected])
                    fc = fc_matrix(selected_ts)
                    net = mapping.loc[design['ROI_labels'], 'network_id'].to_numpy(dtype=int)
                    measures = features(fc, states, net, compute_local=compute_local)
                    output_rows.append({'design': design['name'], 'pipeline': pipeline, 'SUB_ID': sid, **measures})
                    qc_rows.append({'design': design['name'], 'pipeline': pipeline, 'SUB_ID': sid,
                                    'n_ROIs': len(selected), 'minimum_coverage': float(coverage_fraction[selected].min()),
                                    'n_positive_states': int(np.sum(states > 0)), 'n_volumes': len(ts)})
                    np.savez_compressed(out / f"arrays/{design['name']}_{pipeline}_{sid}.npz",
                                        ROI_labels=labels[selected], states=states, fALFF=falff[selected], FC_FisherZ=fc)
                    if design['name'].startswith('primary') and n_null:
                        for null_name, blocks in [('node_permutation', None), ('network_hemisphere_blocks',
                                net*3+np.where(mapping.loc[design['ROI_labels'], 'centroid_x_mm'].to_numpy() < 0, 0, 1))]:
                            null_rows.append({'pipeline': pipeline, 'SUB_ID': sid, 'null': null_name,
                                'n_ROIs': len(states), 'n_positive_states': int(np.sum(states > 0)),
                                'delta_H_per_uniform_edge_shift': float(-((states.sum())**2-len(states))/2),
                                **null_energy_statistics(fc, states, blocks, n_null, seed=10000+sid+(pipeline=='GSR'))})
            except Exception as exc:
                failures.append({'pipeline': pipeline, 'SUB_ID': sid, 'error': str(exc), 'traceback': traceback.format_exc()})
                print('FAILED', pipeline, sid, exc, flush=True)
            if counter % 5 == 0 or counter == len(metadata):
                pd.DataFrame(output_rows).to_csv(out / 'features/all_designs_raw_untrimmed.csv', index=False)
                pd.DataFrame(failures, columns=['pipeline', 'SUB_ID', 'error', 'traceback']).to_csv(out / 'audit/failures.csv', index=False)
                print(f'{pipeline} {counter}/{len(metadata)}; feature rows {len(output_rows)}', flush=True)
    pd.DataFrame(qc_rows).to_csv(out / 'audit/rebuilt_measurement_QC.csv', index=False)
    pd.DataFrame(source_rows).to_csv(out / 'audit/input_hashes.csv', index=False)
    pd.DataFrame(null_rows).to_csv(out / 'statistics/participant_null_scores.csv', index=False)
    expected = 2*sum(d['n_subjects'] for d in plan['designs'])
    if failures or len(output_rows) != expected:
        json_save(out / 'stage2_status.json', {'complete': False, 'expected_feature_rows': expected,
            'feature_rows': len(output_rows), 'failures': len(failures), 'statistics_started': False})
        raise RuntimeError('Feature extraction differs from frozen design; inspect saved failures before inference')
    return pd.DataFrame(output_rows), metadata
