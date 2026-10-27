"""Stage 1: validate ABIDE derivatives, rebuild features, and audit legacy trimming.

No harmonization, participant trimming, hypothesis tests, or classifier fitting.
NiBabel is needed only for image I/O; pure-array operations are testable offline.
"""
import hashlib
import json
import math
import platform
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import fft as scipy_fft
from scipy.ndimage import affine_transform
from scipy.sparse.csgraph import shortest_path

NETWORKS = {1: 'Visual', 2: 'SomatoMotor', 3: 'DorsalAttn', 4: 'Salience_VentAttn',
            5: 'Limbic', 6: 'Frontoparietal', 7: 'Default'}
PIPELINES = {'GSR': ('filt_global', 'nofilt_global'),
             'noGSR': ('filt_noglobal', 'nofilt_noglobal')}


def json_save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def resolve_nifti(path):
    """Locate compressed, plain, or Kaggle-extracted NIfTI files by exact name."""
    path = Path(path)
    if path.name.endswith('.nii.gz'):
        plain = path.with_suffix('')
        compressed = path
    elif path.name.endswith('.nii'):
        plain = path
        compressed = path.with_name(path.name + '.gz')
    else:
        raise ValueError(f'Expected a .nii or .nii.gz path: {path}')
    candidates = [path, compressed, plain, plain / plain.name, compressed / plain.name]
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('NIfTI not found; checked: ' + ', '.join(str(p) for p in dict.fromkeys(candidates)))


def spatial_image(path):
    import nibabel as nib
    img = nib.load(str(path))
    if int(img.header['sform_code']) == 0 and int(img.header['qform_code']) == 0:
        raise ValueError(f'No explicit spatial transform: {path}')
    values = np.asanyarray(img.dataobj)
    if values.ndim > 3 and all(s == 1 for s in values.shape[3:]):
        values = values.reshape(values.shape[:3])
    if values.ndim != 3 or not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) < 1e-12:
        raise ValueError(f'Invalid spatial image: {path}')
    return values, img.affine.copy()


def resample_labels(source, source_affine, target_shape, target_affine):
    """Nearest-neighbor sampling in world coordinates; this is not registration."""
    transform = np.linalg.solve(source_affine, target_affine)
    return affine_transform(np.asarray(source, dtype=np.int32), transform[:3, :3],
                            offset=transform[:3, 3], output_shape=tuple(target_shape),
                            order=0, mode='constant', cval=0, prefilter=False)


def integer_labels(values):
    if not np.isfinite(values).all() or not np.array_equal(values, np.rint(values)):
        raise ValueError('Atlas must have finite integer labels')
    labels = np.unique(values).astype(int)
    return labels[labels > 0]


def anatomical_mapping(cc, yeo_on_cc, affine):
    labels = integer_labels(cc)
    if not set(np.unique(yeo_on_cc)).issubset(set(range(8))):
        raise ValueError('Expected original Yeo labels 0–7')
    network_sizes = {k: int(np.sum(yeo_on_cc == k)) for k in NETWORKS}
    rows = []
    for label in labels:
        mask = cc == label
        size = int(mask.sum())
        overlap = np.array([np.sum(mask & (yeo_on_cc == k)) for k in NETWORKS], dtype=int)
        dice = np.array([2 * n / (size + network_sizes[k]) if size + network_sizes[k] else 0
                         for k, n in zip(NETWORKS, overlap)])
        winners = np.flatnonzero(np.isclose(dice, dice.max(), rtol=1e-10, atol=1e-12))
        assigned = int(winners[0] + 1) if dice.max() > 0 and len(winners) == 1 else 0
        centroid = np.argwhere(mask).mean(axis=0)
        xyz = affine[:3, :3] @ centroid + affine[:3, 3]
        row = {'ROI_label': int(label), 'ROI_voxels': size, 'network_id': assigned,
               'network_name': NETWORKS.get(assigned, 'Unassigned'),
               'max_Dice': float(dice.max()), 'Yeo_covered_fraction': float(overlap.sum() / size),
               'assignment_tied': bool(dice.max() > 0 and len(winners) > 1),
               'low_cortical_coverage_flag': bool(overlap.sum() / size < .5),
               'centroid_x_mm': float(xyz[0]), 'centroid_y_mm': float(xyz[1]), 'centroid_z_mm': float(xyz[2])}
        for k in NETWORKS:
            row[f'Dice_{k}'] = float(dice[k - 1])
            row[f'overlap_voxels_{k}'] = int(overlap[k - 1])
        rows.append(row)
    return pd.DataFrame(rows)


def read_timeseries(path, atlas_labels, allow_constant=False):
    with Path(path).open() as f:
        header = f.readline().strip()
    if not header.startswith('#'):
        raise ValueError('Missing ROI label header; ordinal column order will not be assumed')
    tokens = header.replace('#', ' ').split()
    if not tokens or any(not re.fullmatch(r'\d+', t) for t in tokens):
        raise ValueError(f'Unrecognized ROI label header: {header[:120]}')
    labels = np.array([int(t) for t in tokens])
    ts = np.loadtxt(path, comments='#', ndmin=2)
    if ts.shape[1] != len(labels) or len(set(labels)) != len(labels):
        raise ValueError('Duplicate ROI labels or header/data column mismatch')
    if set(labels) != set(atlas_labels):
        missing, extra = set(atlas_labels) - set(labels), set(labels) - set(atlas_labels)
        raise ValueError(f'Time-series/atlas labels differ: missing={sorted(missing)}, extra={sorted(extra)}')
    if ts.shape[0] < 50 or not np.isfinite(ts).all():
        raise ValueError('Time series has <50 volumes or nonfinite values')
    if not allow_constant and np.any(np.ptp(ts, axis=0) == 0):
        raise ValueError('Constant ROI time series; FC would be undefined')
    order = [int(np.flatnonzero(labels == k)[0]) for k in atlas_labels]
    return ts[:, order], header


def roi_falff(map_values, roi_on_map, mask_values, atlas_labels, allow_missing=False):
    if map_values.shape != roi_on_map.shape or map_values.shape != mask_values.shape:
        raise ValueError('Map, ROI, and mask grids differ')
    valid = np.isfinite(map_values) & np.isfinite(mask_values) & (mask_values > .5)
    rows, means = [], []
    for label in atlas_labels:
        voxels = roi_on_map == label
        total = int(voxels.sum())
        included = voxels & valid
        n = int(included.sum())
        mean = float(map_values[included].mean()) if n else np.nan
        means.append(mean)
        rows.append({'ROI_label': int(label), 'voxels_on_map_grid': total,
                     'finite_masked_voxels': n, 'coverage_fraction': n / total if total else 0,
                     'mean_fALFF': mean, 'coverage_below_80pct_flag': bool(total == 0 or n / total < .8)})
    if not allow_missing and not np.isfinite(means).all():
        missing = [int(label) for label, mean in zip(atlas_labels, means) if not np.isfinite(mean)]
        raise ValueError(f'ROIs without finite in-mask fALFF voxels: {missing}')
    return np.array(means), rows


def fc_matrix(ts):
    corr = np.corrcoef(ts.T)
    if not np.isfinite(corr).all():
        raise ValueError('Nonfinite correlations')
    z = np.arctanh(np.clip(corr, -.9999, .9999))
    np.fill_diagonal(z, 0)
    return z


def median_states(values):
    return np.where(values >= np.median(values), 1, -1)


def legacy_ratio(ts):
    frequencies = scipy_fft.fftfreq(len(ts), 2.0)
    low = (np.abs(frequencies) >= .01) & (np.abs(frequencies) <= .08)
    power = np.abs(scipy_fft.fft(ts, axis=0)) ** 2
    return power[low].sum(axis=0) / power.sum(axis=0)


def energy(fc, states):
    i, j = np.triu_indices(len(states), 1)
    return float(-np.sum(fc[i, j] * states[i] * states[j]))


def efficiency(fc, local=False):
    w = np.abs(fc)
    distances = np.full(w.shape, np.inf)
    np.divide(1, w, out=distances, where=w > 0)
    np.fill_diagonal(distances, 0)
    def global_eff(d):
        if len(d) < 2:
            return 0.0
        paths = shortest_path(d, directed=False, method='D')
        inv = np.zeros(paths.shape)
        np.divide(1, paths, out=inv, where=(paths > 0) & np.isfinite(paths))
        return float(inv.sum() / (len(d) * (len(d) - 1)))
    g = global_eff(distances)
    if not local:
        return g, np.nan
    values = []
    for k in range(len(fc)):
        neighbors = np.flatnonzero(w[k] > 0)
        values.append(global_eff(distances[np.ix_(neighbors, neighbors)]))
    return g, float(np.mean(values))


def features(fc, states, assignment, compute_local=True):
    n = len(states)
    i, j = np.triu_indices(n, 1)
    w, products = fc[i, j], states[i] * states[j]
    active = w != 0
    if not active.any():
        raise ValueError('All FC edges are zero')
    result = {'Whole_Energy_global': energy(fc, states)}
    # Retain the original spectral descriptor transparently; its interpretation
    # needs justification before inclusion in an inferential feature family.
    sign = np.sign(fc)
    laplacian = np.diag(np.abs(sign).sum(axis=1)) - sign
    _, vecs = np.linalg.eigh(laplacian)
    partition = np.sign(vecs[:, 1])
    result['Whole_Bipolarity_legacy_spectral'] = float(np.mean(np.sign(w[active]) * partition[i[active]] * partition[j[active]] > 0))
    result['Whole_GlobalEfficiency'], result['Whole_LocalEfficiency'] = efficiency(fc, compute_local)
    definitions = {'agreement': (products == 1) & (w > 0),
                   'disagreement': (products == -1) & (w < 0),
                   'imbalanced_same': (products == 1) & (w < 0),
                   'imbalanced_opp': (products == -1) & (w > 0)}
    for name, mask in definitions.items():
        result['Whole_Prop_' + name] = float(mask.sum() / active.sum())
    block_sum = 0.0
    for a in NETWORKS:
        for b in range(a, 8):
            mask = ((assignment[i] == a) & (assignment[j] == b)) | ((assignment[i] == b) & (assignment[j] == a))
            prefix = f'Intra_{NETWORKS[a]}' if a == b else f'Inter_{NETWORKS[a]}__{NETWORKS[b]}'
            result[prefix + '_AvgWeight'] = float(w[mask].mean()) if mask.any() else np.nan
            result[prefix + '_AvgFalffW'] = float((w[mask] * products[mask]).mean()) if mask.any() else np.nan
            value = float(-np.sum(w[mask] * products[mask]))
            result[prefix + '_EnergyBlock'] = value if mask.any() else np.nan
            block_sum += value
    result.update(QC_MeanFisherZ_FC=float(w.mean()), QC_MeanAbsoluteFisherZ_FC=float(np.abs(w).mean()),
                  QC_ZeroEdge_fraction=float(np.mean(~active)),
                  QC_Energy_involving_unassigned_ROIs=float(result['Whole_Energy_global'] - block_sum))
    return result


def flag_iqr(frame, value_column):
    values = pd.to_numeric(frame[value_column], errors='coerce')
    finite = np.isfinite(values)
    if not finite.any():
        return pd.Series(False, index=frame.index), {'n_finite': 0}
    q1, q3 = values[finite].quantile([.25, .75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return finite & ((values < lo) | (values > hi)), {
        'n_finite': int(finite.sum()), 'Q1': float(q1), 'Q3': float(q3),
        'lower_fence': float(lo), 'upper_fence': float(hi), 'pooled_diagnoses': True,
        'quantile_interpolation': 'linear', 'fence_endpoints_included': True}


def derivative(root, file_id, strategy, kind):
    extension = '1D' if kind == 'rois_cc200' else 'nii.gz'
    return Path(root) / 'cpac' / strategy / kind / f'{file_id}_{kind}.{extension}'


def run_stage1(derivatives_root, patch_root, output_root, legacy_retained_ids=None,
               compute_local_efficiency=True, save_fc=True):
    import nibabel as nib
    import scipy
    root, patch, out = map(Path, (derivatives_root, patch_root, output_root))
    out.mkdir(parents=True, exist_ok=True)
    for folder in ('reference', 'features', 'audit', 'participant_arrays'):
        (out / folder).mkdir(exist_ok=True)
    json_save(out / 'run_config.json', {
        'started_utc': datetime.now(timezone.utc).isoformat(), 'derivatives_root': str(root),
        'metadata_patch_root': str(patch), 'primary_Yeo_mask': 'liberal (provisional; mapping audit required)',
        'falff_source': 'downloaded nofilt strategy maps; upstream computation remains to verify',
        'FC_source': 'downloaded filt strategy ROI time series, original length',
        'binarization': 'participant median; values >= median are +1',
        'outliers': 'pooled raw energy IQR flags only; no exclusions',
        'compute_local_efficiency': compute_local_efficiency, 'save_FC': save_fc,
        'packages': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scipy': scipy.__version__, 'nibabel': nib.__version__}})
    participants = pd.read_csv(root / 'metadata/participants_requested.csv')
    if participants.SUB_ID.duplicated().any() or participants.FILE_ID.duplicated().any():
        raise ValueError('Duplicated source participant identifiers')
    tr = pd.read_csv(patch / 'TR_recovery.csv')
    if tr.SUB_ID.duplicated().any() or set(tr.SUB_ID) != set(participants.SUB_ID):
        raise ValueError('TR metadata IDs do not match requested participants')
    participants = participants.merge(tr[['SUB_ID', 'TR_seconds']], on='SUB_ID', how='left', validate='one_to_one')
    if not (pd.to_numeric(participants.TR_seconds, errors='coerce') > 0).all():
        raise ValueError('Every requested participant needs a verified positive TR')
    participants.to_csv(out / 'audit/participants_requested.csv', index=False)
    cc_path = resolve_nifti(root / 'atlases/CC200.nii.gz')
    cc, cc_affine = spatial_image(cc_path)
    labels = integer_labels(cc)
    reference_hashes = {'CC200': {'path': str(cc_path), 'sha256': digest(cc_path)}}
    mappings = {}
    for variant, suffix in [('thin', ''), ('liberal', '_LiberalMask')]:
        name = f'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm{suffix}.normalized.nii.gz'
        path = resolve_nifti(patch / 'atlases/Yeo7_mirror' / name)
        values, affine = spatial_image(path)
        sampled = resample_labels(values, affine, cc.shape, cc_affine)
        nib.save(nib.Nifti1Image(sampled.astype(np.int16), cc_affine), str(out / f'reference/Yeo7_{variant}_on_CC200.nii.gz'))
        mapping = anatomical_mapping(cc, sampled, cc_affine)
        mapping.to_csv(out / f'reference/CC200_Yeo7_mapping_{variant}.csv', index=False)
        mappings[variant] = mapping
        reference_hashes[variant] = {'path': str(path), 'sha256': digest(path)}
    json_save(out / 'reference/source_hashes.json', reference_hashes)
    comparison = mappings['liberal'][['ROI_label', 'network_id']].merge(
        mappings['thin'][['ROI_label', 'network_id']], on='ROI_label', suffixes=('_liberal', '_thin'))
    comparison['assignment_differs'] = comparison.network_id_liberal != comparison.network_id_thin
    comparison.to_csv(out / 'reference/mapping_mask_comparison.csv', index=False)
    assignment = mappings['liberal'].set_index('ROI_label').loc[labels, 'network_id'].to_numpy()
    print(f'Atlas labels: {len(labels)}; unassigned ROIs: {(assignment == 0).sum()}; '
          f'thin/liberal disagreements: {comparison.assignment_differs.sum()}', flush=True)
    rows, audit, coverage, sources, legacy_rows, state_comparisons = [], [], [], [], [], []
    for pipeline, (filtered, unfiltered) in PIPELINES.items():
        for number, participant in enumerate(participants.to_dict('records'), 1):
            sid, fid = int(participant['SUB_ID']), str(participant['FILE_ID'])
            record = {'pipeline': pipeline, 'SUB_ID': sid, 'FILE_ID': fid,
                      'SITE_ID': participant['SITE_ID'], 'DX_GROUP': int(participant['DX_GROUP']),
                      'TR_seconds': float(participant['TR_seconds']), 'status': 'failed', 'error': ''}
            try:
                ts_path = derivative(root, fid, filtered, 'rois_cc200')
                ts, roi_header = read_timeseries(ts_path, labels)
                fc = fc_matrix(ts)
                old_states = median_states(legacy_ratio(ts))
                old_h = energy(fc, old_states)
                legacy_rows.append({'pipeline': pipeline, 'SUB_ID': sid, 'DX_GROUP': int(participant['DX_GROUP']),
                                    'SITE_ID': participant['SITE_ID'], 'legacy_fixed_TR_2_H': old_h})
                record.update(n_volumes=len(ts), n_ROIs=len(labels), roi_header=roi_header,
                              duration_seconds=len(ts) * float(participant['TR_seconds']))
                mask_path = resolve_nifti(derivative(root, fid, unfiltered, 'func_mask'))
                mask, mask_affine = spatial_image(mask_path)
                falff_path = resolve_nifti(derivative(root, fid, unfiltered, 'falff'))
                image, image_affine = spatial_image(falff_path)
                # Masks and CC200 labels use nearest-neighbor interpolation; fALFF values are never resampled.
                roi_grid = resample_labels(cc, cc_affine, image.shape, image_affine)
                mask_grid = resample_labels(mask > .5, mask_affine, image.shape, image_affine)
                falff, cov = roi_falff(image, roi_grid, mask_grid, labels)
                for row in cov:
                    coverage.append({'pipeline': pipeline, 'SUB_ID': sid, **row})
                signs = median_states(falff)
                alt_path = resolve_nifti(derivative(root, fid, filtered, 'falff'))
                alt, alt_affine = spatial_image(alt_path)
                alt_grid = resample_labels(cc, cc_affine, alt.shape, alt_affine)
                alt_mask = resample_labels(mask > .5, mask_affine, alt.shape, alt_affine)
                alt_values, _ = roi_falff(alt, alt_grid, alt_mask, labels)
                state_comparisons.append({'pipeline': pipeline, 'SUB_ID': sid,
                    'filt_nofilt_fALFF_file_hash_identical': digest(falff_path) == digest(alt_path),
                    'filt_nofilt_ROI_max_abs_difference': float(np.max(np.abs(falff - alt_values))),
                    'filt_nofilt_nodal_state_disagreements': int(np.sum(signs != median_states(alt_values))),
                    'downloaded_map_vs_legacy_state_disagreements': int(np.sum(signs != old_states))})
                feat = features(fc, signs, assignment, compute_local_efficiency)
                rows.append({'pipeline': pipeline, 'SUB_ID': sid, 'FILE_ID': fid,
                             'SITE_ID': participant['SITE_ID'], 'DX_GROUP': int(participant['DX_GROUP']), **feat})
                saved = {'ROI_labels': labels, 'fALFF': falff, 'states': signs, 'network_ids': assignment,
                         'legacy_states_fixed_TR_2': old_states}
                if save_fc:
                    saved['FC_FisherZ'] = fc
                np.savez_compressed(out / f'participant_arrays/{pipeline}_{sid}.npz', **saved)
                for kind, path in [('filtered_ROI_time_series', ts_path), ('unfiltered_strategy_fALFF', falff_path),
                                   ('filtered_strategy_fALFF_comparison', alt_path), ('unfiltered_strategy_mask', mask_path)]:
                    sources.append({'pipeline': pipeline, 'SUB_ID': sid, 'kind': kind, 'path': str(path),
                                    'size_bytes': path.stat().st_size, 'sha256': digest(path)})
                record.update(status='ok', n_positive_states=int((signs > 0).sum()),
                              n_median_ties=int(np.sum(falff == np.median(falff))),
                              fALFF_min=float(falff.min()), fALFF_max=float(falff.max()),
                              fALFF_outside_0_1_flag=bool(np.any((falff < 0) | (falff > 1))),
                              n_ROIs_coverage_below_80pct=sum(r['coverage_below_80pct_flag'] for r in cov))
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
                record['traceback'] = traceback.format_exc()
                print(f'{pipeline} {sid}: {record["error"]}', flush=True)
            audit.append(record)
            if number % 5 == 0 or number == len(participants):
                pd.DataFrame(audit).to_csv(out / 'audit/participant_processing.csv', index=False)
                pd.DataFrame(rows).to_csv(out / 'features/features_all_available_untrimmed.csv', index=False)
                print(f'{pipeline} {number}/{len(participants)}; successful features so far: {len(rows)}', flush=True)
    processing = pd.DataFrame(audit)
    pd.DataFrame(coverage).to_csv(out / 'audit/fALFF_ROI_coverage.csv', index=False)
    pd.DataFrame(sources).to_csv(out / 'audit/input_file_hashes.csv', index=False)
    pd.DataFrame(state_comparisons).to_csv(out / 'audit/fALFF_strategy_comparison.csv', index=False)
    all_features = pd.DataFrame(rows)
    old = pd.DataFrame(legacy_rows)
    fences, flags = [], []
    for pipeline in PIPELINES:
        legacy = old.loc[old.pipeline == pipeline].copy() if len(old) else pd.DataFrame()
        candidate = all_features.loc[all_features.pipeline == pipeline].copy() if len(all_features) else pd.DataFrame()
        if len(candidate):
            flagged, limits = flag_iqr(candidate, 'Whole_Energy_global')
            candidate['pooled_energy_IQR_outlier_flag'] = flagged
            candidate.to_csv(out / f'features/features_{pipeline}_untrimmed.csv', index=False)
            fences.append({'pipeline': pipeline, 'definition': 'downloaded_map_candidate', **limits})
            flags.append(candidate[['pipeline', 'SUB_ID', 'pooled_energy_IQR_outlier_flag']])
        if len(legacy):
            flagged, limits = flag_iqr(legacy, 'legacy_fixed_TR_2_H')
            legacy['legacy_pooled_IQR_outlier_flag'] = flagged
            retained = set(legacy_retained_ids.get(pipeline, [])) if legacy_retained_ids else set()
            legacy['historical_post_IQR_export_retained'] = legacy.SUB_ID.isin(retained) if retained else pd.NA
            legacy['reconstructed_vs_historical_mask_agrees'] = (~flagged == legacy.historical_post_IQR_export_retained) if retained else pd.NA
            legacy.to_csv(out / f'audit/legacy_trimming_reconstruction_{pipeline}.csv', index=False)
            fences.append({'pipeline': pipeline, 'definition': 'legacy_fixed_TR_2', **limits})
    pd.DataFrame(fences).to_csv(out / 'audit/IQR_fences_no_exclusions.csv', index=False)
    if flags:
        pd.concat(flags).to_csv(out / 'audit/outlier_flags_no_exclusions.csv', index=False)
    good_ids = {p: set(processing.loc[(processing.pipeline == p) & (processing.status == 'ok'), 'SUB_ID']) for p in PIPELINES}
    common = set.intersection(*good_ids.values())
    participants.assign(technical_features_available_both_pipelines=participants.SUB_ID.isin(common)).to_csv(
        out / 'audit/common_participant_ledger.csv', index=False)
    summary = {'finished_utc': datetime.now(timezone.utc).isoformat(), 'requested_subjects': len(participants),
        'successful_feature_rows': len(rows), 'expected_feature_rows': 2 * len(participants),
        'technical_failures': int((processing.status != 'ok').sum()),
        'feature_rows_by_pipeline': {p: len(ids) for p, ids in good_ids.items()},
        'common_technical_participants': len(common), 'atlas_ROI_count': len(labels),
        'unassigned_ROIs_liberal': int((assignment == 0).sum()),
        'thin_liberal_mapping_disagreements': int(comparison.assignment_differs.sum()),
        'technical_extraction_complete': len(rows) == 2 * len(participants),
        'scientific_cohort_approved': False, 'upstream_fALFF_validity_established': False,
        'outliers_removed': 0, 'harmonization_performed': False, 'group_tests_performed': False}
    json_save(out / 'stage1_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary
