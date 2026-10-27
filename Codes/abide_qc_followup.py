"""Lightweight coverage audit; retain all nodes/subjects and run no statistical tests."""
if 'roi_falff' not in globals():
    from abide_stage1 import *


def run_qc_followup(derivatives_root, output_root):
    root, out = Path(derivatives_root), Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(root / 'metadata/participants_requested.csv')
    cc, cc_affine = spatial_image(resolve_nifti(root / 'atlases/CC200.nii.gz'))
    labels = integer_labels(cc)
    rows, logs = [], []
    # Cases selected from Stage 1 coverage diagnostics, not group-effect results.
    preview_cases = {(50236, 43), (51459, 43), (51459, 52), (51474, 120),
                     (51268, 43), (51020, 43), (51017, 10), (51017, 32)}
    for pipeline, (filtered, unfiltered) in PIPELINES.items():
        for i, person in enumerate(participants.to_dict('records'), 1):
            sid, fid = int(person['SUB_ID']), person['FILE_ID']
            log = {'pipeline': pipeline, 'SUB_ID': sid, 'FILE_ID': fid, 'status': 'failed', 'error': ''}
            try:
                ts, _ = read_timeseries(derivative(root, fid, filtered, 'rois_cc200'), labels, allow_constant=True)
                constant = np.ptp(ts, axis=0) == 0
                log['n_constant_ROIs'] = int(constant.sum())
                log['constant_ROI_labels'] = ' '.join(str(int(k)) for k in labels[constant])
                image, affine = spatial_image(resolve_nifti(derivative(root, fid, unfiltered, 'falff')))
                mask, mask_affine = spatial_image(resolve_nifti(derivative(root, fid, unfiltered, 'func_mask')))
                roi_grid = resample_labels(cc, cc_affine, image.shape, affine)
                mask_grid = resample_labels(mask > .5, mask_affine, image.shape, affine)
                means, coverage = roi_falff(image, roi_grid, mask_grid, labels, allow_missing=True)
                for index, row in enumerate(coverage):
                    rows.append({'pipeline': pipeline, 'SUB_ID': sid, 'SITE_ID': person['SITE_ID'],
                                 'DX_GROUP': int(person['DX_GROUP']), 'constant_time_series': bool(constant[index]), **row})
                log.update(status='ok', n_ROIs_zero_support=int(np.sum(~np.isfinite(means))),
                           minimum_coverage=min(row['coverage_fraction'] for row in coverage))
                if pipeline == 'GSR':
                    for subject_id, label in sorted(preview_cases):
                        if subject_id == sid:
                            coverage_preview(image, mask_grid, roi_grid, label, sid, out)
            except Exception as exc:
                log['error'] = f'{type(exc).__name__}: {exc}'
            logs.append(log)
            if i % 10 == 0 or i == len(participants):
                pd.DataFrame(logs).to_csv(out / 'processing_log.csv', index=False)
                pd.DataFrame(rows).to_csv(out / 'all_participant_ROI_QC.csv', index=False)
                print(f'{pipeline} QC {i}/{len(participants)}', flush=True)
    qc = pd.DataFrame(rows)
    if not len(qc):
        json_save(out / 'qc_summary.json', {'complete': False, 'reason': 'No ROI-level rows; inspect processing_log.csv'})
        return
    summary = qc.groupby(['pipeline', 'ROI_label']).agg(
        n_subjects=('SUB_ID', 'size'), n_constant_TS=('constant_time_series', 'sum'),
        minimum_coverage=('coverage_fraction', 'min'), median_coverage=('coverage_fraction', 'median'),
        n_zero_support=('finite_masked_voxels', lambda v: int((v == 0).sum())))
    for threshold in (.1, .2, .5, .8, .9):
        counts = qc.groupby(['pipeline', 'ROI_label']).coverage_fraction.apply(lambda v: int((v >= threshold).sum()))
        summary[f'n_subjects_coverage_ge_{threshold}'] = counts
    summary.to_csv(out / 'ROI_support_summary.csv')
    # Diagnostic retention counts only. None of these thresholds is applied.
    policies = []
    for threshold in (0, .1, .2, .5, .8, .9):
        per_person = qc.assign(
            all_requirements=(~qc.constant_time_series) & (qc.finite_masked_voxels > 0)
                             & (qc.coverage_fraction >= threshold)).groupby(
            ['pipeline', 'SUB_ID']).all_requirements.all().reset_index()
        per_person = per_person.merge(participants[['SUB_ID', 'DX_GROUP']], on='SUB_ID', validate='many_to_one')
        for pipeline, frame in per_person.groupby('pipeline'):
            good = frame[frame.all_requirements]
            policies.append({'pipeline': pipeline, 'minimum_coverage_every_ROI': threshold,
                             'n': len(good), 'ASD': int(good.DX_GROUP.eq(1).sum()),
                             'Control': int(good.DX_GROUP.eq(2).sum()), 'applied': False})
    pd.DataFrame(policies).to_csv(out / 'coverage_policy_diagnostics_not_applied.csv', index=False)
    json_save(out / 'qc_summary.json', {'requested_subjects': len(participants), 'ROI_labels': len(labels),
        'ROI_rows': len(rows), 'processing_failures': sum(r['status'] != 'ok' for r in logs),
        'participant_exclusions_applied': 0, 'ROI_exclusions_applied': 0,
        'group_tests_performed': False, 'cohort_finalized': False})


def coverage_preview(image, mask, rois, label, sid, output_root):
    import matplotlib.pyplot as plt
    selected = rois == label
    locations = np.argwhere(selected)
    if not len(locations):
        return
    center = np.rint(locations.mean(axis=0)).astype(int)
    supported = selected & (mask > .5) & np.isfinite(image)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, ax in enumerate(axes):
        base = np.take(image, center[axis], axis=axis).T
        whole_roi = np.take(selected, center[axis], axis=axis).T
        supported_roi = np.take(supported, center[axis], axis=axis).T
        ax.imshow(np.ma.masked_invalid(base), origin='lower', cmap='gray')
        if whole_roi.any() and not whole_roi.all():
            ax.contour(whole_roi, levels=[.5], colors='red', linewidths=1)
        ax.imshow(np.ma.masked_where(~supported_roi, supported_roi), origin='lower', cmap='Greens', alpha=.8, vmin=0, vmax=1)
        ax.set_title(f'Axis {axis}'); ax.axis('off')
    figure.suptitle(f'{sid}, ROI {label}: {supported.sum()}/{selected.sum()} supported voxels\n'
                   'Red: full ROI boundary; green: in-mask finite support')
    figure.tight_layout()
    folder = Path(output_root) / 'coverage_previews'
    folder.mkdir(exist_ok=True)
    figure.savefig(folder / f'{sid}_ROI{label}.png', dpi=150)
    plt.close(figure)
