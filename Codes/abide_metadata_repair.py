"""Repair acquisition metadata without downloading the participant derivatives again.

Uses helpers embedded immediately before this module in the repair notebook.
"""
import urllib.parse
from functools import partial

if 'fetch_file' not in globals():
    from abide_download import *  # The Kaggle notebook embeds these helpers first.

TREE_URL = 'https://api.github.com/repos/FCP-INDI/indi_bidsification/git/trees/master?recursive=1'
BIDS_RAW = 'https://raw.githubusercontent.com/FCP-INDI/indi_bidsification'


def validate_tree(path):
    tree = json.loads(Path(path).read_text())
    if tree.get('truncated') or not isinstance(tree.get('tree'), list) or not tree.get('sha'):
        raise ValueError('Incomplete or invalid upstream repository index')
    return {'commit_sha': tree['sha'], 'n_entries': len(tree['tree'])}


def sidecar_paths(tree, subject_id, site_id=None, file_id=None):
    pattern = re.compile(r'(?<!\d)0*' + str(int(subject_id)) + r'(?!\d)')
    individual = [entry['path'] for entry in tree['tree']
            if entry.get('type') == 'blob'
            and entry['path'].startswith('ABIDE/scan_jsons/')
            and entry['path'].endswith('.json')
            and pattern.search(entry['path'])
            and not re.search(r't1w|anat|mprage|t2w', entry['path'], re.I)]
    if individual or site_id is None:
        return individual
    # The actual repository uses inherited site/task metadata, not per-subject files.
    normalize = lambda name: re.sub(r'[^A-Z0-9]', '', str(name).upper())
    site = normalize(site_id)
    aliases = {'CMU': {'CMUA', 'CMUB'}, 'MAXMUN': {'MAXMUNA', 'MAXMUNB', 'MAXMUNC', 'MAXMUND'}}
    allowed = {site} | aliases.get(site, set())
    # FILE_ID can preserve an acquisition variant lost in SITE_ID (CMU_a vs CMU).
    # Only narrow within the explicitly allowed site family; never select by TR magnitude.
    prefix = normalize(str(file_id).rsplit('_', 1)[0]) if file_id else None
    folders = {normalize(entry['path'].split('/')[2]) for entry in tree['tree']
               if entry['path'].startswith('ABIDE/scan_jsons/') and len(entry['path'].split('/')) == 4
               and entry['path'].endswith('/task-rest_bold.json')}
    if prefix in allowed and prefix in folders:
        allowed = {prefix}
    return [entry['path'] for entry in tree['tree']
            if entry.get('type') == 'blob'
            and len(entry['path'].split('/')) == 4
            and entry['path'].startswith('ABIDE/scan_jsons/')
            and entry['path'].endswith('/task-rest_bold.json')
            and normalize(entry['path'].split('/')[2]) in allowed]


YEO_MIRROR = ('https://raw.githubusercontent.com/edickie/scwg2018_python_neuroimaging/master/'
              'resources/rois/yeo_2011/Yeo_JNeurophysiol11_MNI152/')
YEO_MEMBERS = ('Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz',
               'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz',
               'Yeo2011_7Networks_ColorLUT.txt', 'Yeo_JNeurophysiol11_MNI152_README')


def validate_yeo_map(path):
    with gzip.open(path, 'rb') as stream:
        data = stream.read()
    # Older FreeSurfer distributions may use a 348-byte offset without the
    # extension flag, or leave bitpix unset. Interpret these only for this atlas,
    # preserving original bytes and requiring exact payload size plus labels 0–7.
    if len(data) < 348:
        raise ValueError('Truncated atlas header')
    endian = '<' if struct.unpack('<i', data[:4])[0] == 348 else '>'
    datatype = struct.unpack_from(endian + 'h', data, 70)[0]
    types = {2: 'u1', 4: 'i2', 8: 'i4', 16: 'f4', 64: 'f8', 256: 'i1', 512: 'u2', 768: 'u4'}
    if datatype not in types:
        raise ValueError(f'Unsupported Yeo label datatype: {datatype}')
    raw_bitpix = struct.unpack_from(endian + 'h', data, 72)[0]
    raw_offset = struct.unpack_from(endian + 'f', data, 108)[0]
    expected_bitpix = np.dtype(types[datatype]).itemsize * 8
    permitted_bitpix = (0, expected_bitpix, 4) if datatype == 16 else (0, expected_bitpix)
    if raw_bitpix not in permitted_bitpix or raw_offset not in (348, 352):
        raise ValueError(f'Unexpected atlas header: datatype={datatype}, bitpix={raw_bitpix}, vox_offset={raw_offset}')
    normalized = bytearray(data[:348])
    struct.pack_into(endian + 'h', normalized, 72, expected_bitpix)
    struct.pack_into(endian + 'f', normalized, 108, 352)
    header = nifti_header(normalized)
    header.update(vox_offset=int(raw_offset), raw_bitpix=raw_bitpix, raw_vox_offset=raw_offset)
    header['legacy_header_interpretation'] = raw_offset == 348 or raw_bitpix != expected_bitpix
    if len(header['shape']) < 3 or any(d != 1 for d in header['shape'][3:]):
        raise ValueError('Expected a spatial atlas, with optional singleton trailing dimensions')
    expected_size = int(raw_offset) + math.prod(header['shape']) * expected_bitpix // 8
    if len(data) != expected_size:
        raise ValueError(f'Atlas payload length mismatch: {len(data)} != {expected_size}')
    values = np.frombuffer(data, dtype=np.dtype(endian + types[datatype]),
                           count=math.prod(header['shape']), offset=header['vox_offset'])
    slope, intercept = struct.unpack_from(endian + '2f', data, 112)
    # NIfTI slope zero means no scaling; an unset intercept is then ignored.
    values = np.unique(values)
    if math.isfinite(slope) and slope != 0:
        values = values * slope + intercept
    if not np.array_equal(values, np.arange(8)):
        raise ValueError(f'Expected background 0 and original seven network labels 1–7; got {values.tolist()}')
    return {**header, 'labels': values.astype(int).tolist()}


def normalized_yeo_copy(path):
    """Write a standards-compatible copy, preserving voxel bytes and spatial header."""
    path = Path(path)
    meta = validate_yeo_map(path)
    raw = gzip.decompress(path.read_bytes())
    endian = '<' if struct.unpack('<i', raw[:4])[0] == 348 else '>'
    header = bytearray(raw[:348])
    payload = raw[meta['vox_offset']:]
    struct.pack_into(endian + 'h', header, 72, meta['bitpix'])
    struct.pack_into(endian + 'f', header, 108, 352)
    normalized = bytes(header) + b'\x00\x00\x00\x00' + payload
    target = path.with_name(path.name.replace('.nii.gz', '.normalized.nii.gz'))
    target.write_bytes(gzip.compress(normalized, mtime=0))
    validate_yeo_map(target)
    assert normalized[352:] == payload
    assert normalized[252:328] == raw[252:328]  # qform and sform fields unchanged.
    record = {'source_path': str(path), 'source_sha256': sha256(path),
              'path': str(target), 'sha256': sha256(target), 'created_utc': utc_now(),
              'original_bitpix': meta['raw_bitpix'], 'normalized_bitpix': meta['bitpix'],
              'original_vox_offset': meta['raw_vox_offset'], 'normalized_vox_offset': 352,
              'voxel_payload_sha256': hashlib.sha256(payload).hexdigest(),
              'voxel_bytes_unchanged': True, 'spatial_transform_fields_unchanged': True,
              'operation': 'Header normalization only; no resampling or relabeling'}
    write_json(target.with_name(target.name + '.provenance.json'), record)
    return record


def fetch_yeo_mirror(output_root, offline=False):
    records = []
    manifest_path = Path(output_root) / 'yeo_mirror_manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    for name in YEO_MEMBERS:
        destination = Path(output_root) / 'atlases/Yeo7_mirror' / name
        rejected = destination.with_name(destination.name + '.rejected')
        url = YEO_MIRROR + name
        old = next((r for r in previous if r.get('url') == url), {})
        if name.endswith('.nii.gz') and not destination.exists() and rejected.exists() and old.get('rejected_sha256') == sha256(rejected):
            # Revalidate preserved evidence under the corrected, atlas-specific reader.
            details = validate_yeo_map(rejected)
            shutil.copy2(rejected, destination)
            receipt = {'url': url, 'path': str(destination), 'sha256': sha256(destination),
                       'size_bytes': destination.stat().st_size, 'status': 'verified_existing',
                       'checked_utc': utc_now(), 'details': details,
                       'recovered_from_rejected_download': True, 'error': ''}
            write_json(destination.with_name(destination.name + '.receipt.json'), receipt)
        result = fetch_file(url, destination,
                            validate_yeo_map if name.endswith('.nii.gz') else None, retries=2, offline=offline)
        if name.endswith('.nii.gz') and result['status'] in SUCCESS:
            result['normalized_copy'] = normalized_yeo_copy(destination)
        records.append({'resource': name, **result})
    write_json(Path(output_root) / 'yeo_mirror_manifest.json', records)
    return {'resource': 'Yeo7_mirrored_files',
            'status': 'downloaded' if all(r['status'] in SUCCESS for r in records) else 'failed',
            'url': YEO_MIRROR, 'details': {'files': records, 'distribution': 'third-party HTTPS mirror',
            'original_archive_identical_bytes_verified': False},
            'error': '; '.join(r.get('error', '') for r in records if r['status'] not in SUCCESS)}


def sidecar_tr(path, source_path):
    obj = json.loads(Path(path).read_text())
    if not isinstance(obj, dict):
        raise ValueError('Expected BIDS sidecar object')
    identity = source_path + ' ' + str(obj.get('TaskName', ''))
    if not re.search(r'rest|bold|func', identity, re.I):
        raise ValueError('Cannot establish that this is a functional resting-state sidecar')
    value = obj.get('RepetitionTime')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Missing/invalid BIDS RepetitionTime (seconds); no unit guessing')
    return {'TR_seconds': float(value), 'source_path': source_path}


def resolve_sidecar_candidates(candidates, scope='exact_subject'):
    valid = [r for r in candidates if r['status'] in SUCCESS]
    if len(valid) != len(candidates) or not valid:
        return {'TR_seconds': None, 'resolution': 'missing_or_unverified_sidecars'}
    values = [r['details']['TR_seconds'] for r in valid]
    if not all(math.isclose(values[0], v, rel_tol=1e-6, abs_tol=1e-6) for v in values):
        return {'TR_seconds': None, 'resolution': 'conflicting_scan_TRs'}
    return {'TR_seconds': values[0], 'resolution': ('site_inherited_BIDS_sidecars'
            if scope == 'site_inherited' else 'subject_matched_BIDS_sidecars'),
            'TR_source_scope': scope,
            'TR_source_urls': [r['url'] for r in valid]}


def repair_metadata(dataset_root, output_root, offline=False):
    """Read a mounted Kaggle dataset; write a small patch bundle under /kaggle/working."""
    dataset_root, output_root = Path(dataset_root), Path(output_root)
    metadata = dataset_root / 'metadata' if (dataset_root / 'metadata').is_dir() else dataset_root
    fetch = partial(fetch_file, offline=offline)
    output_root.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(metadata / 'participants_requested.csv')
    headers = json.loads((metadata / 'scan_headers.json').read_text())
    resources = json.loads((metadata / 'resource_manifest.json').read_text())
    # Keep the input evidence and metadata in this standalone supplement.
    for name in ('participants_requested.csv', 'scan_headers.json', 'resource_manifest.json', 'completion_summary.json'):
        source = metadata / name
        if source.exists():
            target = output_root / 'original_metadata' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    repaired_resources = []
    for old in resources:
        if old['status'] in SUCCESS:
            repaired_resources.append(old)
            continue
        name = old['resource']
        if name.endswith('.html'):
            page = old['url'].rsplit('/', 1)[-1]
            url = 'https://preprocessed-connectomes-project.github.io/abide/' + page
            result = fetch(url, output_root / 'provenance' / name, retries=2)
        elif name == 'Yeo_JNeurophysiol11_MNI152.zip':
            # Do not repeat the refused FTP request. Retain the original failure below.
            repaired_resources.append(old)
            mirrored = fetch_yeo_mirror(output_root, offline=offline)
            repaired_resources.append(mirrored)
            print('Reference repair:', mirrored['resource'], mirrored['status'], mirrored.get('error', ''))
            continue
        else:
            result = dict(old)
        repaired_resources.append({'resource': name, **result})
        print('Reference repair:', name, result['status'], result.get('error', ''))
    write_json(output_root / 'resource_repair_manifest.json', repaired_resources)

    index = fetch(TREE_URL, output_root / 'provenance/indi_bidsification_tree.json', validate_tree, retries=2)
    write_json(output_root / 'repository_index_download.json', index)
    tree = json.loads(Path(index['path']).read_text()) if index['status'] in SUCCESS else None
    rows, resolved_headers = [], {}
    for i, subject in enumerate(participants.to_dict('records'), 1):
        sid = int(subject['SUB_ID'])
        old = dict(headers.get(str(sid), {}))
        row = {'SUB_ID': sid, 'FILE_ID': subject['FILE_ID'], 'header_time_unit_code': old.get('time_unit_code'),
               'header_TR_seconds': old.get('TR_seconds'), 'header_n_volumes': old.get('shape', [None]*4)[3]}
        header_path = metadata / 'scan_headers' / f'{sid}_nifti_header.bin'
        raw_interval = old.get('raw_pixdim4')
        if header_path.exists():
            raw_interval = nifti_header(header_path.read_bytes())['raw_pixdim4']
        row['header_raw_pixdim4_unknown_units'] = raw_interval
        if old.get('TR_seconds'):
            resolution = {'TR_seconds': old['TR_seconds'], 'resolution': 'header_with_known_units',
                          'TR_source_urls': [old['url']]}
        elif tree:
            candidates = []
            sources = sidecar_paths(tree, sid, subject['SITE_ID'], subject['FILE_ID'])
            row['site_variant_narrowed_using_FILE_ID'] = sources != sidecar_paths(tree, sid, subject['SITE_ID'])
            scope = 'exact_subject' if sidecar_paths(tree, sid) else 'site_inherited'
            for source in sources:
                url = f"{BIDS_RAW}/{tree['sha']}/{urllib.parse.quote(source, safe='/')}"
                result = fetch(url, output_root / 'sidecars' / source,
                                    lambda path, source=source: sidecar_tr(path, source), retries=2)
                candidates.append(result)
            write_json(output_root / 'TR_candidate_downloads' / f'{sid}.json', candidates)
            resolution = resolve_sidecar_candidates(candidates, scope=scope)
        else:
            resolution = {'TR_seconds': None, 'resolution': 'repository_index_unavailable'}
        row.update(resolution)
        row['nyquist_Hz'] = .5 / row['TR_seconds'] if row['TR_seconds'] else None
        row['TR_at_least_2_5_seconds'] = row['TR_seconds'] >= 2.5 if row['TR_seconds'] else None
        # A comparison flag only; unknown header units are never assigned by magnitude.
        row['header_numeric_interval_agrees_if_seconds'] = (
            math.isclose(raw_interval, row['TR_seconds'], rel_tol=1e-5)
            if raw_interval and row['TR_seconds'] else None)
        resolved_headers[str(sid)] = {**old, **resolution}
        rows.append(row)
        if i % 25 == 0 or i == len(participants):
            write_json(output_root / 'scan_headers_resolved.json', resolved_headers)
            pd.DataFrame(rows).to_csv(output_root / 'TR_recovery.csv', index=False)
            print(f"TR recovery {i}/{len(participants)}; resolved {sum(bool(r['TR_seconds']) for r in rows)}")
    required = {'CC200.nii.gz', 'Yeo_JNeurophysiol11_MNI152.zip',
                'pipeline_config_abide_rerun.yml', 'abide_rois.txt'}
    reference_complete = all(any(r['resource'] == name and r['status'] in SUCCESS for r in repaired_resources) for name in required)
    mirrored_ready = any(r['resource'] == 'Yeo7_mirrored_files' and r['status'] in SUCCESS for r in repaired_resources)
    usable_references = all(any(r['resource'] == name and r['status'] in SUCCESS for r in repaired_resources)
                            for name in required - {'Yeo_JNeurophysiol11_MNI152.zip'}) and (
                            reference_complete or mirrored_ready)
    summary = {'finished_utc': utc_now(), 'participant_derivatives_redownloaded': 0,
               'subjects_requested': len(participants), 'subjects_with_TR': sum(bool(r['TR_seconds']) for r in rows),
               'original_reference_archive_complete': reference_complete,
               'required_reference_resources_complete': usable_references,
               'reference_files_available_with_documented_mirror': usable_references,
               'TR_source_scope_counts': pd.Series([r.get('TR_source_scope', r['resolution']) for r in rows]).value_counts().to_dict(),
               'metadata_repair_complete': usable_references and all(bool(r['TR_seconds']) for r in rows),
               'scientific_eligibility_confirmed': False,
               'offline_run': offline,
               'notes': 'This supplements the original dataset; it does not replace its derivative manifests.'}
    write_json(output_root / 'repair_summary.json', summary)
    print(json.dumps(summary, indent=2))
    return pd.DataFrame(rows)
