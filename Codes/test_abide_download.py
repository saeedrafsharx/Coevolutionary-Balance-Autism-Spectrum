"""Offline regression checks: python3 -m unittest discover -s Codes -v."""
import contextlib
import gzip
import io
import json
import struct
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import abide_download as download
import abide_metadata_repair as repair


def nifti_bytes(shape=(2, 2, 2), tr=2.5, unit=8):
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, len(shape), *shape, *([1] * (7-len(shape))))
    struct.pack_into("<h", header, 70, 16)  # float32
    struct.pack_into("<h", header, 72, 32)
    struct.pack_into("<8f", header, 76, 1, 3, 3, 3, tr, 1, 1, 1)
    struct.pack_into("<f", header, 108, 352)
    header[123] = unit
    header[344:348] = b"n+1\0"
    return gzip.compress(bytes(header) + np.ones(shape, dtype="<f4").tobytes())


PHENO = (
    "SUB_ID,FILE_ID,SITE_ID,DX_GROUP,AGE_AT_SCAN,SEX,FIQ,func_mean_fd,ADOS_TOTAL\n"
    "1,Site_0000001,Site,1,24,1,105,0.1,-9999\n"
    "2,Other_0000002,Other,2,17,2,-9999,0.6,10\n"
).encode()


class Response(io.BytesIO):
    def __init__(self, content, content_length=None):
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content) if content_length is None else content_length), "ETag": "fixture"}


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_selection_retains_flagged_and_unknown_subjects(self):
        selected = download.select_subjects(pd.read_csv(io.BytesIO(PHENO)), [1, 2, 999])
        self.assertEqual(selected.SUB_ID.tolist(), [1, 2, 999])
        self.assertTrue(selected.iloc[1].flag_age_outside_18_30_or_missing)
        self.assertTrue(selected.iloc[1].flag_FIQ_not_above_80_or_missing)
        self.assertFalse(selected.iloc[2].downloadable_file_id)
        with self.assertRaises(ValueError):
            download.select_subjects(pd.read_csv(io.BytesIO(PHENO)), [1, 1])
        plan = download.build_plan(selected, self.root)
        self.assertEqual(len(plan), 30)
        self.assertEqual(sum(row['status'] == 'missing_file_id' for row in plan), 10)
        download.export_manifest(self.root, plan)  # Includes unknown metadata without invalid JSON NaN.

    def test_plan_keeps_strategies_separate(self):
        subjects = download.select_subjects(pd.read_csv(io.BytesIO(PHENO)), [1])
        plan = download.build_plan(subjects, self.root)
        self.assertEqual(len(plan), 10)
        self.assertEqual(len({r['url'] for r in plan}), 10)
        self.assertEqual({r['strategy'] for r in plan}, set(download.STRATEGIES))
        for row in plan:
            self.assertIn('/' + row['strategy'] + '/', row['url'])

    def test_atomic_download_and_verified_resume(self):
        path = self.root / 'data.txt'
        with patch.object(download.urllib.request, 'urlopen', return_value=Response(b'valid')) as opener:
            first = download.fetch_file('https://example.org/data', path)
            second = download.fetch_file('https://example.org/data', path)
        self.assertEqual(first['status'], 'downloaded')
        self.assertEqual(second['status'], 'verified_existing')
        self.assertEqual(opener.call_count, 1)
        path.write_bytes(b'corrupt')
        with patch.object(download.urllib.request, 'urlopen', return_value=Response(b'repaired')):
            repaired = download.fetch_file('https://example.org/data', path)
        self.assertEqual(repaired['status'], 'downloaded')
        self.assertEqual(path.read_bytes(), b'repaired')

    def test_interrupted_transfer_is_not_saved(self):
        path = self.root / 'data.txt'
        with patch.object(download.urllib.request, 'urlopen', return_value=Response(b'short', 100)):
            result = download.fetch_file('https://example.org/data', path, retries=1)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(path.exists())
        self.assertFalse(path.with_name('data.txt.part').exists())

    def test_unavailable_strategy_does_not_fallback(self):
        error = urllib.error.HTTPError('https://example.org/filt_global', 404, 'missing', {}, None)
        with patch.object(download.urllib.request, 'urlopen', side_effect=error) as opener:
            result = download.fetch_file(error.url, self.root / 'missing', retries=3)
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(opener.call_count, 1)

    def test_nifti_validates_crc_and_payload(self):
        path = self.root / 'falff.nii.gz'
        path.write_bytes(nifti_bytes())
        self.assertEqual(download.validate_nifti(path)['shape'], [2, 2, 2])
        path.write_bytes(nifti_bytes()[:-5])
        with self.assertRaises((EOFError, OSError)):
            download.validate_nifti(path)
        path.write_bytes(gzip.compress(gzip.decompress(nifti_bytes())[:352]))
        with self.assertRaisesRegex(ValueError, 'Truncated NIfTI voxel'):
            download.validate_nifti(path)

    def test_TR_units_and_3d_not_TR(self):
        header = gzip.decompress(nifti_bytes((2, 2, 2, 3), tr=2500, unit=16))[:348]
        self.assertAlmostEqual(download.nifti_header(header)['TR_seconds'], 2.5)
        three_d = gzip.decompress(nifti_bytes())[:348]
        self.assertIsNone(download.nifti_header(three_d)['TR_seconds'])
        unknown = gzip.decompress(nifti_bytes((2, 2, 2, 3), tr=2, unit=0))[:348]
        self.assertIsNone(download.nifti_header(unknown)['TR_seconds'])
        self.assertEqual(download.nifti_header(unknown)['raw_pixdim4'], 2)

    def test_repair_matches_subject_and_rejects_ambiguous_TR(self):
        tree = {'tree': [{'type': 'blob', 'path': p} for p in (
            'ABIDE/scan_jsons/USM/sub-0050499_task-rest_bold.json',
            'ABIDE/scan_jsons/USM/sub-00504990_task-rest_bold.json',
            'ABIDE/scan_jsons/USM/sub-0050499_T1w.json')]}
        self.assertEqual(repair.sidecar_paths(tree, 50499), [tree['tree'][0]['path']])
        candidates = [{'status': 'downloaded', 'url': 'fixture', 'details': {'TR_seconds': v}} for v in (2, 3)]
        self.assertIsNone(repair.resolve_sidecar_candidates(candidates)['TR_seconds'])
        candidates[1]['details']['TR_seconds'] = 2
        self.assertEqual(repair.resolve_sidecar_candidates(candidates)['TR_seconds'], 2)

    def test_repair_requires_resting_state_identity(self):
        path = self.root / 'sidecar.json'
        path.write_text('{"RepetitionTime": 2.5}')
        with self.assertRaises(ValueError):
            repair.sidecar_tr(path, 'sub-0050499.json')
        self.assertEqual(repair.sidecar_tr(path, 'sub-0050499_task-rest_bold.json')['TR_seconds'], 2.5)

    def test_actual_site_level_metadata_layout(self):
        paths = ['ABIDE/scan_jsons/USM/task-rest_bold.json', 'ABIDE/scan_jsons/USM/T1w.json',
                 'ABIDE/scan_jsons/UM_1/task-rest_bold.json', 'ABIDE/scan_jsons/UM_2/task-rest_bold.json',
                 'ABIDE/scan_jsons/CMU_a/task-rest_bold.json', 'ABIDE/scan_jsons/CMU_b/task-rest_bold.json']
        tree = {'tree': [{'path': p, 'type': 'blob'} for p in paths]}
        self.assertEqual(repair.sidecar_paths(tree, 50499, 'USM'), [paths[0]])
        self.assertEqual(repair.sidecar_paths(tree, 50499, 'UM_1'), [paths[2]])
        self.assertEqual(repair.sidecar_paths(tree, 50499, 'CMU'), paths[4:])
        self.assertEqual(repair.sidecar_paths(tree, 50649, 'CMU', 'CMU_a_0050649'), [paths[4]])
        candidates = [{'status': 'downloaded', 'url': p, 'details': {'TR_seconds': 2}} for p in paths[4:]]
        result = repair.resolve_sidecar_candidates(candidates, scope='site_inherited')
        self.assertEqual(result['resolution'], 'site_inherited_BIDS_sidecars')
        candidates[1]['details']['TR_seconds'] = 3
        self.assertIsNone(repair.resolve_sidecar_candidates(candidates, scope='site_inherited')['TR_seconds'])

    def test_repair_notebook_contains_current_helpers(self):
        folder = Path(__file__).parent
        nb = json.loads((folder / 'abide-metadata-repair.ipynb').read_text())
        for cell in nb['cells']:
            if cell.get('id') == 'repair-helpers':
                self.assertEqual(''.join(cell['source']), (folder / 'abide_metadata_repair.py').read_text())
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')

    def test_yeo_validation_accepts_singleton_4d_and_checks_labels(self):
        path = self.root / 'yeo.nii.gz'
        raw = gzip.decompress(nifti_bytes((2, 2, 2, 1)))
        path.write_bytes(gzip.compress(raw[:352] + np.arange(8, dtype='<f4').tobytes()))
        self.assertEqual(repair.validate_yeo_map(path)['labels'], list(range(8)))
        path.write_bytes(nifti_bytes((2, 2, 2)))
        with self.assertRaises(ValueError):
            repair.validate_yeo_map(path)

    def test_yeo_legacy_offset_and_unset_bitpix(self):
        path = self.root / 'legacy.nii.gz'
        raw = bytearray(gzip.decompress(nifti_bytes((2, 2, 2)))[:348])
        struct.pack_into('<f', raw, 108, 348)
        struct.pack_into('<h', raw, 72, 0)
        path.write_bytes(gzip.compress(bytes(raw) + np.arange(8, dtype='<f4').tobytes()))
        original = path.read_bytes()
        result = repair.validate_yeo_map(path)
        self.assertTrue(result['legacy_header_interpretation'])
        self.assertEqual(result['raw_vox_offset'], 348)
        self.assertEqual(path.read_bytes(), original)

    def test_actual_mirror_bitpix_error_normalizes_without_changing_voxels(self):
        path = self.root / 'atlas.nii.gz'
        raw = bytearray(gzip.decompress(nifti_bytes((2, 2, 2, 1)))[:352])
        struct.pack_into('<h', raw, 72, 4)  # Actual mirror bug: bytes instead of bits.
        payload = np.arange(8, dtype='<f4').tobytes()
        path.write_bytes(gzip.compress(bytes(raw) + payload))
        original = path.read_bytes()
        result = repair.normalized_yeo_copy(path)
        fixed = gzip.decompress(Path(result['path']).read_bytes())
        self.assertEqual(struct.unpack_from('<h', fixed, 72)[0], 32)
        self.assertEqual(fixed[352:], payload)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(fixed[252:328], bytes(raw)[252:328])

    def test_metadata_repair_does_not_request_derivatives(self):
        dataset = self.root / 'input'
        metadata = dataset / 'metadata'
        metadata.mkdir(parents=True)
        pd.read_csv(io.BytesIO(PHENO)).iloc[:1].to_csv(metadata / 'participants_requested.csv', index=False)
        (metadata / 'scan_headers.json').write_text(json.dumps({'1': {
            'shape': [2, 2, 2, 3], 'TR_seconds': None, 'time_unit_code': 0}}))
        resources = [{'resource': name, 'status': 'downloaded', 'url': 'already downloaded'} for name in (
            'CC200.nii.gz', 'Yeo_JNeurophysiol11_MNI152.zip',
            'pipeline_config_abide_rerun.yml', 'abide_rois.txt')]
        (metadata / 'resource_manifest.json').write_text(json.dumps(resources))
        source_path = 'ABIDE/scan_jsons/Site/sub-0000001_task-rest_bold.json'
        def open_fixture(request, **kwargs):
            url = request.full_url
            if url == repair.TREE_URL:
                return Response(json.dumps({'sha': 'a' * 40, 'truncated': False,
                    'tree': [{'type': 'blob', 'path': source_path}]}).encode())
            self.assertIn('/' + 'a' * 40 + '/' + source_path, url)
            return Response(b'{"RepetitionTime": 2.5}')
        patch_root = self.root / 'patch'
        with patch.object(download.urllib.request, 'urlopen', side_effect=open_fixture) as opener, \
                contextlib.redirect_stdout(io.StringIO()):
            result = repair.repair_metadata(dataset, patch_root)
        self.assertEqual(opener.call_count, 2)
        self.assertEqual(result.TR_seconds.tolist(), [2.5])
        summary = json.loads((patch_root / 'repair_summary.json').read_text())
        self.assertTrue(summary['metadata_repair_complete'])
        self.assertEqual(summary['participant_derivatives_redownloaded'], 0)

    def test_roi_nonfinite_and_labels(self):
        path = self.root / 'rois.1D'
        path.write_text('#1\t#2\n1 2\n3 4\n')
        details = download.validate_roi(path)
        self.assertEqual(details['roi_labels'], ['1', '2'])
        self.assertEqual(details['n_volumes'], 2)
        self.assertNotEqual(details['warning'], '')
        path.write_text('1 nan\n2 3\n')
        with self.assertRaises(ValueError):
            download.validate_roi(path)

    def test_mocked_full_run_exports_metadata_before_zip(self):
        def open_fixture(request, **kwargs):
            url = request.full_url
            if url == download.PHENO_URL:
                return Response(PHENO)
            if url.endswith('.1D'):
                values = np.arange(600).reshape(3, 200)
                text = io.StringIO()
                np.savetxt(text, values, header=' '.join(str(i) for i in range(1, 201)))
                return Response(text.getvalue().encode())
            if '/func_preproc/' in url:
                return Response(nifti_bytes((2, 2, 2, 3)))
            if url.endswith('.nii.gz'):
                return Response(nifti_bytes())
            if url == download.YEO_URL:
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w') as archive:
                    archive.writestr('Yeo2011_7Networks.nii.gz', nifti_bytes())
                return Response(buffer.getvalue())
            return Response(b'upstream provenance fixture\n')
        def check_archive(*args):
            self.assertTrue((self.root / 'README.md').exists())
            self.assertTrue((self.root / 'metadata/completion_summary.json').exists())
        with patch.object(download.urllib.request, 'urlopen', side_effect=open_fixture), \
                patch.object(download.shutil, 'make_archive', side_effect=check_archive), \
                contextlib.redirect_stdout(io.StringIO()):
            complete = download.run_acquisition(self.root, [1, 2, 999], create_zip=True)
        self.assertEqual(complete.SUB_ID.tolist(), [1, 2, 999])
        self.assertEqual(complete.all_four_strategies_complete.tolist(), [True, True, False])
        manifest = json.loads((self.root / 'metadata/download_manifest.json').read_text())
        self.assertEqual(len(manifest), 30)
        self.assertEqual(sum(r['status'] in download.SUCCESS for r in manifest), 20)
        summary = json.loads((self.root / 'metadata/completion_summary.json').read_text())
        self.assertFalse(summary['acquisition_complete'])
        self.assertEqual(summary['subjects_with_TR'], 2)
        self.assertTrue((self.root / 'metadata/participants_missing_normalized.csv').exists())

    def test_notebook_is_standalone_and_matches_tested_helpers(self):
        folder = Path(__file__).parent
        nb = json.loads((folder / 'abide-gsr-downloader.ipynb').read_text())
        helpers = next(c for c in nb['cells'] if c.get('id') == 'download-helpers')
        self.assertEqual(''.join(helpers['source']), (folder / 'abide_download.py').read_text())
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), '<notebook cell>', 'exec')
                self.assertEqual(cell['outputs'], [])


if __name__ == '__main__':
    unittest.main()
