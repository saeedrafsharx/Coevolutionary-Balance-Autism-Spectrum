"""Scientific array-level checks; NiBabel/image I/O is validated on Kaggle."""
import tempfile
import unittest
import contextlib
import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from abide_stage1 import (anatomical_mapping, efficiency, energy, features, flag_iqr,
                         median_states, read_timeseries, resample_labels, roi_falff,
                         run_stage1, derivative, resolve_nifti)


class Stage1Tests(unittest.TestCase):
    def test_kaggle_nifti_extraction_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            nested = root / 'CC200.nii/CC200.nii'
            nested.parent.mkdir()
            nested.write_bytes(b'fixture')
            self.assertEqual(resolve_nifti(root / 'CC200.nii.gz'), nested)
            plain = root / 'Yeo.normalized.nii'
            plain.write_bytes(b'fixture')
            self.assertEqual(resolve_nifti(root / 'Yeo.normalized.nii.gz'), plain)
            compressed = root / 'map.nii.gz'
            compressed.write_bytes(b'fixture')
            self.assertEqual(resolve_nifti(compressed), compressed)
            with self.assertRaises(FileNotFoundError):
                resolve_nifti(root / 'missing.nii.gz')

    def test_roi_header_reorders_by_real_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'roi.1D'
            values = np.column_stack([np.arange(50), np.arange(50) * 2])
            np.savetxt(path, values, header='20 10')
            aligned, _ = read_timeseries(path, np.array([10, 20]))
            np.testing.assert_array_equal(aligned, values[:, ::-1])
            with self.assertRaises(ValueError):
                read_timeseries(path, np.array([1, 2]))

    def test_resampling_uses_affine_and_nearest_labels(self):
        source = np.arange(27).reshape(3, 3, 3)
        target_affine = np.eye(4)
        target_affine[0, 3] = 1
        result = resample_labels(source, np.eye(4), (2, 3, 3), target_affine)
        np.testing.assert_array_equal(result, source[1:, :, :])

    def test_dice_mapping_is_spatial_and_does_not_force_uncovered_nodes(self):
        cc = np.array([[[10, 10, 20, 20, 30, 30]]])
        yeo = np.array([[[2, 2, 1, 1, 0, 0]]])
        mapping = anatomical_mapping(cc, yeo, np.eye(4))
        self.assertEqual(mapping.network_id.tolist(), [2, 1, 0])

    def test_masked_roi_means_and_coverage(self):
        cc = np.array([[[10, 10, 20, 20]]])
        values = np.array([[[.1, .9, .3, .5]]])
        mask = np.array([[[1, 0, 1, 1]]])
        means, audit = roi_falff(values, cc, mask, [10, 20])
        np.testing.assert_allclose(means, [.1, .4])
        self.assertEqual(audit[0]['coverage_fraction'], .5)
        with self.assertRaises(ValueError):
            roi_falff(values, cc, mask, [10, 20, 999])

    def test_block_energies_and_unassigned_contribution_reconcile(self):
        fc = np.array([[0, .3, -.2, .1], [.3, 0, .4, -.5], [-.2, .4, 0, .6], [.1, -.5, .6, 0]])
        states = np.array([1, 1, -1, -1])
        result = features(fc, states, np.array([1, 1, 2, 0]), compute_local=False)
        block_sum = sum(v for k, v in result.items() if k.endswith('_EnergyBlock') and np.isfinite(v))
        self.assertAlmostEqual(result['Whole_Energy_global'], energy(fc, states))
        self.assertAlmostEqual(block_sum + result['QC_Energy_involving_unassigned_ROIs'], energy(fc, states))
        self.assertAlmostEqual(sum(v for k, v in result.items() if k.startswith('Whole_Prop_')), 1)

    def test_iqr_is_pooled_inclusive_and_nonmutating(self):
        frame = pd.DataFrame({'H': [0, 1, 2, 3, 100], 'DX': [1, 1, 1, 2, 2]})
        before = frame.copy(deep=True)
        flags, fence = flag_iqr(frame, 'H')
        self.assertEqual(flags.tolist(), [False, False, False, False, True])
        self.assertEqual(fence['upper_fence'], 6)
        pd.testing.assert_frame_equal(frame, before)

    def test_efficiency_for_three_node_chain(self):
        fc = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        g, local = efficiency(fc, local=True)
        self.assertAlmostEqual(g, 5/6)
        self.assertAlmostEqual(local, 0)

    def test_median_ties_are_retained_explicitly(self):
        self.assertEqual(median_states(np.array([1, 2, 2, 3])).tolist(), [-1, 1, 1, 1])

    def test_qc_options_observe_invalid_ROIs_without_changing_default_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'roi.1D'
            np.savetxt(path, np.column_stack([np.ones(50), np.arange(50)]), header='1 2')
            with self.assertRaises(ValueError):
                read_timeseries(path, [1, 2])
            ts, _ = read_timeseries(path, [1, 2], allow_constant=True)
            self.assertEqual(np.ptp(ts[:, 0]), 0)
        means, audit = roi_falff(np.array([[[.1, .2]]]), np.array([[[1, 2]]]),
                                 np.array([[[0, 1]]]), [1, 2], allow_missing=True)
        self.assertTrue(np.isnan(means[0]))
        self.assertEqual(audit[0]['coverage_fraction'], 0)

    def test_qc_notebook_contains_tested_helpers(self):
        folder = Path(__file__).parent
        nb = json.loads((folder / 'abide-revision-stage1b-qc.ipynb').read_text())
        for cell in nb['cells']:
            if cell.get('id') == 'qc-helpers':
                self.assertEqual(''.join(cell['source']), (folder / 'abide_qc_followup.py').read_text())
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')

    def test_qc_followup_exports_constant_and_zero_support_cases(self):
        import abide_qc_followup as qc
        with tempfile.TemporaryDirectory() as folder:
            root, out = Path(folder) / 'input', Path(folder) / 'output'
            (root / 'metadata').mkdir(parents=True)
            pd.DataFrame([{'SUB_ID': 1, 'FILE_ID': 'Site_0000001', 'SITE_ID': 'Site', 'DX_GROUP': 1}]).to_csv(
                root / 'metadata/participants_requested.csv', index=False)
            atlas = root / 'atlases/CC200.nii.gz'
            atlas.parent.mkdir()
            atlas.write_bytes(b'fixture')
            for strategy in ('filt_global', 'filt_noglobal'):
                path = derivative(root, 'Site_0000001', strategy, 'rois_cc200')
                path.parent.mkdir(parents=True)
                np.savetxt(path, np.column_stack([np.ones(50), np.arange(50)]), header='1 2')
            for strategy in ('nofilt_global', 'nofilt_noglobal'):
                for kind in ('falff', 'func_mask'):
                    path = derivative(root, 'Site_0000001', strategy, kind)
                    path.parent.mkdir(parents=True)
                    path.write_bytes(b'fixture')
            def image_fixture(path):
                if 'func_mask' in str(path):
                    values = np.array([[[0, 1]]])
                elif 'falff' in str(path):
                    values = np.array([[[.1, .2]]])
                else:
                    values = np.array([[[1, 2]]])
                return values, np.eye(4)
            with patch('abide_qc_followup.spatial_image', side_effect=image_fixture), \
                    contextlib.redirect_stdout(io.StringIO()):
                qc.run_qc_followup(root, out)
            rows = pd.read_csv(out / 'all_participant_ROI_QC.csv')
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows.constant_time_series.sum(), 2)
            self.assertEqual(rows.finite_masked_voxels.eq(0).sum(), 2)
            summary = json.loads((out / 'qc_summary.json').read_text())
            self.assertEqual(summary['participant_exclusions_applied'], 0)

    def test_mocked_image_io_complete_pipeline_exports_untrimmed_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            root, supplement, out = base / 'input', base / 'patch', base / 'output'
            (root / 'metadata').mkdir(parents=True)
            supplement.mkdir()
            pd.DataFrame([{'SUB_ID': 1, 'FILE_ID': 'Site_0000001', 'SITE_ID': 'Site',
                           'DX_GROUP': 1}]).to_csv(root / 'metadata/participants_requested.csv', index=False)
            pd.DataFrame([{'SUB_ID': 1, 'TR_seconds': 2}]).to_csv(supplement / 'TR_recovery.csv', index=False)
            paths = [root / 'atlases/CC200.nii.gz']
            for suffix in ('', '_LiberalMask'):
                paths.append(supplement / 'atlases/Yeo7_mirror' /
                             f'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm{suffix}.normalized.nii.gz')
            for strategy in ('filt_global', 'filt_noglobal', 'nofilt_global', 'nofilt_noglobal'):
                paths.append(derivative(root, 'Site_0000001', strategy, 'falff'))
                if strategy.startswith('nofilt'):
                    paths.append(derivative(root, 'Site_0000001', strategy, 'func_mask'))
                else:
                    path = derivative(root, 'Site_0000001', strategy, 'rois_cc200')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savetxt(path, np.random.default_rng(3).normal(size=(50, 7)), header='1 2 3 4 5 6 7')
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'mocked image bytes')
            labels = np.arange(8).reshape(2, 2, 2)
            def image_fixture(path):
                if 'func_mask' in str(path):
                    values = np.ones((2, 2, 2))
                elif 'falff' in str(path):
                    values = (labels + 1) / 10
                else:
                    values = labels
                return values, np.eye(4)
            nib = SimpleNamespace(__version__='test-stub', Nifti1Image=lambda a, affine: a,
                                  save=lambda image, path: Path(path).write_bytes(b'mocked saved image'))
            with patch.dict(sys.modules, {'nibabel': nib}), \
                    patch('abide_stage1.spatial_image', side_effect=image_fixture), \
                    contextlib.redirect_stdout(io.StringIO()):
                summary = run_stage1(root, supplement, out, {'GSR': [1], 'noGSR': [1]}, compute_local_efficiency=False)
            self.assertTrue(summary['technical_extraction_complete'])
            self.assertEqual(summary['outliers_removed'], 0)
            self.assertEqual(summary['common_technical_participants'], 1)
            self.assertFalse(summary['group_tests_performed'])
            self.assertTrue((out / 'features/features_GSR_untrimmed.csv').is_file())
            old = pd.read_csv(out / 'audit/legacy_trimming_reconstruction_GSR.csv')
            self.assertTrue(old.reconstructed_vs_historical_mask_agrees.all())

    def test_stage1_notebook_contains_tested_helpers_and_compiles(self):
        folder = Path(__file__).parent
        nb = json.loads((folder / 'abide-revision-stage1.ipynb').read_text())
        for cell in nb['cells']:
            if cell.get('id') == 'stage1-helpers':
                self.assertEqual(''.join(cell['source']), (folder / 'abide_stage1.py').read_text())
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')


if __name__ == '__main__':
    unittest.main()
