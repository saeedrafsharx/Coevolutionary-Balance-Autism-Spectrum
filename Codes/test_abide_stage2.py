"""Offline checks for the frozen design and statistical helpers.

The notebook additionally verifies statsmodels HC3 against the matrix formula.
"""
import json
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from abide_stage2 import (adjusted_pvalues, hedge_g, minimum_detectable_d,
                          model_design, null_energy_statistics, residual_partial_correlation, hc3_fit)
from abide_stage2_predictive import site_group_splits


class Stage2Tests(unittest.TestCase):
    def test_multiple_testing_corrections(self):
        np.testing.assert_allclose(adjusted_pvalues([.01, .04, .03], 'BH'), [.03, .04, .04])
        np.testing.assert_allclose(adjusted_pvalues([.01, .04, .03], 'Holm'), [.03, .06, .06])
        result = adjusted_pvalues([.01, np.nan, .04], 'Holm')
        self.assertTrue(np.isnan(result[1]))
        np.testing.assert_allclose(result[[0, 2]], [.02, .04])

    def test_model_preserves_group_and_covariates_explicitly(self):
        rng = np.random.default_rng(3)
        frame = pd.DataFrame({'DX_GROUP': np.tile([1, 2], 20), 'SITE_ID': np.repeat(['a', 'b', 'c', 'd'], 10),
                              'AGE_AT_SCAN': rng.uniform(18, 30, 40), 'FIQ': rng.uniform(90, 120, 40),
                              'func_mean_fd': rng.uniform(.01, .2, 40)})
        x = model_design(frame)
        self.assertIn('diagnosis_ASD', x)
        self.assertEqual(x.shape[1], 8)
        self.assertEqual(np.linalg.matrix_rank(x), 8)
        frame.loc[0, 'FIQ'] = -9999
        with self.assertRaises(ValueError):
            model_design(frame)

    def test_partial_correlation_removes_known_confound(self):
        rng = np.random.default_rng(4)
        z = rng.normal(size=200)
        a, b = 10*z+rng.normal(size=200), 10*z+rng.normal(size=200)
        r, p, df = residual_partial_correlation(a, b, np.column_stack([np.ones(200), z]))
        residual_a = a-np.column_stack([np.ones(200), z])@np.linalg.lstsq(np.column_stack([np.ones(200), z]), a, rcond=None)[0]
        residual_b = b-np.column_stack([np.ones(200), z])@np.linalg.lstsq(np.column_stack([np.ones(200), z]), b, rcond=None)[0]
        self.assertAlmostEqual(r, stats.pearsonr(residual_a, residual_b).statistic)
        self.assertEqual(df, 197)
        self.assertLess(abs(r), .2)

    def test_zero_variance_null_is_unavailable_not_significant(self):
        fc = np.ones((4, 4))-np.eye(4)
        row = null_energy_statistics(fc, np.ones(4), permutations=30)
        self.assertTrue(row['zero_null_variance'])
        self.assertIsNone(row['z'])
        self.assertIsNone(row['p_lower_energy'])

    def test_effect_size_and_power_are_directional_and_plausible(self):
        self.assertGreater(hedge_g(np.array([3., 4., 5.]), np.array([1., 2., 3.])), 0)
        self.assertGreater(minimum_detectable_d(60, 60), .45)
        self.assertLess(minimum_detectable_d(60, 60), .6)
        self.assertGreater(minimum_detectable_d(60, 60, alpha=.025), minimum_detectable_d(60, 60))

    def test_power_does_not_use_unstable_large_noncentral_t_survival_tail(self):
        with patch('abide_stage2.stats.nct.sf', side_effect=AssertionError('unstable tail must not be called')):
            self.assertTrue(np.isfinite(minimum_detectable_d(67, 69)))
            self.assertTrue(np.isfinite(minimum_detectable_d(30, 34, alpha=.025)))

    def test_hc3_matches_closed_form_two_group_variance(self):
        x = pd.DataFrame({'intercept': np.ones(8), 'diagnosis_ASD': [0]*4+[1]*4})
        row, _ = hc3_fit(np.array([0., 1., 2., 3., 1., 2., 3., 4.]), x)
        self.assertAlmostEqual(row['beta'], 1)
        self.assertAlmostEqual(row['SE_HC3'], np.sqrt(2*5/(16*(1-.25)**2)))
        self.assertEqual(row['df_residual'], 6)

    def test_repaired_site_folds_cover_all140_with_both_classes(self):
        repo = Path(__file__).resolve().parent.parent
        metadata = pd.read_csv(repo / 'revision/analysis_plan/primary_demographic_manifest_140.csv')
        groups, y = metadata.SITE_ID.to_numpy(), metadata.DX_GROUP.eq(1).astype(int).to_numpy()
        tested = []
        for train, test in site_group_splits(groups, y, 5):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            self.assertEqual(set(y[test]), {0, 1})
            for inner_train, inner_test in site_group_splits(groups[train], y[train], 3):
                self.assertEqual(set(y[train][inner_train]), {0, 1})
                self.assertEqual(set(y[train][inner_test]), {0, 1})
            tested.extend(test.tolist())
        self.assertEqual(sorted(tested), list(range(140)))

    def test_frozen_design_matches_both_pipeline_QC_and_demographics(self):
        repo = Path(__file__).resolve().parent.parent
        plan = json.loads((repo / 'revision/analysis_plan/stage2_frozen_design.json').read_text())
        self.assertEqual([(d['n_ROIs'], d['n_subjects']) for d in plan['designs']], [(106, 140), (181, 104), (200, 69)])
        qc = pd.read_csv(repo / 'results/abide_stage1b_qc_review/all_participant_ROI_QC.csv')
        for design in plan['designs']:
            subset = qc[qc.SUB_ID.isin(design['SUB_IDs']) & qc.ROI_label.isin(design['ROI_labels'])]
            self.assertEqual(len(subset), 2*design['n_ROIs']*design['n_subjects'])
            self.assertFalse(subset.constant_time_series.any())
            self.assertTrue(subset.coverage_fraction.ge(.5).all())
            self.assertTrue(subset.finite_masked_voxels.gt(0).all())

    def test_notebook_embeds_current_modules(self):
        folder = Path(__file__).parent
        notebook = json.loads((folder / 'abide-revision-stage2.ipynb').read_text())
        modules = {'stage1-helpers': 'abide_stage1.py', 'stage2-helpers': 'abide_stage2.py',
                   'predictive-helpers': 'abide_stage2_predictive.py', 'run-helpers': 'abide_stage2_run.py'}
        for cell in notebook['cells']:
            if cell.get('id') in modules:
                self.assertEqual(''.join(cell['source']), (folder / modules[cell['id']]).read_text())
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')


if __name__ == '__main__':
    unittest.main()
