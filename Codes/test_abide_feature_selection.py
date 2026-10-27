"""Train/test separation and deterministic representation checks for ML follow-up."""
import unittest
import numpy as np
from sklearn.base import clone
from abide_feature_selection import RedundancyPruner, RankedSelector

class FeatureSelectionChecks(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(17)
        self.x=rng.normal(size=(80,8));self.y=(self.x[:,0]+.2*rng.normal(size=80)>0).astype(int)
    def test_redundancy_keeps_first_signed_scaled_representative(self):
        a,b,c=self.x[:,:3].T
        x=np.column_stack([a,2*a,b,-b,c])
        pruner=RedundancyPruner().fit(x)
        np.testing.assert_array_equal(pruner.get_support(),[True,False,True,False,True])
        self.assertEqual(pruner.transform(x).shape,(80,3))
    def test_held_out_transform_does_not_refit_pruning_or_selection(self):
        for method in ['anova','mutual_info','l1']:
            selector=RankedSelector(method=method,k=3).fit(self.x,self.y)
            before=selector.get_support()
            selector.transform(np.random.default_rng(21).normal(size=(7,8))*100)
            np.testing.assert_array_equal(before,selector.get_support())
            pruner=RedundancyPruner().fit(self.x)
            before=pruner.get_support()
            pruner.transform(np.zeros((7,8)))
            np.testing.assert_array_equal(before,pruner.get_support())
    def test_anova_detects_training_signal_and_caps_k(self):
        selector=RankedSelector('anova',k=1).fit(self.x,self.y)
        self.assertTrue(selector.get_support()[0])
        self.assertEqual(selector.get_support().sum(),1)
        self.assertEqual(RankedSelector('anova',k=20).fit(self.x,self.y).get_support().sum(),8)
    def test_clones_and_schema_changes(self):
        for method in ['all','anova','mutual_info','l1']:
            a=clone(RankedSelector(method,k=3)).fit(self.x,self.y)
            b=clone(RankedSelector(method,k=3)).fit(self.x,self.y)
            np.testing.assert_array_equal(a.get_support(),b.get_support())
            with self.assertRaises(ValueError):a.transform(self.x[:,:-1])
        with self.assertRaises(ValueError):RedundancyPruner().fit(np.zeros((10,2)))

if __name__=='__main__':unittest.main()
