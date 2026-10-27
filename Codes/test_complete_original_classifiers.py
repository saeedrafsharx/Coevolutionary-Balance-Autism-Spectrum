"""Real-OOF checks for extension completeness and independent bootstrap parity."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from complete_original_classifiers import validate_predictions, conditional_summary
ROOT=Path(__file__).resolve().parents[1]

class ClassifierExtensionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata=pd.read_csv(ROOT/'results/abide_stage2_repaired/audit/demographic_manifest.csv')
        cls.pred=pd.read_csv(ROOT/'results/abide_stage2_repaired/statistics/classification_OOF_predictions.csv')
    def test_complete_actual_predictions(self):
        validate_predictions(self.pred,self.metadata)
    def test_missing_or_duplicate_prediction_rejected(self):
        with self.assertRaises(ValueError):validate_predictions(self.pred.iloc[1:],self.metadata)
        with self.assertRaises(ValueError):validate_predictions(pd.concat([self.pred,self.pred.iloc[:1]]),self.metadata)
    def test_crossing_site_and_incorrect_truth_rejected(self):
        crossed=self.pred.copy();crossed.loc[0,'fold']=4
        with self.assertRaises(ValueError):validate_predictions(crossed,self.metadata)
        wrong=self.pred.copy();wrong.loc[0,'y_true']=1-wrong.loc[0,'y_true']
        with self.assertRaises(ValueError):validate_predictions(wrong,self.metadata)
    def test_bootstrap_matches_independent_dataframe_resampling(self):
        block=self.pred[self.pred.pipeline.eq('GSR')&self.pred.model.eq('LogisticRegression')]
        rng=np.random.default_rng(303);sites=block.site.unique();values=[]
        for _ in range(100):
            sample=pd.concat([block[block.site.eq(site)] for site in rng.choice(sites,len(sites),replace=True)])
            if sample.y_true.nunique()==2:values.append(balanced_accuracy_score(sample.y_true,sample.y_pred))
        result=conditional_summary(block,draws=100)
        np.testing.assert_allclose([result['conditional_site_bootstrap_CI95_low'],result['conditional_site_bootstrap_CI95_high']],np.percentile(values,[2.5,97.5]),atol=1e-12)

if __name__=='__main__':unittest.main()
