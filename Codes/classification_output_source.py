"""Select completed classification outputs without changing the original run."""
from pathlib import Path
import hashlib
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/abide_stage2_repaired'
EXT=ROOT/'results/abide_stage2_classification_extension'

def classification_statistics_directory():
    if not (EXT/'completion_summary.json').exists():return BASE/'statistics'
    summary=json.loads((EXT/'completion_summary.json').read_text())
    expected=hashlib.sha256((BASE/'features/all_designs_raw_untrimmed.csv').read_bytes()).hexdigest()
    if summary['feature_sha256']!=expected:raise ValueError('Classification extension refers to a different feature table')
    table=pd.read_csv(EXT/'statistics/classification_conditional_intervals.csv')
    if len(table)!=summary['completed_pipeline_model_pairs'] or not table.n_OOF_predictions.eq(140).all():
        raise ValueError('Classification extension summary is inconsistent')
    return EXT/'statistics'
