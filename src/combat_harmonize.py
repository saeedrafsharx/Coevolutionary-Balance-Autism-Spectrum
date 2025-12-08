"""
ComBat Harmonization for Multi-Site Batch Effect Correction

Applies per-feature ComBat harmonization to correct site-related
batch effects in ABIDE neuroimaging data.

Usage:
    python combat_harmonize.py --input features_with_site.csv \
                               --output features_harmonized.csv
"""

import argparse
import numpy as np
import pandas as pd

try:
    from neuroHarmonize import harmonizationLearn
    HARMONIZER = 'neuroHarmonize'
except ImportError:
    from neuroCombat import neuroCombat
    HARMONIZER = 'neuroCombat'


def add_phenotypic_data(features_path, phenotypic_path, output_path):
    """
    Merge phenotypic data (Site, IQ) with features.
    
    Args:
        features_path: Path to features CSV (needs 'Subject' column)
        phenotypic_path: Path to ABIDE Phenotypic_V1_0b.csv
        output_path: Output path for merged CSV
    """
    features = pd.read_csv(features_path)
    pheno = pd.read_csv(phenotypic_path)
    
    # Clean IQ (-9999 = missing in ABIDE)
    pheno['FIQ'] = pheno['FIQ'].replace(-9999, np.nan)
    mean_iq = pheno['FIQ'].dropna().mean()
    
    # Merge
    pheno_sub = pheno[['SUB_ID', 'SITE_ID', 'FIQ']].rename(columns={'SUB_ID': 'Subject'})
    merged = features.merge(pheno_sub, on='Subject', how='left')
    merged = merged.rename(columns={'SITE_ID': 'Center'})
    merged['IQ'] = merged['FIQ'].fillna(mean_iq)
    
    merged.to_csv(output_path, index=False)
    print(f"[INFO] Merged data saved to {output_path}")
    print(f"[INFO] Mean IQ for imputation: {mean_iq:.1f}")
    
    return merged


def harmonize_features(input_path, output_path, site_col='Center'):
    """
    Apply per-feature ComBat harmonization.
    
    Args:
        input_path: Input CSV with features and site info
        output_path: Output path for harmonized CSV
        site_col: Column name for site/batch variable
    """
    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded {input_path}: {df.shape}")
    
    # Identify metadata vs feature columns
    meta_cols = ['Subject', 'Group', 'Center', 'FIQ', 'IQ']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"[INFO] Features to harmonize: {len(feature_cols)}")
    
    # Build covariates
    covars = pd.DataFrame({
        'SITE': df[site_col].astype(str),
        'Group': df['Group'].map({'ASD': 1, 'Control': 0}),
        'IQ': pd.to_numeric(df['IQ'], errors='coerce')
    })
    
    # Impute missing covariates
    for col in ['Group', 'IQ']:
        if covars[col].isna().any():
            covars[col] = covars[col].fillna(covars[col].median())
    
    # Harmonize each feature
    harmonized = df[meta_cols].copy()
    kept, dropped = [], []
    
    for feat in feature_cols:
        x = pd.to_numeric(df[feat], errors='coerce')
        
        # Skip constant features
        if x.dropna().nunique() <= 1:
            dropped.append((feat, 'constant'))
            continue
        
        # Impute NaN
        x = x.fillna(x.median())
        data = x.values.reshape(-1, 1)
        
        try:
            if HARMONIZER == 'neuroHarmonize':
                _, adjusted = harmonizationLearn(data, covars)
                result = adjusted[:, 0]
            else:
                out = neuroCombat(dat=data.T, covars=covars, batch_col='SITE')
                result = out['data'].flatten()
            
            if np.isnan(result).all():
                dropped.append((feat, 'all NaN result'))
                continue
            
            harmonized[feat] = result
            kept.append(feat)
            
        except Exception as e:
            dropped.append((feat, str(e)[:50]))
    
    print(f"[INFO] Kept: {len(kept)}, Dropped: {len(dropped)}")
    
    harmonized.to_csv(output_path, index=False)
    print(f"[INFO] Harmonized data saved to {output_path}")
    
    return harmonized


def main():
    parser = argparse.ArgumentParser(description='ComBat harmonization')
    parser.add_argument('--input', required=True, help='Input CSV with site info')
    parser.add_argument('--output', required=True, help='Output harmonized CSV')
    parser.add_argument('--phenotypic', help='ABIDE phenotypic CSV (optional)')
    args = parser.parse_args()
    
    if args.phenotypic:
        # First add phenotypic data
        temp_path = args.input.replace('.csv', '_with_pheno.csv')
        add_phenotypic_data(args.input, args.phenotypic, temp_path)
        harmonize_features(temp_path, args.output)
    else:
        harmonize_features(args.input, args.output)


if __name__ == '__main__':
    main()
