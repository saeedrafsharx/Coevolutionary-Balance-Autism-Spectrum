"""
Statistical Analysis for ASD vs Control Comparisons

Performs stratified group comparisons with FDR correction across
whole-brain, intra-network, and inter-network feature levels.

Usage:
    python statistical_analysis.py --input features.csv \
                                   --output results.csv \
                                   --plots output_dir/
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns


def categorize_features(columns):
    """Categorize features by level: Whole, Intra, Inter."""
    categories = {'Whole': [], 'Intra': [], 'Inter': []}
    for col in columns:
        if col.startswith('Whole_'):
            categories['Whole'].append(col)
        elif col.startswith('Intra_'):
            categories['Intra'].append(col)
        elif col.startswith('Inter_'):
            categories['Inter'].append(col)
    return categories


def adaptive_test(group1, group2, alpha=0.05):
    """
    Choose t-test or Mann-Whitney based on normality/variance.
    
    Returns: (test_name, statistic, p_value)
    """
    # Check normality (Shapiro-Wilk)
    _, p1 = stats.shapiro(group1) if len(group1) >= 3 else (0, 0)
    _, p2 = stats.shapiro(group2) if len(group2) >= 3 else (0, 0)
    normal = p1 > alpha and p2 > alpha
    
    # Check equal variance (Levene)
    _, p_var = stats.levene(group1, group2)
    equal_var = p_var > alpha
    
    if normal and equal_var:
        stat, p = stats.ttest_ind(group1, group2)
        return 't-test', stat, p
    else:
        stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        return 'Mann-Whitney U', stat, p


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled if pooled > 0 else np.nan


def run_comparisons(df, output_path, alpha=0.05):
    """Run stratified statistical comparisons."""
    asd = df[df['Group'] == 'ASD']
    ctrl = df[df['Group'] == 'Control']
    
    meta_cols = {'Subject', 'Group', 'Center', 'FIQ', 'IQ'}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    categories = categorize_features(feature_cols)
    
    all_results = []
    
    print("=" * 80)
    print("STATISTICAL COMPARISONS: ASD vs Control")
    print("=" * 80)
    
    for category, features in categories.items():
        if not features:
            continue
        
        print(f"\n{category.upper()} ({len(features)} features)")
        print("-" * 40)
        
        results = []
        for feat in features:
            asd_vals = asd[feat].dropna().values
            ctrl_vals = ctrl[feat].dropna().values
            
            if len(asd_vals) < 3 or len(ctrl_vals) < 3:
                continue
            
            test, stat, p = adaptive_test(asd_vals, ctrl_vals)
            d = cohens_d(asd_vals, ctrl_vals)
            
            results.append({
                'Feature': feat,
                'Category': category,
                'ASD_Mean': asd_vals.mean(),
                'Control_Mean': ctrl_vals.mean(),
                'ASD_Std': asd_vals.std(),
                'Control_Std': ctrl_vals.std(),
                'Test_Used': test,
                'p_value': p,
                'Effect_Size': d
            })
        
        cat_df = pd.DataFrame(results)
        
        # FDR correction within category
        reject, p_corr, _, _ = multipletests(cat_df['p_value'], method='fdr_bh')
        cat_df['p_corrected'] = p_corr
        cat_df['Significant_FDR'] = reject
        cat_df['Significant_Uncorrected'] = cat_df['p_value'] < alpha
        
        cat_df = cat_df.sort_values('p_corrected')
        
        # Print summary
        n_sig_uncorr = cat_df['Significant_Uncorrected'].sum()
        n_sig_fdr = cat_df['Significant_FDR'].sum()
        print(f"  Significant (uncorrected): {n_sig_uncorr}")
        print(f"  Significant (FDR):         {n_sig_fdr}")
        
        if n_sig_fdr > 0:
            print("\n  Top FDR-significant features:")
            top = cat_df[cat_df['Significant_FDR']].head(5)
            for _, row in top.iterrows():
                print(f"    {row['Feature']}: p_FDR={row['p_corrected']:.4f}, d={row['Effect_Size']:.2f}")
        
        all_results.append(cat_df)
    
    # Combine and save
    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(output_path, index=False)
    print(f"\n[INFO] Results saved to {output_path}")
    
    return final


def plot_significant_features(df, results, category, output_dir, max_plots=6):
    """Create boxplots for significant features."""
    sig = results[(results['Category'] == category) & 
                  (results['Significant_Uncorrected'])].head(max_plots)
    
    if len(sig) == 0:
        return
    
    n = len(sig)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, (_, row) in enumerate(sig.iterrows()):
        feat = row['Feature']
        data = df[['Group', feat]].dropna()
        
        sns.boxplot(data=data, x='Group', y=feat, ax=axes[idx],
                   palette={'Control': '#2E8B57', 'ASD': '#DC143C'})
        sns.stripplot(data=data, x='Group', y=feat, ax=axes[idx],
                     color='black', alpha=0.4, size=3)
        
        # Simplify title
        title = feat.replace('Whole_', '').replace('Intra_', '').replace('Inter_', '')
        title = title.replace('__', ' ↔ ').replace('_', ' ')
        axes[idx].set_title(f'{title}\np={row["p_value"]:.2e}', fontsize=9)
        axes[idx].set_xlabel('')
    
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{category}_significant_features.pdf', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Statistical analysis')
    parser.add_argument('--input', required=True, help='Features CSV')
    parser.add_argument('--output', required=True, help='Results CSV')
    parser.add_argument('--plots', default='.', help='Directory for plots')
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    results = run_comparisons(df, args.output)
    
    # Generate plots
    for cat in ['Whole', 'Intra', 'Inter']:
        plot_significant_features(df, results, cat, args.plots)
    
    print(f"[INFO] Plots saved to {args.plots}/")


if __name__ == '__main__':
    main()
