import os
import glob
import random
import numpy as np
import networkx as nx
from scipy.stats import ttest_rel, wilcoxon, shapiro
from statsmodels.stats.multitest import multipletests
from numba import njit

# --------------------------
# PARAMETERS
# --------------------------
network_dir = 'directory-to-observed-networks'
n_nulls = 1000
random_seed = 42

# --------------------------
# 1) Numba-accelerated energy computation for edge lists
# --------------------------
@njit
def compute_energy_numba(falff, edges_u, edges_v, weights):
    total = 0.0
    for k in range(edges_u.shape[0]):
        i = edges_u[k]
        j = edges_v[k]
        total += falff[i] * weights[k] * falff[j]
    return -total

# --------------------------
# 2) Outlier detection (paired samples) using IQR
# --------------------------
def detect_outliers_iqr(differences, factor=1.5):
    q1, q3 = np.percentile(differences, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return (differences < lower) | (differences > upper)

# --------------------------
# MAIN: Null-distribution sampling & statistics
# --------------------------
random.seed(random_seed)
np.random.seed(random_seed)

graph_files = glob.glob(os.path.join(network_dir, "*.graphml"))
if not graph_files:
    raise RuntimeError("No graphml files found in directory")

# containers for results
obs_energies = []
null_means   = []
z_scores     = []
emp_p_values = []

for gf in graph_files:
    G = nx.read_graphml(gf)
    # node to index mapping
    nodes = list(G.nodes())
    idx_map = {node: i for i, node in enumerate(nodes)}
    # fALFF vector
    falff = np.array([float(G.nodes[n]['fALFF']) for n in nodes], dtype=np.float64)
    # edge lists
    edges_u, edges_v, weights = [], [], []
    for u, v, data in G.edges(data=True):
        if 'weight' not in data:
            continue
        edges_u.append(idx_map[u])
        edges_v.append(idx_map[v])
        weights.append(float(data['weight']))
    edges_u = np.array(edges_u, dtype=np.int64)
    edges_v = np.array(edges_v, dtype=np.int64)
    weights = np.array(weights, dtype=np.float64)

    # observed energy
    obs_E = compute_energy_numba(falff, edges_u, edges_v, weights)

    # generate null distribution by permuting falff and weights
    null_Es = np.empty(n_nulls, dtype=np.float64)
    for i in range(n_nulls):
        perm_f = np.random.permutation(falff)
        perm_w = np.random.permutation(weights)
        null_Es[i] = compute_energy_numba(perm_f, edges_u, edges_v, perm_w)
    mu_null = null_Es.mean()
    sd_null = null_Es.std(ddof=1)
    z = (obs_E - mu_null) / sd_null if sd_null > 0 else np.nan

    # empirical two-tailed p-value
    if obs_E >= mu_null:
        emp_p = (null_Es >= obs_E).sum() / n_nulls * 2
    else:
        emp_p = (null_Es <= obs_E).sum() / n_nulls * 2
    emp_p = min(emp_p, 1.0)

    obs_energies.append(obs_E)
    null_means.append(mu_null)
    z_scores.append(z)
    emp_p_values.append(emp_p)

# convert to arrays
e_obs   = np.array(obs_energies)
e_null  = np.array(null_means)
differences = e_obs - e_null

# outlier removal
mask      = detect_outliers_iqr(differences)
e_obs_cl  = e_obs[~mask]
e_null_cl = e_null[~mask]
diff_cl   = differences[~mask]
print(f"Final sample size after outlier removal: {len(diff_cl)}")

# normality test
sh = shapiro(diff_cl)
print(f"Shapiro–Wilk: W={sh.statistic:.3f}, p={sh.pvalue:.4g}")

# choose parametric vs nonparametric paired test
t_stat, t_pval, w_stat, w_pval = None, None, None, None
if sh.pvalue > 0.05:
    t_stat, t_pval = ttest_rel(e_obs_cl, e_null_cl)
    print(f"Paired t-test: t={t_stat:.3f}, p={t_pval:.4g}")
else:
    w_stat, w_pval = wilcoxon(e_obs_cl, e_null_cl)
    print(f"Wilcoxon signed-rank: W={w_stat:.3f}, p={w_pval:.4g}")

# robust Wilcoxon on cleaned differences
w_all, p_all = wilcoxon(diff_cl)
print(f"Robust Wilcoxon on all differences: W={w_all:.3f}, p={p_all:.4g}")

# FDR correction across empirical p-values
reject, pvals_fdr, _, _ = multipletests(emp_p_values, alpha=0.05, method='fdr_bh')
print(f"FDR-corrected p-values (empirical):")
for fname, pval_fdr, rej in zip(graph_files, pvals_fdr, reject):
    print(f" {os.path.basename(fname)}  p_fdr={pval_fdr:.4g}, significant={rej}")
