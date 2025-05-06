import os
import glob
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, ttest_ind, ranksums

###############################################################################
# 1) COEVOLUTIONARY ENERGY FUNCTION
###############################################################################
def compute_energy_from_graph(graph):
    """
    Compute the 2-body coevolutionary energy:
      E = - sum_{(u,v)} [fALFF[u] * weight[u,v] * fALFF[v]],
    skipping edges if any value is NaN or None.
    """
    energy_sum = 0.0
    for u, v, data in graph.edges(data=True):
        falff_u = graph.nodes[u].get('fALFF', None)
        falff_v = graph.nodes[v].get('fALFF', None)
        w = data.get('weight', 0.0)

        # If any is None or NaN, skip this edge
        if falff_u is None or falff_v is None:
            continue
        if np.isnan(falff_u) or np.isnan(falff_v) or np.isnan(w):
            continue

        energy_sum += falff_u * w * falff_v

    # Final energy is negative sum
    E = -energy_sum

    # If E is somehow NaN, we'll detect it in the caller.
    return E

def cohen_d(x, y):
    """
    Compute Cohen's d for two independent samples x, y.
    d = (mean_x - mean_y) / pooled_std
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    nx_ = len(x)
    ny_ = len(y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    pooled_var = ((nx_-1)*var_x + (ny_-1)*var_y) / (nx_ + ny_ - 2)
    pooled_std = np.sqrt(pooled_var) if pooled_var>0 else 1e-12
    d = (mean_x - mean_y) / pooled_std
    return d

###############################################################################
# 2) MAIN SCRIPT
###############################################################################
def main():
    # Directories containing the .graphml files
    asd_dir = r"directory-to-ASD-networks"
    control_dir = r"directory-to-typical-networks"

    # Gather GraphML files
    asd_files = glob.glob(os.path.join(asd_dir, "*.graphml"))
    control_files = glob.glob(os.path.join(control_dir, "*.graphml"))

    if not asd_files:
        print("No ASD graphml files found in:", asd_dir)
    if not control_files:
        print("No Control graphml files found in:", control_dir)
    if (not asd_files) or (not control_files):
        return

    # Lists to store energies
    asd_energies = []
    control_energies = []

    # Process ASD
    for gf in asd_files:
        G = nx.read_graphml(gf)
        E = compute_energy_from_graph(G)
        if np.isnan(E):
            print(f"NaN energy encountered for {gf}, skipping.")
            continue
        asd_energies.append(E)

    # Process Control
    for gf in control_files:
        G = nx.read_graphml(gf)
        E = compute_energy_from_graph(G)
        if np.isnan(E):
            print(f"NaN energy encountered for {gf}, skipping.")
            continue
        control_energies.append(E)

    asd_energies = np.array(asd_energies, dtype=float)
    control_energies = np.array(control_energies, dtype=float)

    print(f"\nFinal sample sizes: ASD={len(asd_energies)}, Control={len(control_energies)}")
    if len(asd_energies)==0 or len(control_energies)==0:
        print("No valid data to compare. Exiting.")
        return
    # Optional for Debugging
    #print("ASD energies:\n", asd_energies)
    #print("Control energies:\n", control_energies)

    # Statistical tests
    print("\n=== Normality Tests (Shapiro–Wilk) ===")
    sw_asd = shapiro(asd_energies)
    sw_ctrl = shapiro(control_energies)
    print(f"ASD: W={sw_asd.statistic:.3f}, p={sw_asd.pvalue:.3g}")
    print(f"Control: W={sw_ctrl.statistic:.3f}, p={sw_ctrl.pvalue:.3g}")

    print("\n=== Levene Test (Equal Variances) ===")
    lev = levene(asd_energies, control_energies)
    print(f"Levene: stat={lev.statistic:.3f}, p={lev.pvalue:.3g}")

    print("\n=== t-test (independent) ===")
    ttest_res = ttest_ind(asd_energies, control_energies, equal_var=(lev.pvalue>0.05))
    print(f"t-test: t={ttest_res.statistic:.3f}, p={ttest_res.pvalue:.3g}")

    print("\n=== Wilcoxon rank-sum test ===")
    ranksum_res = ranksums(asd_energies, control_energies)
    print(f"Rank-sum: stat={ranksum_res.statistic:.3f}, p={ranksum_res.pvalue:.3g}")

    d_val = cohen_d(asd_energies, control_energies)
    print(f"\n=== Effect size (Cohen's d) ===")
    print(f"Cohen's d = {d_val:.3f}")

    # Plot
    import pandas as pd
    plt.figure(figsize=(6,6))
    group_list = ["ASD"]*len(asd_energies) + ["Control"]*len(control_energies)
    energy_list = np.concatenate([asd_energies, control_energies])
    df = pd.DataFrame({"Group": group_list, "Energy": energy_list})

    sns.boxplot(x="Group", y="Energy", data=df, showfliers=False,
                palette=["#dd5c96", "#3c9d5f"])
    sns.stripplot(x="Group", y="Energy", data=df, color="black", size=5,
                  alpha=0.6, jitter=True)
    plt.title("ASD vs. Control Energy Comparison")
    plt.ylabel("Energy")
    plt.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
