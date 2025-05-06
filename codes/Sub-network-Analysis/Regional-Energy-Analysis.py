import os
import glob
import numpy as np
import nibabel as nib
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from nibabel.processing import resample_from_to
from scipy.stats import ttest_ind, ranksums
from statsmodels.stats.multitest import multipletests

###############################################################################
# 1) BUILD ROI->NETWORK MAP (CC200 -> YEO7) USING DICE COEFFICIENT
###############################################################################
def build_roi_to_network_map(cc200_file, yeo7_file):
    """
    For each ROI in the Craddock 200 atlas, compute the Dice coefficient
    with each Yeo7 network and assign to the network with highest Dice.
    Dice = 2*|ROI ∩ Net| / (|ROI| + |Net|)
    """
    cc200_img  = nib.load(cc200_file)
    cc200_data = cc200_img.get_fdata().astype(int)
    yeo_img    = nib.load(yeo7_file)
    yeo_data   = yeo_img.get_fdata().astype(int)

    # Resample Yeo7 to CC200 if needed
    if cc200_data.shape != yeo_data.shape:
        yeo_img    = resample_from_to(yeo_img, cc200_img, order=0)
        yeo_data   = yeo_img.get_fdata().astype(int)

    roi_ids = np.unique(cc200_data[cc200_data > 0])
    yeo_ids = np.unique(yeo_data[yeo_data > 0])
    # Precompute Yeo network sizes
    net_sizes = {net: np.sum(yeo_data == net) for net in yeo_ids}

    roi_to_net = {}
    for roi in roi_ids:
        roi_mask = (cc200_data == roi)
        roi_size = roi_mask.sum()
        best_net, best_dice = None, -1.0
        for net in yeo_ids:
            inter = np.sum(roi_mask & (yeo_data == net))
            denom = roi_size + net_sizes[net]
            if denom == 0:
                continue
            dice = 2.0 * inter / denom
            if dice > best_dice:
                best_dice, best_net = dice, int(net)
        roi_to_net[int(roi)] = best_net

    return roi_to_net

###############################################################################
# 2) COMPUTE SUB-NETWORK ENERGY (as before)
###############################################################################
def compute_network_energy(graph, roi_to_net, net_id):
    energy_sum = 0.0
    for u, v, data in graph.edges(data=True):
        try:
            u_int = int(u); v_int = int(v)
        except ValueError:
            continue
        if roi_to_net.get(u_int) == net_id and roi_to_net.get(v_int) == net_id:
            falff_u = graph.nodes[u].get('fALFF', None)
            falff_v = graph.nodes[v].get('fALFF', None)
            w       = data.get('weight', None)
            if falff_u is None or falff_v is None or w is None:
                continue
            if any(np.isnan([falff_u, falff_v, w])):
                continue
            energy_sum += falff_u * w * falff_v
    return -energy_sum

###############################################################################
# 3) IQR-BASED OUTLIER DETECTION
###############################################################################
def identify_outliers_iqr(values, iqr_factor=1.5):
    if len(values) < 2:
        return np.zeros(len(values), dtype=bool)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr
    return (values < lower) | (values > upper)

###############################################################################
# 4) PARSE SUBJECT ID
###############################################################################
def parse_subject_id(fname):
    base = os.path.basename(fname)
    no_ext = os.path.splitext(base)[0]
    parts = no_ext.split('_')
    return parts[1] if len(parts) >= 2 else no_ext

###############################################################################
# 5) MAIN
###############################################################################
def main():
    # (A) atlas paths
    cc200_file = 'directory-to-cc200-atlas'
    yeo7_file  = 'directory-to-yeo7-atlas'
    roi_to_net = build_roi_to_network_map(cc200_file, yeo7_file)

    # (B) GraphML directories
    asd_dir   = 'directory-to-asd-networks'
    ctrl_dir  = 'directory-to-control-networks'
    asd_files  = glob.glob(os.path.join(asd_dir, "*.graphml"))
    ctrl_files = glob.glob(os.path.join(ctrl_dir, "*.graphml"))

    # (C) Prepare containers
    network_ids    = [1,2,3,4,5,6,7]
    network_names  = {
        1: "Visual", 2: "SomatoMotor", 3: "DorsalAttn",
        4: "Salience/VentAttn", 5: "Limbic",
        6: "Frontoparietal", 7: "Default"
    }
    net_asd = {n: [] for n in network_ids}
    net_ctl = {n: [] for n in network_ids}

    # Compute energies
    for fn in asd_files:
        G = nx.read_graphml(fn)
        sid = parse_subject_id(fn)
        for n in network_ids:
            e = compute_network_energy(G, roi_to_net, n)
            if not np.isnan(e):
                net_asd[n].append((sid, e))

    for fn in ctrl_files:
        G = nx.read_graphml(fn)
        sid = parse_subject_id(fn)
        for n in network_ids:
            e = compute_network_energy(G, roi_to_net, n)
            if not np.isnan(e):
                net_ctl[n].append((sid, e))

    # Outlier removal
    outliers = set()
    for n in network_ids:
        combined = net_asd[n] + net_ctl[n]
        if len(combined) < 2:
            continue
        vals   = np.array([v for (_,v) in combined], float)
        mask   = identify_outliers_iqr(vals)
        for i, m in enumerate(mask):
            if m:
                outliers.add(combined[i][0])
    for n in network_ids:
        net_asd[n] = [(sid,v) for sid,v in net_asd[n] if sid not in outliers]
        net_ctl[n] = [(sid,v) for sid,v in net_ctl[n] if sid not in outliers]

    # Stats + plotting preparation
    results = []
    all_data = []
    for n in network_ids:
        name = network_names[n]
        asd_vals = [v for _,v in net_asd[n]]
        ctl_vals = [v for _,v in net_ctl[n]]
        if len(asd_vals) < 2 or len(ctl_vals) < 2:
            print(f"Not enough data for {name}")
            continue
        t_res = ttest_ind(asd_vals, ctl_vals, equal_var=False)
        r_res = ranksums( asd_vals, ctl_vals )
        results.append({
            "NetworkID":  n,
            "NetworkName": name,
            "ASD_n":      len(asd_vals),
            "CTRL_n":     len(ctl_vals),
            "MeanASD":    np.mean(asd_vals),
            "MeanCTRL":   np.mean(ctl_vals),
            "TTest_t":    t_res.statistic,
            "TTest_p":    t_res.pvalue,
            "RankSum_stat": r_res.statistic,
            "RankSum_p":    r_res.pvalue
        })
        for v in asd_vals:
            all_data.append({"Group":"ASD", "Network":name, "Energy":v})
        for v in ctl_vals:
            all_data.append({"Group":"Control", "Network":name, "Energy":v})

    # Build DataFrame
    results_df = pd.DataFrame(results)

    # --- MULTIPLE-COMPARISON ADJUSTMENT (FDR) ---
    p_t = results_df["TTest_p"].values
    p_r = results_df["RankSum_p"].values
    _, p_t_fdr, _, _ = multipletests(p_t, alpha=0.05, method='fdr_bh')
    _, p_r_fdr, _, _ = multipletests(p_r, alpha=0.05, method='fdr_bh')
    results_df["TTest_p_fdr"]   = p_t_fdr
    results_df["RankSum_p_fdr"] = p_r_fdr
    results_df["TTest_sig_fdr"]   = results_df["TTest_p_fdr"] < 0.05
    results_df["RankSum_sig_fdr"] = results_df["RankSum_p_fdr"] < 0.05
    # -------------------------------------------

    # Output
    print("\n=== Network-level comparison (with FDR) ===")
    print(results_df)

    # Plot
    if all_data:
        df_all = pd.DataFrame(all_data)
        plt.figure(figsize=(10,6))
        sns.boxplot(x="Network", y="Energy", hue="Group",
                    data=df_all, showfliers=False)
        sns.stripplot(x="Network", y="Energy", hue="Group",
                      data=df_all, dodge=True, alpha=0.6,
                      color="black", size=3)
        plt.title("Network Energy (Outliers Excluded): ASD vs Control")
        plt.xlabel("Yeo7 Network")
        plt.ylabel("Energy")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
