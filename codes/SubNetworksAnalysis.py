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

###############################################################################
# 1) BUILD ROI->NETWORK MAP (CC200 -> YEO7)
###############################################################################
def build_roi_to_network_map(cc200_file, yeo7_file):
    """
    For each ROI in the Craddock 200 atlas (cc200_file),
    find which Yeo7 network (1..7) it overlaps the most.
    Returns a dict: roi_to_net[roi_id] = net_id (1..7).
    """
    print("Loading CC200 atlas:", cc200_file)
    cc200_img = nib.load(cc200_file)
    cc200_data = cc200_img.get_fdata()

    print("Loading Yeo7 atlas:", yeo7_file)
    yeo_img = nib.load(yeo7_file)
    yeo_data = yeo_img.get_fdata()

    # Resample Yeo7 if needed
    if cc200_data.shape != yeo_data.shape:
        print("Resampling Yeo7 to CC200 resolution...")
        yeo_img = resample_from_to(yeo_img, cc200_img, order=0)
        yeo_data = yeo_img.get_fdata()

    roi_ids = np.unique(cc200_data[cc200_data>0])  # e.g. 1..200
    yeo_ids = np.unique(yeo_data[yeo_data>0])      # e.g. 1..7

    print(f"Found {len(roi_ids)} CC200 ROIs, {len(yeo_ids)} Yeo7 networks.")

    roi_to_net = {}
    for roi in roi_ids:
        mask = (cc200_data == roi)
        overlap_counts = {}
        for net in yeo_ids:
            overlap_counts[net] = np.sum(yeo_data[mask] == net)
        # find net with maximum overlap
        best_net = max(overlap_counts, key=overlap_counts.get)
        roi_to_net[int(roi)] = int(best_net)

    return roi_to_net

###############################################################################
# 2) COMPUTE SUB-NETWORK ENERGY (SKIP NaN)
###############################################################################
def compute_network_energy(graph, roi_to_net, net_id):
    """
    E_net = - sum_{(u,v)} [fALFF[u]*weight[u,v]*fALFF[v]]
    for edges (u,v) where roi_to_net[u_int] == net_id and roi_to_net[v_int] == net_id.
    Skips edges if any relevant value is None or NaN.
    """
    energy_sum = 0.0
    
    for u, v, data in graph.edges(data=True):
        # Convert string IDs to int
        try:
            u_int = int(u)
            v_int = int(v)
        except ValueError:
            continue

        if roi_to_net.get(u_int, None) == net_id and roi_to_net.get(v_int, None) == net_id:
            falff_u = graph.nodes[u].get('fALFF', None)
            falff_v = graph.nodes[v].get('fALFF', None)
            w = data.get('weight', None)

            # Skip if missing or NaN
            if (falff_u is None) or (falff_v is None) or (w is None):
                continue
            if any(np.isnan([falff_u, falff_v, w])):
                continue

            energy_sum += falff_u * w * falff_v

    return -energy_sum

###############################################################################
# 3) IQR-BASED OUTLIER DETECTION
###############################################################################
def identify_outliers_iqr(values, iqr_factor=1.5):
    """
    Returns a boolean mask of outliers:
      below Q1 - iqr_factor*IQR or above Q3 + iqr_factor*IQR.
    """
    if len(values) < 2:
        return np.zeros(len(values), dtype=bool)

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr

    return (values < lower_bound) | (values > upper_bound)

###############################################################################
# 4) PARSE SUBJECT ID FROM FILENAME
###############################################################################
def parse_subject_id(fname):
    """
    E.g. "Pitt_0050007_network.graphml" => subject_id = "0050007"
    Adjust if your naming pattern differs.
    """
    base = os.path.basename(fname)
    no_ext = os.path.splitext(base)[0]
    parts = no_ext.split('_')
    if len(parts) >= 2:
        return parts[1]
    else:
        return no_ext

###############################################################################
# 5) MAIN
###############################################################################
def main():
    # (A) Provide atlas paths
    cc200_file = r"directory-to-atlas-file"
    yeo7_file  = r"directory-to-yeo7-file"

    # Build the ROI->network map
    roi_to_net = build_roi_to_network_map(cc200_file, yeo7_file)

    # (B) Directories for ASD & Control GraphML
    asd_dir   = r"directory-to-ASD-networks"
    ctrl_dir  = r"directory-to-typical-networks"

    asd_files  = glob.glob(os.path.join(asd_dir, "*.graphml"))
    ctrl_files = glob.glob(os.path.join(ctrl_dir, "*.graphml"))
    
    if not asd_files and not ctrl_files:
        print("No GraphML files found in either directory. Exiting.")
        return

    # Yeo 7 networks
    network_ids = [1,2,3,4,5,6,7]
    network_labels = {
        1: "Visual",
        2: "SomatoMotor",
        3: "DorsalAttn",
        4: "Salience/VentAttn",
        5: "Limbic",
        6: "Frontoparietal",
        7: "Default"
    }

    # We'll store sub-network energies as:
    # net_energies_asd[net] = list of (subject_id, energy)
    # net_energies_ctrl[net] = list of (subject_id, energy)
    net_energies_asd = {net: [] for net in network_ids}
    net_energies_ctrl = {net: [] for net in network_ids}

    # (C) Compute sub-network energies for ASD
    for gf in asd_files:
        G = nx.read_graphml(gf)
        subject_id = parse_subject_id(gf)
        for net in network_ids:
            E_val = compute_network_energy(G, roi_to_net, net)
            if not np.isnan(E_val):
                net_energies_asd[net].append((subject_id, E_val))

    # (D) Compute sub-network energies for Control
    for gf in ctrl_files:
        G = nx.read_graphml(gf)
        subject_id = parse_subject_id(gf)
        for net in network_ids:
            E_val = compute_network_energy(G, roi_to_net, net)
            if not np.isnan(E_val):
                net_energies_ctrl[net].append((subject_id, E_val))

    # (E) Identify outliers for each network, gather them in a set => remove from all networks
    outlier_subjects = set()

    for net in network_ids:
        # Combine ASD + Control for this net
        combined_data = net_energies_asd[net] + net_energies_ctrl[net]
        if len(combined_data) < 2:
            continue

        energies = np.array([item[1] for item in combined_data], dtype=float)
        mask = identify_outliers_iqr(energies, iqr_factor=1.5)
        for idx, is_out in enumerate(mask):
            if is_out:
                subj_id = combined_data[idx][0]
                outlier_subjects.add(subj_id)

    print(f"Found {len(outlier_subjects)} outlier subjects across all networks.")
    if outlier_subjects:
        print("Outlier subject IDs:")
        for s in sorted(outlier_subjects):
            print("  ", s)

    # (F) Remove outliers from all networks
    for net in network_ids:
        # ASD
        new_asd_list = []
        for (subj_id, val) in net_energies_asd[net]:
            if subj_id not in outlier_subjects:
                new_asd_list.append((subj_id, val))
        net_energies_asd[net] = new_asd_list

        # Control
        new_ctrl_list = []
        for (subj_id, val) in net_energies_ctrl[net]:
            if subj_id not in outlier_subjects:
                new_ctrl_list.append((subj_id, val))
        net_energies_ctrl[net] = new_ctrl_list

    # (G) Final stats + plotting
    results = []
    all_data = []

    for net in network_ids:
        net_name = network_labels.get(net, f"Net{net}")

        asd_vals = [x[1] for x in net_energies_asd[net]]
        ctrl_vals = [x[1] for x in net_energies_ctrl[net]]

        n_asd = len(asd_vals)
        n_ctrl = len(ctrl_vals)
        if n_asd < 2 or n_ctrl < 2:
            print(f"Not enough valid data for network {net_name} to compare.")
            continue

        t_res = ttest_ind(asd_vals, ctrl_vals, equal_var=False)
        r_res = ranksums(asd_vals, ctrl_vals)

        row = {
            "NetworkID": net,
            "NetworkName": net_name,
            "ASD_n": n_asd,
            "CTRL_n": n_ctrl,
            "MeanASD": np.mean(asd_vals),
            "MeanCTRL": np.mean(ctrl_vals),
            "TTest_t": t_res.statistic,
            "TTest_p": t_res.pvalue,
            "RankSum_stat": r_res.statistic,
            "RankSum_p": r_res.pvalue
        }
        results.append(row)

        # For plotting
        for val in asd_vals:
            all_data.append({"Group":"ASD", "Network":net_name, "Energy":val})
        for val in ctrl_vals:
            all_data.append({"Group":"Control", "Network":net_name, "Energy":val})

    results_df = pd.DataFrame(results)
    print("\n=== Outlier-Excluded Network-level comparison (ASD vs Control) ===")
    print(results_df)

    if len(all_data) > 0:
        df_all = pd.DataFrame(all_data)
        plt.figure(figsize=(10,6))
        sns.boxplot(x="Network", y="Energy", hue="Group", data=df_all, showfliers=False)
        sns.stripplot(x="Network", y="Energy", hue="Group", data=df_all,
                      dodge=True, alpha=0.6, color="black", size=3)
        plt.title("Network-level Energy (Outliers Excluded): ASD vs Control")
        plt.xlabel("Yeo7 Network")
        plt.ylabel("Energy")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
