import os
import numpy as np
import nibabel as nib
import networkx as nx

from nibabel.processing import resample_from_to
from scipy.stats import ttest_ind, ranksums
from NetworkEnergyComputer import compute_network_energy


###############################################################################
# 1) BUILD ROI->NETWORK MAP (CC200 -> YEO7)
###############################################################################
def build_roi_to_network_map(cc200_file, yeo7_file):
    """
    For each ROI in the Craddock 200 atlas (cc200_file),
    find which Yeo7 network (1..7) it overlaps the most.
    Returns a dict: roi_to_net[roi_id] = net_id (1..7).
    """

    cc200_img = nib.load(cc200_file)
    cc200_data = cc200_img.get_fdata()

    yeo_img = nib.load(yeo7_file)
    yeo_data = yeo_img.get_fdata()

    # Resample Yeo7 if needed
    if cc200_data.shape != yeo_data.shape:
        yeo_img = resample_from_to(yeo_img, cc200_img, order=0)
        yeo_data = yeo_img.get_fdata()

    roi_ids = np.unique(cc200_data[cc200_data > 0])
    yeo_ids = np.unique(yeo_data[yeo_data > 0])

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
def compute_subnetwork_energy(graph, roi_to_net, net_id):
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
# 5) EXTRACT SUBNETWORK ENERGY FEATURES
###############################################################################
def subnetwork_energy_features(path, label):
    cc200_file = r'directory-to-cc200-atlas'
    yeo7_file = r'directory-to-yeo7-atlas'

    roi_to_net = build_roi_to_network_map(cc200_file, yeo7_file)

    # Yeo 7 networks
    network_ids = [1, 2, 3, 4, 5, 6, 7]
    network_labels = {
        1: "Visual",
        2: "SomatoMotor",
        3: "DorsalAttn",
        4: "Salience/VentAttn",
        5: "Limbic",
        6: "Frontoparietal",
        7: "Default"
    }

    G = nx.read_graphml(path)
    whole_network_energy = compute_network_energy(G)
    energies = [whole_network_energy]
    for net_id in network_ids:
        energies.append(compute_subnetwork_energy(G, roi_to_net, net_id))

    return (energies, label)


###############################################################################
# 6) EXTRACT SUBNETWORK INTERCONNECTIVITY FEATURES
###############################################################################
def subnetwork_interconnectivity_features(path, label):
    cc200_file = r'D:\University\projects\CBTProject\CoevolutionaryBalanceTheory\atlas\CC200.nii'
    yeo7_file = r'D:\University\projects\CBTProject\CoevolutionaryBalanceTheory\atlas\Yeo7_1mm_reoriented.nii.gz'

    roi_to_net = build_roi_to_network_map(cc200_file, yeo7_file)

    network_ids = [1, 2, 3, 4, 5, 6, 7]
    network_labels = {
        1: "Visual",
        2: "SomatoMotor",
        3: "DorsalAttn",
        4: "Salience/VentAttn",
        5: "Limbic",
        6: "Frontoparietal",
        7: "Default"
    }

    G = nx.read_graphml(path)

    nnet = len(network_ids)
    SumW = np.zeros((nnet, nnet));
    CntW = np.zeros((nnet, nnet), int)
    SumF = np.zeros((nnet, nnet));
    CntF = np.zeros((nnet, nnet), int)
    E = np.zeros((nnet, nnet))

    for u, v, data in G.edges(data=True):
        ui, vi = int(u), int(v)
        nu, nv = roi_to_net[ui], roi_to_net[vi]
        w = data.get('weight', np.nan)
        fu = G.nodes[u]['fALFF']
        fv = G.nodes[v]['fALFF']
        if np.isnan(w) or np.isnan(fu) or np.isnan(fv):
            continue

        i = network_ids.index(nu)
        j = network_ids.index(nv)
        SumW[i, j] += w;
        CntW[i, j] += 1
        SumF[i, j] += w * fu * fv;
        CntF[i, j] += 1
        E[i, j] += -(w * fu * fv)
        if i != j:
            SumW[j, i] += w;
            CntW[j, i] += 1
            SumF[j, i] += w * fu * fv;
            CntF[j, i] += 1
            E[j, i] += -(w * fu * fv)

    AvgW = np.divide(SumW, CntW, out=np.full_like(SumW, np.nan), where=CntW > 0)
    AvgF = np.divide(SumF, CntF, out=np.full_like(SumF, np.nan), where=CntF > 0)

    # Record
    inter_energies_w_f = {}
    for i, ni in enumerate(network_ids):
        for j, nj in enumerate(network_ids):
            inter_energies_w_f.update({'Energy_{}_to_{}'.format(ni, nj) :  E[i, j],
            'AvgWeight_{}_to_{}'.format(ni, nj) :  AvgW[i, j],
            'AvgFalffW_{}_to_{}'.format(ni, nj) :  AvgF[i, j]})

    return (inter_energies_w_f, label)

