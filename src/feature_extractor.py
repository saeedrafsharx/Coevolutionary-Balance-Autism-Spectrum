"""
Feature Extraction for Coevolutionary Balance Analysis

Extracts whole-brain, intra-network, and inter-network coevolutionary 
features from brain network graphs (GraphML format).

Usage:
    python feature_extractor.py --asd_dir ./ASD --ctrl_dir ./Control \
                                --cc200 CC200.nii --yeo7 Yeo7.nii \
                                --output features.csv
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import nibabel as nib
from nibabel.processing import resample_from_to


# Yeo 7-Network Labels
NETWORKS = {
    1: "Visual",
    2: "SomatoMotor", 
    3: "DorsalAttn",
    4: "Salience/VentAttn",
    5: "Limbic",
    6: "Frontoparietal",
    7: "Default"
}


def identify_outliers_iqr(values, factor=1.5):
    """Identify outliers using Tukey's IQR rule."""
    values = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return (values < q1 - factor * iqr) | (values > q3 + factor * iqr)


def build_roi_network_map(cc200_path, yeo7_path):
    """Map CC200 ROIs to Yeo 7 networks using Dice overlap."""
    cc200 = nib.load(cc200_path)
    cc_data = cc200.get_fdata().astype(int)
    
    yeo = nib.load(yeo7_path)
    yeo_data = yeo.get_fdata().astype(int)
    
    # Resample if needed
    if cc_data.shape != yeo_data.shape:
        yeo = resample_from_to(yeo, cc200, order=0)
        yeo_data = yeo.get_fdata().astype(int)
    
    roi_ids = np.unique(cc_data[cc_data > 0])
    net_ids = np.unique(yeo_data[yeo_data > 0])
    net_sizes = {n: np.sum(yeo_data == n) for n in net_ids}
    
    mapping = {}
    for roi in roi_ids:
        mask = cc_data == roi
        size = mask.sum()
        best_net, best_dice = None, -1
        
        for net in net_ids:
            inter = np.sum(mask & (yeo_data == net))
            dice = 2 * inter / (size + net_sizes[net]) if (size + net_sizes[net]) > 0 else 0
            if dice > best_dice:
                best_net, best_dice = int(net), dice
        
        mapping[int(roi)] = best_net
    
    return mapping


def binarize_falff(graph):
    """Binarize fALFF: +1 if >= median, -1 otherwise."""
    vals = [d.get('fALFF') for _, d in graph.nodes(data=True) 
            if d.get('fALFF') is not None and not np.isnan(d.get('fALFF'))]
    
    if not vals:
        return {}, np.nan
    
    threshold = np.median(vals)
    signs = {}
    for node, data in graph.nodes(data=True):
        f = data.get('fALFF')
        if f is not None and not np.isnan(f):
            signs[node] = 1 if f >= threshold else -1
    
    return signs, threshold


def compute_hamiltonian(graph, signs):
    """Compute coevolutionary energy: H = -Σ s_i * w_ij * s_j"""
    energy = 0.0
    for u, v, data in graph.edges(data=True):
        if u in signs and v in signs:
            w = data.get('weight', 0)
            if w is not None and not np.isnan(w):
                energy += signs[u] * w * signs[v]
    return -energy


def compute_edge_types(graph, signs):
    """Classify edges into agreement/disagreement/imbalanced types."""
    counts = {'agreement': 0, 'disagreement': 0, 
              'imbalanced_same': 0, 'imbalanced_opp': 0}
    
    for u, v, data in graph.edges(data=True):
        if u not in signs or v not in signs:
            continue
        w = data.get('weight', 0)
        if w is None or np.isnan(w):
            continue
        
        link_sign = 1 if w > 0 else -1
        same_state = signs[u] == signs[v]
        
        if same_state and link_sign == 1:
            counts['agreement'] += 1
        elif not same_state and link_sign == -1:
            counts['disagreement'] += 1
        elif same_state and link_sign == -1:
            counts['imbalanced_same'] += 1
        else:
            counts['imbalanced_opp'] += 1
    
    total = sum(counts.values()) or 1
    props = {k: v / total for k, v in counts.items()}
    return counts, props


def compute_bipolarity(graph):
    """Compute structural balance via signed Laplacian Fiedler partition."""
    A = nx.to_numpy_array(graph, weight='weight')
    if A.size == 0:
        return np.nan
    
    S = np.sign(A)
    D = np.diag(np.abs(S).sum(axis=1))
    L = D - S
    
    vals, vecs = np.linalg.eigh(L)
    if vecs.shape[1] < 2:
        return np.nan
    
    fiedler = vecs[:, 1]
    partition = {n: (1 if fiedler[i] >= 0 else -1) for i, n in enumerate(graph.nodes())}
    
    balanced = 0
    total = 0
    for u, v, data in graph.edges(data=True):
        w = data.get('weight', 0)
        if w is None or np.isnan(w):
            continue
        sign = 1 if w > 0 else -1
        balanced += int(sign * partition[u] * partition[v] > 0)
        total += 1
    
    return balanced / total if total else np.nan


def extract_subject_features(graph, roi_to_net):
    """Extract all features for a single subject."""
    signs, _ = binarize_falff(graph)
    if not signs:
        return None
    
    # Whole-brain metrics
    features = {
        'Whole_Energy_global': compute_hamiltonian(graph, signs),
        'Whole_Bipolarity': compute_bipolarity(graph)
    }
    
    _, props = compute_edge_types(graph, signs)
    for k, v in props.items():
        features[f'Whole_Prop_{k}'] = v
    
    # Network-level metrics
    n_nets = len(NETWORKS)
    net_ids = list(NETWORKS.keys())
    
    SumW = np.zeros((n_nets, n_nets))
    CntW = np.zeros((n_nets, n_nets), dtype=int)
    SumF = np.zeros((n_nets, n_nets))
    CntF = np.zeros((n_nets, n_nets), dtype=int)
    Eblk = np.zeros((n_nets, n_nets))
    
    for u, v, data in graph.edges(data=True):
        try:
            ui, vi = int(u), int(v)
        except ValueError:
            continue
        
        nu, nv = roi_to_net.get(ui), roi_to_net.get(vi)
        if nu is None or nv is None:
            continue
        
        w = data.get('weight')
        if w is None or np.isnan(w):
            continue
        
        if u not in signs or v not in signs:
            continue
        
        try:
            i, j = net_ids.index(nu), net_ids.index(nv)
        except ValueError:
            continue
        
        su, sv = signs[u], signs[v]
        
        SumW[i, j] += w
        CntW[i, j] += 1
        SumF[i, j] += w * su * sv
        CntF[i, j] += 1
        Eblk[i, j] -= w * su * sv
        
        if i != j:
            SumW[j, i] += w
            CntW[j, i] += 1
            SumF[j, i] += w * su * sv
            CntF[j, i] += 1
            Eblk[j, i] -= w * su * sv
    
    AvgW = np.divide(SumW, CntW, out=np.full_like(SumW, np.nan), where=CntW > 0)
    AvgF = np.divide(SumF, CntF, out=np.full_like(SumF, np.nan), where=CntF > 0)
    
    # Intra-network (diagonal)
    for i, net_id in enumerate(net_ids):
        name = NETWORKS[net_id]
        features[f'Intra_{name}_AvgWeight'] = AvgW[i, i]
        features[f'Intra_{name}_AvgFalffW'] = AvgF[i, i]
        features[f'Intra_{name}_EnergyBlock'] = Eblk[i, i]
    
    # Inter-network (upper triangle)
    for i, ni in enumerate(net_ids):
        for j in range(i + 1, len(net_ids)):
            nj = net_ids[j]
            pair = f'Inter_{NETWORKS[ni]}__{NETWORKS[nj]}'
            features[f'{pair}_AvgWeight'] = AvgW[i, j]
            features[f'{pair}_AvgFalffW'] = AvgF[i, j]
            features[f'{pair}_EnergyBlock'] = Eblk[i, j]
    
    return features


def main():
    parser = argparse.ArgumentParser(description='Extract coevolutionary features')
    parser.add_argument('--asd_dir', required=True, help='Directory with ASD GraphML files')
    parser.add_argument('--ctrl_dir', required=True, help='Directory with Control GraphML files')
    parser.add_argument('--cc200', required=True, help='CC200 atlas NIfTI file')
    parser.add_argument('--yeo7', required=True, help='Yeo 7-network atlas NIfTI file')
    parser.add_argument('--output', default='features.csv', help='Output CSV path')
    args = parser.parse_args()
    
    print("[INFO] Building ROI-to-network mapping...")
    roi_to_net = build_roi_network_map(args.cc200, args.yeo7)
    
    # Pass 1: Get global energies for outlier detection
    print("[INFO] Pass 1: Computing global energies...")
    subjects = []
    energies = []
    
    for group, directory in [('ASD', args.asd_dir), ('Control', args.ctrl_dir)]:
        for path in glob.glob(os.path.join(directory, '*.graphml')):
            G = nx.read_graphml(path)
            sid = os.path.basename(path).split('_')[1]
            signs, _ = binarize_falff(G)
            
            if signs:
                energy = compute_hamiltonian(G, signs)
                subjects.append({'Subject': sid, 'Group': group, 'Path': path})
                energies.append(energy)
    
    # Remove outliers
    outliers = identify_outliers_iqr(energies)
    good_subjects = [s for s, is_out in zip(subjects, outliers) if not is_out]
    print(f"[INFO] Kept {len(good_subjects)}/{len(subjects)} subjects after outlier removal")
    
    # Pass 2: Extract all features
    print("[INFO] Pass 2: Extracting features...")
    rows = []
    
    for subj in good_subjects:
        G = nx.read_graphml(subj['Path'])
        feats = extract_subject_features(G, roi_to_net)
        
        if feats:
            feats['Subject'] = subj['Subject']
            feats['Group'] = subj['Group']
            rows.append(feats)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns
    meta = ['Subject', 'Group']
    other = [c for c in df.columns if c not in meta]
    df = df[meta + sorted(other)]
    
    df.to_csv(args.output, index=False)
    print(f"[INFO] Saved {len(df)} subjects to {args.output}")


if __name__ == '__main__':
    main()
