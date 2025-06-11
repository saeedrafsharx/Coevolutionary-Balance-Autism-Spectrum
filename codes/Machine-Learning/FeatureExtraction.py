#!/usr/bin/env python3
"""
feature_extractor.py

Comprehensive feature extraction for ASD/TD graphs:
1) Global coevolutionary-balance metrics: H, E, q, m, Tc
2) Subnetwork coevolutionary metrics (H, E, q, m, Tc for each Yeo7)
3) Inter-network connectivity: Energy, AvgWeight, AvgFalffW for each pair
4) Edge-type proportions: agreement, disagreement, imbalanced_same, imbalanced_opp
5) Bipolarity (balance in bipartition via Fiedler vector)

Outputs a CSV of all features per graph.
"""
import os
import numpy as np
import networkx as nx
import nibabel as nib
from nibabel.processing import resample_from_to
import pandas as pd

# ── USER CONFIG ───────────────────────────────────────────────────────────────
ASD_DIR    = "ASD"
TD_DIR     = "TD"
CC200_FILE = "CC200.nii"
YEO7_FILE  = "Yeo7_1mm_reoriented.nii"
# ──────────────────────────────────────────────────────────────────────────────

def build_roi_to_network_map(cc200_file, yeo7_file):
    cc200 = nib.load(cc200_file)
    cc200_data = cc200.get_fdata()
    yeo = nib.load(yeo7_file)
    yeo_data = yeo.get_fdata()
    if cc200_data.shape != yeo_data.shape:
        yeo = resample_from_to(yeo, cc200, order=0)
        yeo_data = yeo.get_fdata()
    roi_to_net = {}
    for roi in np.unique(cc200_data[cc200_data > 0]):
        mask = (cc200_data == roi)
        counts = [(net, np.sum(yeo_data[mask] == net))
                  for net in np.unique(yeo_data[yeo_data > 0])]
        best_net = max(counts, key=lambda x: x[1])[0]
        roi_to_net[int(roi)] = int(best_net)
    return roi_to_net


def compute_global_metrics(fALFF, weights, idx_i, idx_j):
    """
    Compute global coevolution metrics:
      H (Hamiltonian), E (energy per triplet), q (node-link correlation),
      m (mean fALFF), Tc (critical temperature = sqrt(n-1)).
    """
    n = len(fALFF)
    H = -np.sum(fALFF[idx_i] * weights * fALFF[idx_j])
    E = H / weights.size if weights.size > 0 else np.nan
    q = np.mean(weights * fALFF[idx_j]) if weights.size > 0 else np.nan
    m = fALFF.mean() if n > 0 else np.nan
    Tc = np.sqrt(n - 1) if n > 1 else 0.0
    return {'H': H, 'E': E, 'q': q, 'm': m, 'Tc': Tc}


def compute_subnetwork_coev_metrics(G, roi_to_net, fALFF):
    """
    Compute coevolution metrics per Yeo7 subnetwork (H, E, q, m, Tc).
    """
    feats = {}
    # group node indices by network
    nets = sorted(set(roi_to_net.values()))
    nodes_by_net = {net: [i for i in G.nodes() if roi_to_net.get(int(i),-1)==net]
                    for net in nets}
    for net in nets:
        nodes = nodes_by_net[net]
        if not nodes:
            feats.update({f'H_sub{net}': np.nan, f'E_sub{net}': np.nan,
                          f'q_sub{net}': np.nan, f'm_sub{net}': np.nan,
                          f'Tc_sub{net}': np.nan})
            continue
        # collect edges within subnetwork
        edge_list = [(u,v) for u,v in G.edges() if u in nodes and v in nodes]
        idx_i, idx_j = zip(*edge_list) if edge_list else ([],[])
        idx_i = np.array([int(u) for u in idx_i],dtype=int)
        idx_j = np.array([int(v) for v in idx_j],dtype=int)
        weights = np.array([G.edges[u,v]['weight'] for u,v in edge_list]) if edge_list else np.array([])
        # fALFF_sub
        f_sub = np.array([fALFF[int(i)] for i in nodes])
        # coevolution metrics
        H = -np.sum(fALFF[idx_i] * weights * fALFF[idx_j]) if weights.size>0 else np.nan
        E = H / weights.size if weights.size>0 else np.nan
        q = np.mean(weights * fALFF[idx_j]) if weights.size>0 else np.nan
        m = f_sub.mean() if f_sub.size>0 else np.nan
        Tc = np.sqrt(len(nodes)-1) if len(nodes)>1 else 0.0
        feats.update({f'H_sub{net}':H, f'E_sub{net}':E,
                      f'q_sub{net}':q, f'm_sub{net}':m,
                      f'Tc_sub{net}':Tc})
    return feats


def compute_inter_network_metrics(G, roi_to_net, fALFF):
    feats = {}
    nets = sorted(set(roi_to_net.values()))
    nnet = len(nets)
    SumW = np.zeros((nnet, nnet)); CntW = np.zeros((nnet, nnet))
    SumF = np.zeros((nnet, nnet)); CntF = np.zeros((nnet, nnet))
    Emat = np.zeros((nnet, nnet))
    for u, v, data in G.edges(data=True):
        ui, vi = roi_to_net.get(int(u),-1), roi_to_net.get(int(v),-1)
        if ui<1 or vi<1: continue
        i,j = nets.index(ui), nets.index(vi)
        fu,fv = fALFF[int(u)], fALFF[int(v)]
        w = data.get('weight',0.0)
        if np.isnan(fu) or np.isnan(fv) or np.isnan(w): continue
        SumW[i,j]+=w; CntW[i,j]+=1
        SumF[i,j]+=w*fu*fv; CntF[i,j]+=1
        Emat[i,j]+=-w*fu*fv
        if i!=j:
            SumW[j,i]+=w; CntW[j,i]+=1
            SumF[j,i]+=w*fu*fv; CntF[j,i]+=1
            Emat[j,i]+=-w*fu*fv
    AvgW = np.divide(SumW, CntW, where=CntW>0, out=np.full_like(SumW,np.nan))
    AvgF = np.divide(SumF, CntF, where=CntF>0, out=np.full_like(SumF,np.nan))
    for ii,ni in enumerate(nets):
        for jj,nj in enumerate(nets):
            feats[f'Energy_{ni}_to_{nj}']    = Emat[ii,jj]
            feats[f'AvgWeight_{ni}_to_{nj}'] = AvgW[ii,jj]
            feats[f'AvgFalffW_{ni}_to_{nj}'] = AvgF[ii,jj]
    return feats


def compute_pairwise_types(G):
    vals=[d.get('fALFF') for _,d in G.nodes(data=True) if d.get('fALFF') is not None]
    thresh=np.median(vals) if vals else 0.0
    sign={n:(1 if data.get('fALFF',0.0)>=thresh else -1) for n,data in G.nodes(data=True)}
    counts=dict.fromkeys(['agreement','disagreement','imbalanced_same','imbalanced_opp'],0)
    for u,v,data in G.edges(data=True):
        w=data.get('weight',0.0)
        if np.isnan(w) or u not in sign or v not in sign: continue
        sigma=1 if w>0 else -1
        if sign[u]==sign[v] and sigma==1: counts['agreement']+=1
        elif sign[u]!=sign[v] and sigma==-1: counts['disagreement']+=1
        elif sign[u]==sign[v] and sigma==-1: counts['imbalanced_same']+=1
        else: counts['imbalanced_opp']+=1
    total=sum(counts.values()) or 1
    return {f'PROP_{k.upper()}':counts[k]/total for k in counts}


def compute_bipolarity(G):
    A=nx.to_numpy_array(G,weight='weight')
    S=np.sign(A)
    D=np.diag(np.abs(S).sum(axis=1))
    L=D-S
    _,vecs=np.linalg.eigh(L)
    fiedler=vecs[:,1]
    assign={n:(1 if fiedler[i]>=0 else -1) for i,n in enumerate(G.nodes())}
    happy=total=0
    for u,v,data in G.edges(data=True):
        w=data.get('weight',0.0)
        if np.isnan(w): continue
        sigma=1 if w>0 else -1
        happy+=(sigma*assign[u]*assign[v]>0)
        total+=1
    return happy/total if total else 0.0


def extract_graph_features(path, roi_to_net=None):
    G=nx.read_graphml(path)
    mapping={n:i for i,n in enumerate(sorted(G.nodes()))}
    G=nx.relabel_nodes(G,mapping)
    n=G.number_of_nodes()
    fALFF=np.array([float(G.nodes[i].get('fALFF',0.0)) for i in range(n)])
    for u,v,data in G.edges(data=True): data['weight']=float(data.get('weight',0.0))
    i,j=np.triu_indices(n,1)
    W=np.array([G.edges[u,v]['weight'] for u,v in zip(i,j)])
    feats={}
    feats.update(compute_global_metrics(fALFF,W,i,j))
    if roi_to_net is None: roi_to_net=build_roi_to_network_map(CC200_FILE,YEO7_FILE)
    feats.update(compute_subnetwork_coev_metrics(G,roi_to_net,fALFF))
    feats.update(compute_inter_network_metrics(G,roi_to_net,fALFF))
    feats.update(compute_pairwise_types(G))
    feats['bipolarity']=compute_bipolarity(G)
    return feats


def main():
    all_feats=[]
    roi_to_net=build_roi_to_network_map(CC200_FILE,YEO7_FILE)
    for label,dirpath in [('ASD',ASD_DIR),('TD',TD_DIR)]:
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith('.graphml'): continue
            path=os.path.join(dirpath,fname)
            f=extract_graph_features(path,roi_to_net)
            f['label']=label; f['subject']=fname
            all_feats.append(f)
    df=pd.DataFrame(all_feats)
    print(df.head())
    df.to_csv('graph_features_Aged.csv',index=False)

if __name__=='__main__':
    main()
