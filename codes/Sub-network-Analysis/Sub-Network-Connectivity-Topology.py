import os
import glob
import numpy as np
import pandas as pd
import networkx as nx
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.stats import (
    shapiro, levene, ttest_ind,
    ranksums, mannwhitneyu
)
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

def identify_outliers_iqr(values, iqr_factor=1.5):
    if len(values) < 2:
        return np.zeros(len(values), dtype=bool)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
    return (values < lower) | (values > upper)

def build_roi_to_network_map(cc200_file, yeo7_file):
    cc200 = nib.load(cc200_file)
    cc_data = cc200.get_fdata().astype(int)
    yeo = nib.load(yeo7_file)
    yeo_data = yeo.get_fdata().astype(int)
    if cc_data.shape != yeo_data.shape:
        yeo = resample_from_to(yeo, cc200, order=0)
        yeo_data = yeo.get_fdata().astype(int)

    roi_ids = np.unique(cc_data[cc_data>0])
    net_ids = np.unique(yeo_data[yeo_data>0])
    net_sizes = {n: np.sum(yeo_data==n) for n in net_ids}

    roi_to_net = {}
    for roi in roi_ids:
        mask = (cc_data == roi)
        size = mask.sum()
        best_net, best_dice = None, -1
        for net in net_ids:
            inter = np.sum(mask & (yeo_data==net))
            denom = size + net_sizes[net]
            if denom <= 0:
                continue
            dice = 2 * inter / denom
            if dice > best_dice:
                best_net, best_dice = int(net), dice
        roi_to_net[int(roi)] = best_net

    return roi_to_net

def main():
    cc200_file = "/kaggle/input/net-cbt/CC200.nii"
    yeo7_file  = "/kaggle/input/net-cbt/Yeo7_1mm_reoriented.nii"
    asd_dir    = "/kaggle/input/net-cbt/ASD"
    ctrl_dir   = "/kaggle/input/net-cbt/Control"

    roi_to_net    = build_roi_to_network_map(cc200_file, yeo7_file)
    network_ids   = [1,2,3,4,5,6,7]
    network_names = {
        1:"Visual",2:"SomatoMotor",3:"DorsalAttn",
        4:"Salience/VentAttn",5:"Limbic",
        6:"Frontoparietal",7:"Default"
    }

    conn_records, energy_records, assort_records = [], [], []

    for group, d in [("ASD", asd_dir), ("Control", ctrl_dir)]:
        for fn in glob.glob(os.path.join(d, "*.graphml")):
            G = nx.read_graphml(fn)
            sid = os.path.basename(fn)

            if G.number_of_edges() == 0:
                print(f"[SKIP {sid}] no edges")
                continue

            bad = False
            for n,data in G.nodes(data=True):
                falff = data.get('fALFF', None)
                if falff is None or np.isnan(float(falff)):
                    print(f"[SKIP {sid}] node {n} missing fALFF")
                    bad = True
                    break
                try:
                    roi = int(n)
                except:
                    print(f"[SKIP {sid}] node label {n} not integer")
                    bad = True
                    break
                if roi_to_net.get(roi) is None:
                    print(f"[SKIP {sid}] ROI {roi} not in mapping")
                    bad = True
                    break
            if bad:
                continue

            nnet = len(network_ids)
            SumW = np.zeros((nnet,nnet)); CntW = np.zeros((nnet,nnet),int)
            SumF = np.zeros((nnet,nnet)); CntF = np.zeros((nnet,nnet),int)
            E    = np.zeros((nnet,nnet))

            for u,v,data in G.edges(data=True):
                ui,vi = int(u), int(v)
                nu, nv = roi_to_net[ui], roi_to_net[vi]
                w  = data.get('weight', np.nan)
                fu = G.nodes[u]['fALFF']
                fv = G.nodes[v]['fALFF']
                if np.isnan(w) or np.isnan(fu) or np.isnan(fv):
                    continue

                i = network_ids.index(nu)
                j = network_ids.index(nv)
                SumW[i,j] += w;   CntW[i,j] += 1
                SumF[i,j] += w*fu*fv; CntF[i,j] += 1
                E[i,j]    += -(w*fu*fv)
                if i!=j:
                    SumW[j,i] += w;   CntW[j,i] += 1
                    SumF[j,i] += w*fu*fv; CntF[j,i] += 1
                    E[j,i]    += -(w*fu*fv)

            AvgW = np.divide(SumW, CntW, out=np.full_like(SumW,np.nan), where=CntW>0)
            AvgF = np.divide(SumF, CntF, out=np.full_like(SumF,np.nan), where=CntF>0)

            for i,ni in enumerate(network_ids):
                for j,nj in enumerate(network_ids):
                    conn_records.append({
                        'Subject':sid,'Group':group,
                        'From':network_names[ni],'To':network_names[nj],
                        'AvgWeight':AvgW[i,j],'AvgFalffW':AvgF[i,j]
                    })
                    energy_records.append({
                        'Subject':sid,'Group':group,
                        'From':network_names[ni],'To':network_names[nj],
                        'Energy':E[i,j]
                    })

            labels = { n: roi_to_net[int(n)] for n in G.nodes() }
            nx.set_node_attributes(G, labels, 'net_label')
            assort = nx.attribute_assortativity_coefficient(G, 'net_label')
            assort_records.append({
                'Subject':sid,'Group':group,'Assortativity':assort
            })

    conn_df   = pd.DataFrame(conn_records)
    energy_df = pd.DataFrame(energy_records)
    assort_df = pd.DataFrame(assort_records)

    bad_subs = set()
    for df,col in [
        (conn_df,'AvgWeight'),
        (conn_df,'AvgFalffW'),
        (energy_df,'Energy')
    ]:
        for _,grp in df.groupby(['From','To']):
            mask = identify_outliers_iqr(grp[col].values)
            bad_subs.update(grp.loc[mask,'Subject'])
    conn_df   = conn_df[~conn_df.Subject.isin(bad_subs)]
    energy_df = energy_df[~energy_df.Subject.isin(bad_subs)]
    assort_df = assort_df[~assort_df.Subject.isin(bad_subs)]

    surv_counts = assort_df.groupby('Group')['Subject'].nunique()
    print(f"\nSubjects survived after filtering: "
          f"ASD={surv_counts.get('ASD',0)}, "
          f"Control={surv_counts.get('Control',0)}")

    stats_conn = []
    for metric in ['AvgWeight','AvgFalffW']:
        for (u,v),grp in conn_df.groupby(['From','To']):
            a = grp[grp.Group=='ASD'][metric].dropna()
            c = grp[grp.Group=='Control'][metric].dropna()
            if len(a)>=2 and len(c)>=2:
                t = ttest_ind(a,c,equal_var=False)
                r = ranksums(a,c)
                stats_conn.append({
                    'Metric':metric,'From':u,'To':v,
                    'T_p':t.pvalue,'R_p':r.pvalue
                })
    stats_conn_df = pd.DataFrame(stats_conn)
    stats_conn_df['T_p_fdr'] = multipletests(stats_conn_df['T_p'],method='fdr_bh')[1]
    stats_conn_df['R_p_fdr'] = multipletests(stats_conn_df['R_p'],method='fdr_bh')[1]
    sig_conn = stats_conn_df.query('T_p_fdr<0.05 or R_p_fdr<0.05')

    stats_energy = []
    for (u,v),grp in energy_df.groupby(['From','To']):
        a = grp[grp.Group=='ASD']['Energy'].dropna()
        c = grp[grp.Group=='Control']['Energy'].dropna()
        if len(a)>=2 and len(c)>=2:
            t = ttest_ind(a,c,equal_var=False)
            r = ranksums(a,c)
            stats_energy.append({
                'From':u,'To':v,
                'T_p':t.pvalue,'R_p':r.pvalue
            })
    stats_energy_df = pd.DataFrame(stats_energy)
    stats_energy_df['T_p_fdr'] = multipletests(stats_energy_df['T_p'],method='fdr_bh')[1]
    stats_energy_df['R_p_fdr'] = multipletests(stats_energy_df['R_p'],method='fdr_bh')[1]
    sig_energy = stats_energy_df.query('T_p_fdr<0.05 or R_p_fdr<0.05')


    a_asd = assort_df.loc[assort_df.Group=='ASD','Assortativity'].values
    a_ctl = assort_df.loc[assort_df.Group=='Control','Assortativity'].values
    print("ASD assortativity unique values:", np.unique(a_asd))
    print("Control assortativity unique values:", np.unique(a_ctl))
    alpha = 0.05
    if np.ptp(a_asd)==0 and np.ptp(a_ctl)==0:
        test_name, p_assort = "No variance (identical)", 1.0
    else:
        p_sh_asd = shapiro(a_asd).pvalue
        p_sh_ctl = shapiro(a_ctl).pvalue
        normal_asd = (p_sh_asd>alpha)
        normal_ctl = (p_sh_ctl>alpha)
        p_lev = levene(a_asd,a_ctl).pvalue
        if np.isnan(p_lev):
            var_asd = np.var(a_asd,ddof=1)
            var_ctl = np.var(a_ctl,ddof=1)
            equal_var = (var_asd==0 and var_ctl==0)
        else:
            equal_var = (p_lev>alpha)
        if normal_asd and normal_ctl:
            if equal_var:
                test_name = "Student’s t"
                _, p_assort = ttest_ind(a_asd,a_ctl,equal_var=True)
            else:
                test_name = "Welch’s t"
                _, p_assort = ttest_ind(a_asd,a_ctl,equal_var=False)
        else:
            test_name = "Mann–Whitney U"
            _, p_assort = mannwhitneyu(a_asd,a_ctl,alternative='two-sided')

    print("\nSignificant connectivity (FDR<0.05):")
    print(sig_conn if not sig_conn.empty else "None")
    print("\nSignificant energy (FDR<0.05):")
    print(sig_energy if not sig_energy.empty else "None")
    print("\n--- Assortativity comparison ---")
    print(f"Using {test_name}, p={p_assort:.3g} →",
          "Significant" if p_assort<alpha else "Not significant")