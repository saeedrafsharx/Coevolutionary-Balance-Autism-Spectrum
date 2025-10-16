import os
import glob
import nibabel as nib
import numpy as np
import networkx as nx
from nibabel.processing import resample_from_to

def extract_falff_per_roi(falff_file, atlas_file):
    """
    Load the subject's fALFF 3D image and the atlas,
    then compute the mean fALFF value for each ROI defined in the atlas.
    If shapes differ, the atlas is resampled using nearest-neighbor interpolation.
    """
    falff_img = nib.load(falff_file)
    falff_data = falff_img.get_fdata()
    print("Loaded fALFF data shape:", falff_data.shape)
    
    atlas_img = nib.load(atlas_file)
    atlas_data = atlas_img.get_fdata()
    print("Loaded atlas data shape:", atlas_data.shape)
    
    if falff_data.shape != atlas_data.shape:
        print("Shapes do not match. Resampling atlas to fALFF dimensions using nearest-neighbor interpolation...")
        atlas_img = resample_from_to(atlas_img, falff_img, order=0)
        atlas_data = atlas_img.get_fdata()
        print("Resampled atlas data shape:", atlas_data.shape)
    
    rois = np.unique(atlas_data)
    rois = rois[rois != 0]
    print("Unique ROI values in atlas:", rois)
    
    roi_falff = {}
    for roi in rois:
        mask = (atlas_data == roi)
        roi_voxels = falff_data[mask]
        if roi_voxels.size > 0:
            mean_val = np.mean(roi_voxels)
            roi_falff[int(roi)] = mean_val
            print(f"ROI {int(roi)}: mean fALFF = {mean_val}")
        else:
            roi_falff[int(roi)] = None
            print(f"ROI {int(roi)}: no voxels found")
    return roi_falff

def compute_functional_connectivity(timeseries_file):
    """
    Read the time series from a .1D (or similar) file:
      - First line: ROI labels (e.g., "#2001 #2002 #2101 ...")
      - Next lines: numerical data, one row per time point, columns = ROIs.

    Then TRUNCATE to 116 time points if the file has more than 116 rows.
    Return (roi_labels, fc_matrix).
    """
    with open(timeseries_file, 'r') as f:
        header_line = f.readline().strip()
    header_tokens = header_line.split()
    roi_labels = [token.lstrip('#') for token in header_tokens]
    print("Time series ROI labels:", roi_labels)
    
    data = np.loadtxt(timeseries_file, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    original_len = data.shape[0]
    print("Original time series data shape:", data.shape)
    
    TRUNC_LEN = 116
    if original_len > TRUNC_LEN:
        data = data[:TRUNC_LEN, :]
        print(f"Truncating from {original_len} to {TRUNC_LEN} time points.")
    else:
        print(f"No truncation needed (time points = {original_len}).")
    
    print("Final time series shape (after truncation):", data.shape)
    
    fc_matrix = np.corrcoef(data, rowvar=False)
    print("Functional connectivity (correlation) matrix shape:", fc_matrix.shape)
    return roi_labels, fc_matrix

def create_network(roi_labels, fc_matrix, roi_falff):
    """
    Create a NetworkX graph where each node is an ROI with its fALFF value
    and each edge weight is the correlation between the two ROIs.
    """
    G = nx.Graph()
    
    # Add nodes
    for label in roi_labels:
        roi_int = int(label)
        falff_val = roi_falff.get(roi_int, None)
        G.add_node(roi_int, fALFF=falff_val)
        print(f"Added node {roi_int} with fALFF = {falff_val}")
    
    # Add edges
    n_rois = len(roi_labels)
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            weight = fc_matrix[i, j]
            roi_i = int(roi_labels[i])
            roi_j = int(roi_labels[j])
            G.add_edge(roi_i, roi_j, weight=weight)
            #print: print(f"Edge ROI {roi_i}-{roi_j} weight={weight:.4f}") TODO: DEBUG
    return G

def process_subject(falff_file, timeseries_file, atlas_file):

    print(f"Processing subject with fALFF file: {falff_file}")
    
    roi_falff = extract_falff_per_roi(falff_file, atlas_file)
    
    print("\nComputing Functional Connectivity for:", timeseries_file)
    roi_labels, fc_matrix = compute_functional_connectivity(timeseries_file)
    
    print("\nCreating Network...")
    network = create_network(roi_labels, fc_matrix, roi_falff)
    print("Network created for subject.")
    return network

def main():
    falff_dir = r"directory-to-alff-files"
    timeseries_dir = r"directory-to-timeseries-files"
    atlas_file = r"atlasfile.nii.gz"
    output_dir = r"output-directory"
    os.makedirs(output_dir, exist_ok=True)
    
    falff_files = glob.glob(os.path.join(falff_dir, "*.nii.gz"))
    falff_dict = {}
    for ff in falff_files:
        base = os.path.basename(ff)
        subject_id = base.split('_falff')[0]
        falff_dict[subject_id] = ff
    
    ts_files = glob.glob(os.path.join(timeseries_dir, "*.1D"))
    ts_dict = {}
    for tf in ts_files:
        base = os.path.basename(tf)
        subject_id = base.split('_rois_cc200')[0]
        ts_dict[subject_id] = tf
    
    common_subjects = sorted(set(falff_dict.keys()) & set(ts_dict.keys()))
    print(f"Found {len(common_subjects)} matching subjects in falff & timeseries dirs.")
    if not common_subjects:
        print("No matching subjects found. Check your filenames.")
        return
    
    for subject_id in common_subjects:
        falff_file = falff_dict[subject_id]
        timeseries_file = ts_dict[subject_id]
        
        print(f"\n=== Processing subject: {subject_id} ===")
        network = process_subject(falff_file, timeseries_file, atlas_file)
        
        network_filename = os.path.join(output_dir, f"{subject_id}_network.graphml")
        nx.write_graphml(network, network_filename)
        print(f"Network saved for subject {subject_id} at {network_filename}")
