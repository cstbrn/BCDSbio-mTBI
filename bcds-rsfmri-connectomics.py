#!/usr/bin/env python3
"""
===============================================================================
PIPELINE 3: RESTING-STATE fMRI (rs-fMRI) DUAL-ATLAS PATHOCONNECTOMICS
===============================================================================
Project: BCDSbio - Computational Neuroengineering Framework for mTBI
Author: Costanza Brunati - Brain Mapping Lab, Department of Biomedical, Dental Sciences and Morphological and Functional Imaging, University of Messina, Messina, Italy

OPEN-SOURCE CODEBASES & SOURCES TO ADAPT / CITATION REFERENCES:
1. Nilearn (fMRI Signal Extraction, Nuisance Regression, Connectivity):
   - GitHub: https://github.com/nilearn/nilearn
   - Citation: Abraham et al. Machine learning for neuroimaging with scikit-learn and Nilearn. Front Neuroinform 2014.
2. Schaefer-400 Cortical Atlas & Tian Subcortical Atlas:
   - Schaefer Cortical Atlas GitHub: https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal
   - Citation: Schaefer et al. Local-Global Parcellation of the Human Cerebral Cortex... Cereb Cortex 2018.
   - Tian Subcortical Atlas GitHub: https://github.com/yetianmed/subcortex
   - Citation: Tian et al. Topographic organization of the human subcortex unlocked with functional MRI. Nat Neurosci 2020.
3. NetworkX (Graph Theory Topology - Eglob, Clustering, Modularity):
   - GitHub: https://github.com/networkx/networkx
   - Citation: Hagberg et al. Exploring network structure, dynamics, and function using NetworkX. SciPy 2008.
   - Graph Metrics: Latora & Marchiori (2001) [Eglob], Watts & Strogatz (1998) [Ci], Newman (2006) [Q].
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as stats
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets

# Try importing community for modularity Q (python-louvain / networkx algorithms)
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


# =============================================================================
# 1. GRAPH THEORY METRICS CALCULATION
# =============================================================================

def compute_graph_theoretical_metrics(corr_matrix, threshold_density=0.15):
    """
    Computes graph theoretical topological properties from a Fisher Z-transformed correlation matrix.
    corr_matrix: (N_rois, N_rois) symmetric correlation matrix
    threshold_density: proportional thresholding (retaining top % strongest positive connections)
    
    Returns: Global Efficiency (Eglob), Mean Clustering Coefficient (Ci), Modularity (Q).
    """
    # Remove negative weights and zero diagonal
    w_matrix = np.copy(corr_matrix)
    np.fill_diagonal(w_matrix, 0.0)
    w_matrix[w_matrix < 0] = 0.0
    
    # Proportional thresholding to construct weighted adjacency matrix
    n_nodes = w_matrix.shape[0]
    n_edges_total = n_nodes * (n_nodes - 1) // 2
    n_edges_keep = int(np.round(n_edges_total * threshold_density))
    
    triu_indices = np.triu_indices(n_nodes, k=1)
    weights = w_matrix[triu_indices]
    cutoff_val = np.sort(weights)[::-1][n_edges_keep]
    
    adj_matrix = np.zeros_like(w_matrix)
    adj_matrix[w_matrix >= cutoff_val] = w_matrix[w_matrix >= cutoff_val]
    
    # Build NetworkX Graph
    G = nx.from_numpy_array(adj_matrix)
    
    # 1. Global Efficiency (Eglob) - Latora & Marchiori (2001)
    # Convert weights to distances: dist = 1 / weight
    G_dist = G.copy()
    for u, v, d in G_dist.edges(data=True):
        d['weight'] = 1.0 / (d['weight'] + 1e-9) if d['weight'] > 0 else 1e9
        
    shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G_dist))
    
    e_glob_sum = 0.0
    count = 0
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            dist = shortest_paths[u].get(v, np.inf)
            if dist > 0 and not np.isinf(dist):
                e_glob_sum += 1.0 / dist
            count += 1
            
    eglob = float(e_glob_sum / count) if count > 0 else 0.0
    
    # 2. Local Clustering Coefficient (Ci) - Watts & Strogatz (1998)
    clustering_dict = nx.clustering(G, weight='weight')
    ci_mean = float(np.mean(list(clustering_dict.values())))
    
    # 3. Modularity Index (Q) - Newman (2006) / Louvain
    if HAS_LOUVAIN:
        partition = community_louvain.best_partition(G, weight='weight', random_state=42)
        q_modularity = float(community_louvain.modularity(partition, G, weight='weight'))
    else:
        # Fallback to NetworkX greedy modularity
        communities = nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
        q_modularity = float(nx.algorithms.community.modularity(G, communities, weight='weight'))
        
    return eglob, ci_mean, q_modularity


# =============================================================================
# 2. MAIN rs-fMRI PROCESSING PIPELINE
# =============================================================================

def process_rsfmri_subject(bold_file_path, confounds_file_path, atlas_nii_path, output_dir, subject_id):
    """
    Executes rs-fMRI pathoconnectomics processing:
    1. QC Filter: Framewise Displacement (Mean FD <= 0.35mm), BOLD tSNR >= 40
    2. Nuisance Regression & Bandpass filtering (0.01 - 0.1 Hz)
    3. Time-series extraction using Schaefer-400 Cortical + Tian Subcortical integrated atlas
    4. Fisher Z-transformed functional connectivity matrix calculation
    5. Graph theoretical topology extraction (Eglob, Ci, Q)
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Starting rs-fMRI Processing for Subject: {subject_id}")
    
    # Step 1: Quality Control Checks (Mean FD & tSNR)
    confounds_df = pd.read_csv(confounds_file_path, sep='\t')
    if 'framewise_displacement' in confounds_df.columns:
        mean_fd = confounds_df['framewise_displacement'].mean(skipna=True)
    else:
        mean_fd = 0.15 # Default placeholder if pre-cleaned
        
    qc_passed_fd = mean_fd <= 0.35
    print(f"    - Mean Framewise Displacement (FD): {mean_fd:.3f} mm | QC Passed: {qc_passed_fd}")
    
    # Step 2: Extract BOLD Time-Series with Dual-Atlas Masker
    print(f"    - Extracting BOLD signal with Integrated Schaefer-400 + Tian Subcortical Atlas...")
    
    # NiftiLabelsMasker handles resampling, nuisance regression, and bandpass filtering
    masker = NiftiLabelsMasker(
        labels_img=atlas_nii_path,
        standardize=True,
        high_pass=0.01,
        low_pass=0.10,
        t_r=2.0, # Adjust to acquisition TR
        memory='nilearn_cache',
        memory_level=1,
        verbose=0
    )
    
    # Extract ROI time series shape: (n_timepoints, n_rois)
    time_series = masker.fit_transform(bold_file_path, confounds=confounds_df[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']])
    n_timepoints, n_rois = time_series.shape
    print(f"    - Extracted {n_rois} ROI time-series across {n_timepoints} BOLD frames.")
    
    # Compute tSNR (temporal SNR)
    signal_mean = np.mean(time_series, axis=0)
    signal_std = np.std(time_series, axis=0)
    tsnr_per_roi = np.where(signal_std > 0, signal_mean / signal_std, 0)
    mean_tsnr = float(np.mean(tsnr_per_roi))
    qc_passed_tsnr = mean_tsnr >= 40.0
    print(f"    - Mean BOLD Temporal SNR (tSNR): {mean_tsnr:.1f} | QC Passed: {qc_passed_tsnr}")
    
    # Step 3: Compute Pearson Correlation Matrix & Fisher Z-Transformation
    correlation_measure = ConnectivityMeasure(kind='correlation')
    corr_matrix = correlation_measure.fit_transform([time_series])[0]
    
    # Apply Fisher Z-transformation: Z = 0.5 * ln((1+r)/(1-r))
    # Clip r to [-0.9999, 0.9999] for numerical stability
    corr_clipped = np.clip(corr_matrix, -0.9999, 0.9999)
    fisher_z_matrix = 0.5 * np.log((1.0 + corr_clipped) / (1.0 - corr_clipped))
    np.fill_diagonal(fisher_z_matrix, 0.0)
    
    # Save functional connectome matrix
    conn_csv = os.path.join(output_dir, f"{subject_id}_functional_connectome_fisherz.csv")
    pd.DataFrame(fisher_z_matrix).to_csv(conn_csv, index=False)
    
    # Step 4: Extract Graph Theoretical Topology
    print("    - Computing Graph Theory Topological Metrics (Eglob, Ci, Q)...")
    eglob, ci_mean, q_modularity = compute_graph_theoretical_metrics(fisher_z_matrix, threshold_density=0.15)
    
    # Step 5: Save Subject Features
    feature_vector = {
        "subject_id": subject_id,
        "fmri_qc_passed": bool(qc_passed_fd and qc_passed_tsnr),
        "fmri_mean_fd_mm": float(mean_fd),
        "fmri_mean_tsnr": float(mean_tsnr),
        "fmri_Eglob": float(eglob),
        "fmri_Ci": float(ci_mean),
        "fmri_Q_modularity": float(q_modularity)
    }
    
    out_csv = os.path.join(output_dir, f"{subject_id}_rsfmri_features.csv")
    pd.DataFrame([feature_vector]).to_csv(out_csv, index=False)
    print(f"[✓] Saved rs-fMRI features to: {out_csv}\n")
    return feature_vector


if __name__ == "__main__":
    print("rs-fMRI Pipeline Module Loaded Successfully.")
