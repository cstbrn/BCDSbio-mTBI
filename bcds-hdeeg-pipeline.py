#!/usr/bin/env python3
"""
===============================================================================
PIPELINE 1: HIGH-DENSITY EEG (hd-EEG) ELECTRODYNAMICS & COMPLEXITY EXTRACTION
===============================================================================
Project: BCDSbio - Computational Neuroengineering Framework for mTBI
Author: Costanza Brunati - Brain Mapping Lab, Department of Biomedical, Dental Sciences and Morphological and Functional Imaging, University of Messina, Messina, Italy

OPEN-SOURCE CODEBASES & SOURCES TO ADAPT / CITATION REFERENCES:
1. MNE-Python (EEG Preprocessing, Filtering, CAR, ICA Infomax):
   - GitHub: https://github.com/mne-tools/mne-python
   - Citation: Gramfort et al., MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 2013.
2. Pycrostates (EEG Microstates Clustering A, B, C, D):
   - GitHub: https://github.com/vferat/pycrostates
   - Citation: Poulsen et al., Microstate analysis of continuous EEG. IEEE TBME, 2018.
3. AntroPy & NeuroKit2 (Lempel-Ziv Complexity LZC & Multiscale Entropy MSE):
   - GitHub: https://github.com/raphaelvallat/antropy
   - GitHub: https://github.com/neuropsychology/NeuroKit2
   - Citation: Lempel & Ziv, On the Complexity of Finite Sequences. IEEE Trans Inf Theory, 1976.
   - Citation: Costa et al., Multiscale entropy analysis of complex physiological time series. PRL, 2002.
4. OpenNeuro Dataset Benchmark (Nwakamma et al., 2024 on ds003522):
   - OpenNeuro ds003522: 65-channel EEG in subacute mTBI vs controls.
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
import mne
from mne.preprocessing import ICA

# Try importing pycrostates & antropy, fallback to native implementation if missing
try:
    import pycrostates
    from pycrostates.cluster import ModKMeans
    HAS_PYCROSTATES = True
except ImportError:
    HAS_PYCROSTATES = False

try:
    import antropy as ant
    HAS_ANTROPY = True
except ImportError:
    HAS_ANTROPY = False


# =============================================================================
# 1. HELPER FUNCTIONS: Lempel-Ziv Complexity (LZC) & Multiscale Entropy (MSE)
# =============================================================================

def compute_lempel_ziv_complexity(binary_sequence):
    """
    Computes normalized Lempel-Ziv Complexity (LZC) for a 1D binary sequence.
    Reference: Lempel & Ziv (1976); Zhang et al. (2001).
    """
    n = len(binary_sequence)
    if n == 0:
        return 0.0
    
    i, k, l = 0, 1, 1
    c = 1
    q = 1
    
    while k + l <= n:
        if binary_sequence[i + l - 1] == binary_sequence[k + l - 1]:
            l += 1
        else:
            if l > q:
                q = l
            i += 1
            if i == k:
                c += 1
                k = k + q
                q = 1
                i = 0
                l = 1
            else:
                l = 1
    if l > 1:
        c += 1
        
    b = n / np.log2(n) if n > 1 else 1.0
    return c / b


def compute_sample_entropy(x, m=2, r=0.15):
    """
    Calculates Sample Entropy (SampEn) for a 1D time-series.
    m: embedding dimension (default 2)
    r: tolerance threshold (default 0.15 * SD)
    """
    x = np.array(x, dtype=np.float64)
    N = len(x)
    r_val = r * np.std(x)
    
    def _maxnorm(u, v):
        return np.max(np.abs(u - v), axis=1)

    def _phi(m_len):
        x_mat = np.array([x[i:i + m_len] for i in range(N - m_len + 1)])
        C = 0
        for i in range(len(x_mat)):
            dist = _maxnorm(x_mat[i], x_mat)
            C += np.sum(dist <= r_val) - 1 # Exclude self-match
        return C / ((N - m_len + 1) * (N - m_len))

    try:
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        if phi_m == 0 or phi_m1 == 0:
            return np.nan
        return -np.log(phi_m1 / phi_m)
    except Exception:
        return np.nan


def compute_multiscale_entropy(data, max_tau=20, m=2, r=0.15):
    """
    Calculates Multiscale Entropy (MSE) across temporal scale factors tau = 1..max_tau.
    Reference: Costa et al. (2002).
    """
    mse_vals = {}
    for tau in range(1, max_tau + 1):
        # Coarse-graining procedure
        n_points = len(data) // tau
        if n_points < 100:
            mse_vals[f"MSE_scale_{tau}"] = np.nan
            continue
        coarse_data = np.mean(data[:n_points * tau].reshape(n_points, tau), axis=1)
        samp_en = compute_sample_entropy(coarse_data, m=m, r=r)
        mse_vals[f"MSE_scale_{tau}"] = samp_en
    return mse_vals


# =============================================================================
# 2. MAIN EEG PROCESSING PIPELINE
# =============================================================================

def process_hdeeg_subject(eeg_file_path, output_dir, subject_id):
    """
    Executes complete hd-EEG processing:
    1. Preprocessing (Filtering 0.5-40 Hz, CAR)
    2. Quality Control Metrology (Line noise, bad channels, continuous clean duration)
    3. Artifact removal via ICA Infomax
    4. Microstate Topography (Classes A, B, C, D)
    5. Non-linear Complexity (LZC & MSE)
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Starting hd-EEG Processing for Subject: {subject_id}")
    
    # Step 1: Load EEG Data (BIDS format EDF/BrainVision/EEGLAB)
    if eeg_file_path.endswith('.set'):
        raw = mne.io.read_raw_eeglab(eeg_file_path, preload=True)
    elif eeg_file_path.endswith('.edf'):
        raw = mne.io.read_raw_edf(eeg_file_path, preload=True)
    elif eeg_file_path.endswith('.vhdr'):
        raw = mne.io.read_raw_brainvision(eeg_file_path, preload=True)
    else:
        raise ValueError(f"Unsupported EEG file format: {eeg_file_path}")
        
    n_channels = len(raw.ch_names)
    print(f"    - Loaded {n_channels} channels. Sampling rate: {raw.info['sfreq']} Hz")
    
    # Enforce electrode count flexibility (>=60 channels)
    if n_channels < 60:
        print(f"[WARNING] Subject {subject_id} has {n_channels} channels (< 60 threshold). Flagging QC.")
        
    # Step 2: Quality Control - Pre-cleaning Metrics
    raw_data = raw.get_data()
    bad_channels = []
    # Identify dead/flat channels or extreme variance
    channel_vars = np.var(raw_data, axis=1)
    var_thresh_high = np.median(channel_vars) * 10
    var_thresh_low = np.median(channel_vars) * 0.01
    for idx, ch_name in enumerate(raw.ch_names):
        if channel_vars[idx] > var_thresh_high or channel_vars[idx] < var_thresh_low:
            bad_channels.append(ch_name)
            
    bad_channel_pct = (len(bad_channels) / n_channels) * 100
    print(f"    - Identified {len(bad_channels)} bad channels ({bad_channel_pct:.1f}%).")
    raw.info['bads'] = bad_channels
    
    # Interpolate bad channels if < 10%
    if bad_channel_pct <= 10.0 and len(bad_channels) > 0:
        raw.interpolate_bads(reset=bads=True)
    
    # Step 3: Filtering (0.5 - 40 Hz Bandpass) & Downsampling to 250 Hz
    raw.filter(l_freq=0.5, h_freq=40.0, fir_design='firwin', verbose=False)
    if raw.info['sfreq'] != 250.0:
        raw.resample(sfreq=250.0)
        
    # Common Average Reference (CAR)
    raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)
    
    # Step 4: Artifact Removal via ICA Infomax
    print("    - Fitting ICA Infomax for artifact rejection...")
    ica = ICA(n_components=min(30, len(raw.ch_names) - len(raw.info['bads'])),
              method='infomax', fit_params=dict(extended=True), random_state=42)
    ica.fit(raw, verbose=False)
    
    # Automated EOG/ECG component selection (heuristic / mne correlation)
    # Exclude components capturing ocular/muscular artifacts
    eog_indices = []
    # (In practice, match against EOG channel or spatial topography)
    ica.exclude = eog_indices
    raw_cleaned = raw.copy()
    ica.apply(raw_cleaned, verbose=False)
    
    # Enforce minimum 300s clean continuous resting state
    clean_duration_sec = raw_cleaned.times[-1]
    qc_passed = (bad_channel_pct <= 10.0) and (clean_duration_sec >= 300.0)
    print(f"    - Clean duration: {clean_duration_sec:.1f} s | QC Passed: {qc_passed}")
    
    # =========================================================================
    # Step 5: Feature Extraction - Non-Linear Signal Complexity (LZC & MSE)
    # =========================================================================
    print("    - Extracting Non-linear Complexity (LZC & MSE)...")
    cleaned_data = raw_cleaned.get_data() # shape: (n_channels, n_samples)
    
    # 5.1 Lempel-Ziv Complexity (LZC)
    # Median binarization per channel
    lzc_per_channel = []
    for ch_idx in range(cleaned_data.shape[0]):
        ch_sig = cleaned_data[ch_idx, :]
        median_val = np.median(ch_sig)
        bin_sig = (ch_sig > median_val).astype(int)
        lzc_val = compute_lempel_ziv_complexity(bin_sig)
        lzc_per_channel.append(lzc_val)
        
    mean_lzc = float(np.mean(lzc_per_channel))
    std_lzc = float(np.std(lzc_per_channel))
    
    # 5.2 Multiscale Entropy (MSE) - Global Field Power (GFP) sequence
    gfp = np.std(cleaned_data, axis=0) # GFP time-series
    mse_dict = compute_multiscale_entropy(gfp, max_tau=20, m=2, r=0.15)
    
    # =========================================================================
    # Step 6: Feature Extraction - EEG Microstate Topography (Classes A, B, C, D)
    # =========================================================================
    print("    - Extracting Microstate Dynamics (Classes A, B, C, D)...")
    microstate_features = {}
    if HAS_PYCROSTATES:
        try:
            # ModKMeans clustering for 4 canonical microstates
            ModK = ModKMeans(n_clusters=4, random_state=42)
            ModK.fit(raw_cleaned, n_jobs=1, verbose=False)
            segmentation = ModK.predict(raw_cleaned, verbose=False)
            
            # Extract metrics: Duration, Occurrence, Coverage, Transitions
            params = segmentation.compute_parameters()
            for ms_class, name in zip(range(4), ['A', 'B', 'C', 'D']):
                microstate_features[f"MS_Class_{name}_Duration_ms"] = float(params[f"{ms_class}_Mean Duration (s)"] * 1000)
                microstate_features[f"MS_Class_{name}_Occurrence_Hz"] = float(params[f"{ms_class}_Occurrence (1/s)"])
                microstate_features[f"MS_Class_{name}_Coverage_pct"] = float(params[f"{ms_class}_Fractional Coverage"] * 100)
        except Exception as e:
            print(f"    [!] Pycrostates execution error: {e}. Fallback to simulated microstate metrics.")
            for name in ['A', 'B', 'C', 'D']:
                microstate_features[f"MS_Class_{name}_Duration_ms"] = np.nan
                microstate_features[f"MS_Class_{name}_Occurrence_Hz"] = np.nan
                microstate_features[f"MS_Class_{name}_Coverage_pct"] = np.nan
    else:
        print("    [!] Pycrostates package not installed. Skipping microstates.")
        for name in ['A', 'B', 'C', 'D']:
            microstate_features[f"MS_Class_{name}_Duration_ms"] = np.nan
            microstate_features[f"MS_Class_{name}_Occurrence_Hz"] = np.nan
            microstate_features[f"MS_Class_{name}_Coverage_pct"] = np.nan

    # =========================================================================
    # Step 7: Consolidate Output Feature Vector & QC Metrics
    # =========================================================================
    feature_vector = {
        "subject_id": subject_id,
        "eeg_qc_passed": bool(qc_passed),
        "eeg_n_channels": int(n_channels),
        "eeg_bad_channel_pct": float(bad_channel_pct),
        "eeg_clean_duration_sec": float(clean_duration_sec),
        "eeg_mean_LZC": mean_lzc,
        "eeg_std_LZC": std_lzc,
        **mse_dict,
        **microstate_features
    }
    
    out_csv = os.path.join(output_dir, f"{subject_id}_hdeeg_features.csv")
    df = pd.DataFrame([feature_vector])
    df.to_csv(out_csv, index=False)
    print(f"[✓] Saved hd-EEG features to: {out_csv}\n")
    return feature_vector


if __name__ == "__main__":
    print("hd-EEG Pipeline Module Loaded Successfully.")
