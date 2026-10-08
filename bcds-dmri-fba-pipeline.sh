#!/usr/bin/env bash
# ===============================================================================
# PIPELINE 2: dMRI FIXEL-BASED ANALYSIS (FBA) & PATHOCONNECTOMICS
# ===============================================================================
# Project: BCDSbio - Computational Neuroengineering Framework for mTBI
# Author: Costanza Brunati - Brain Mapping Lab, Department of Biomedical, Dental Sciences and Morphological and Functional Imaging, University of Messina, Messina, Italy
#
# OPEN-SOURCE CODEBASES & SOURCES TO ADAPT / CITATION REFERENCES:
# 1. MRtrix3 & FBA Framework:
#    - Official Site & Docs: https://www.mrtrix.org / https://mrtrix.readthedocs.io
#    - Citation: Tournier et al. MRtrix3: A fast, flexible and open software framework... NeuroImage 2019.
#    - Citation: Raffelt et al. Investigating white matter fibre density and cross-section using fixel-based analysis. NeuroImage 2017.
# 2. ACT & SIFT2 Filtering:
#    - Citation: Smith et al. Anatomically-constrained tractography (ACT). NeuroImage 2012.
#    - Citation: Smith et al. SIFT2: Spherical-deconvolution informed filtering of tractograms. NeuroImage 2015.
# 3. FSL Eddy & Distortion Correction:
#    - GitHub / FSL: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy
#    - Citation: Andersson & Sotiropoulos. An integrated platform for preprocessing dMRI data. NeuroImage 2016.
# ===============================================================================

set -e # Exit immediately on error

SUBJECT_ID=$1
BIDS_DIR=$2
OUTPUT_DIR=$3

if [ -z "$SUBJECT_ID" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: bash 02_dmri_fba_pipeline.sh <subject_id> <bids_dir> <output_dir>"
    exit 1
fi

echo "==================================================================="
echo "[*] Starting dMRI FBA & Connectomics Pipeline for: ${SUBJECT_ID}"
echo "==================================================================="

SCRATCH_DIR="${OUTPUT_DIR}/scratch_${SUBJECT_ID}"
mkdir -p "${SCRATCH_DIR}"
mkdir -p "${OUTPUT_DIR}/fba_metrics"
mkdir -p "${OUTPUT_DIR}/connectomics"

# Inputs
DWI_RAW="${BIDS_DIR}/${SUBJECT_ID}/dwi/${SUBJECT_ID}_dwi.nii.gz"
BVAL="${BIDS_DIR}/${SUBJECT_ID}/dwi/${SUBJECT_ID}_dwi.bval"
BVEC="${BIDS_DIR}/${SUBJECT_ID}/dwi/${SUBJECT_ID}_dwi.bvec"
T1W_RAW="${BIDS_DIR}/${SUBJECT_ID}/anat/${SUBJECT_ID}_T1w.nii.gz"

# Step 1: Convert BIDS to MRtrix .mif format
echo "[Step 1] Converting input DWI to mif format..."
mrconvert "${DWI_RAW}" "${SCRATCH_DIR}/dwi_raw.mif" -fslgrad "${BVEC}" "${BVAL}" -force

# Step 2: Denoising & MP-PCA noise map
echo "[Step 2] Applying MP-PCA Denoising (dwidenoise)..."
dwidenoise "${SCRATCH_DIR}/dwi_raw.mif" "${SCRATCH_DIR}/dwi_denoised.mif" -noise "${SCRATCH_DIR}/noise.mif" -force

# Step 3: Removal of Gibbs Ringing Artifacts
echo "[Step 3] Removing Gibbs Ringing (mrdegibbs)..."
mrdegibbs "${SCRATCH_DIR}/dwi_denoised.mif" "${SCRATCH_DIR}/dwi_unring.mif" -force

# Step 4: Motion, Eddy Current, and Slice Dropout Correction via FSL Eddy
echo "[Step 4] Preprocessing with FSL Eddy & Motion Correction (dwifslpreproc)..."
dwifslpreproc "${SCRATCH_DIR}/dwi_unring.mif" "${SCRATCH_DIR}/dwi_preproc.mif" \
    -rpe_none -pe_dir AP \
    -eddy_options " --slm=linear --repol " -force

# Step 5: Bias Field Correction (dwibiascorrect ants/fast)
echo "[Step 5] Bias Field Correction (dwibiascorrect)..."
dwibiascorrect ants "${SCRATCH_DIR}/dwi_preproc.mif" "${SCRATCH_DIR}/dwi_unbiased.mif" -bias "${SCRATCH_DIR}/bias.mif" -force

# Step 6: Global Intensity Normalization
echo "[Step 6] Global Intensity Normalization (dwinormalise)..."
dwi2mask "${SCRATCH_DIR}/dwi_unbiased.mif" "${SCRATCH_DIR}/dwi_mask.mif" -force
dwinormalise group "${SCRATCH_DIR}/dwi_unbiased.mif" "${SCRATCH_DIR}/dwi_mask.mif" "${SCRATCH_DIR}/dwi_norm.mif" -force

# Step 7: Response Function Estimation (Multi-Shell Multi-Tissue CSD - dhollander)
echo "[Step 7] Estimating Tissue Response Functions (MSMT-CSD)..."
dwi2response dhollander "${SCRATCH_DIR}/dwi_norm.mif" \
    "${SCRATCH_DIR}/wm_response.txt" \
    "${SCRATCH_DIR}/gm_response.txt" \
    "${SCRATCH_DIR}/csf_response.txt" -force

# Step 8: Constrained Spherical Deconvolution (dwi2fod msmt_csd)
echo "[Step 8] Calculating Fiber Orientation Distributions (FODs)..."
dwi2fod msmt_csd "${SCRATCH_DIR}/dwi_norm.mif" \
    "${SCRATCH_DIR}/wm_response.txt" "${SCRATCH_DIR}/wm_fod.mif" \
    "${SCRATCH_DIR}/gm_response.txt" "${SCRATCH_DIR}/gm_fod.mif" \
    "${SCRATCH_DIR}/csf_response.txt" "${SCRATCH_DIR}/csf_fod.mif" \
    -mask "${SCRATCH_DIR}/dwi_mask.mif" -force

# Step 9: Fixel-Based Analysis (FBA) Metrics Generation
echo "[Step 9] Segmenting FODs and Extracting Fixel Metrics (FD, log-FC, FDC)..."
fod2fixel "${SCRATCH_DIR}/wm_fod.mif" "${SCRATCH_DIR}/fixels_indiv" -mask "${SCRATCH_DIR}/dwi_mask.mif" -force

# Compute Fibre Density (FD)
fixelreorient "${SCRATCH_DIR}/fixels_indiv" "${SCRATCH_DIR}/dwi_preproc.mif" "${SCRATCH_DIR}/fixels_reoriented" -force

# Step 10: Structural Tractography with ACT (Anatomically-Constrained Tractography) & SIFT2
echo "[Step 10] Generating 5TT structural mask and ACT Tractography (10M streamlines)..."
mrconvert "${T1W_RAW}" "${SCRATCH_DIR}/t1.mif" -force
5ttgen fsl "${SCRATCH_DIR}/t1.mif" "${SCRATCH_DIR}/5tt.mif" -preprocessed -force

# 10 Million Probabilistic Streamlines (iFOD2)
tckgen "${SCRATCH_DIR}/wm_fod.mif" "${SCRATCH_DIR}/tracks_10M.tck" \
    -act "${SCRATCH_DIR}/5tt.mif" -backtrack -crop_at_gmwmi \
    -maxlength 250 -minlength 10 -select 10M -force

# SIFT2 Weighting
echo "[Step 10b] Applying SIFT2 Streamline Weighting..."
tcksift2 "${SCRATCH_DIR}/tracks_10M.tck" "${SCRATCH_DIR}/wm_fod.mif" "${SCRATCH_DIR}/sift2_weights.txt" -force

# Step 11: Structural Connectome Matrix Generation (Schaefer-400 + Tian Subcortical Atlas)
echo "[Step 11] Generating SIFT2-weighted Structural Connectome Matrix..."
# Parcellation image: Combined Schaefer-400 + Tian subcortical atlas
PARCELLATION="${BIDS_DIR}/atlases/schaefer400_tianSubcortical_combined.nii.gz"
mrconvert "${PARCELLATION}" "${SCRATCH_DIR}/parcellation.mif" -force

tck2connectome "${SCRATCH_DIR}/tracks_10M.tck" "${SCRATCH_DIR}/parcellation.mif" \
    "${OUTPUT_DIR}/connectomics/${SUBJECT_ID}_structural_connectome_sift2.csv" \
    -tck_weights_in "${SCRATCH_DIR}/sift2_weights.txt" \
    -symmetric -zero_diagonal -force

echo "==================================================================="
echo "[✓] dMRI FBA & Connectomics Pipeline Completed for: ${SUBJECT_ID}"
echo "==================================================================="
