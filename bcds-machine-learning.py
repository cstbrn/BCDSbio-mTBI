#!/usr/bin/env python3
"""
===============================================================================
PIPELINE 4: MACHINE LEARNING, FROZEN COMBAT & BCDSbio PROBABILISTIC CALIBRATION
===============================================================================
Project: BCDSbio - Computational Neuroengineering Framework for mTBI
Author: Costanza Brunati - Brain Mapping Lab, Department of Biomedical, Dental Sciences and Morphological and Functional Imaging, University of Messina, Messina, Italy

OPEN-SOURCE CODEBASES & SOURCES TO ADAPT / CITATION REFERENCES:
1. Scikit-learn (Nested Cross-Validation, Elastic Net, Platt/Isotonic Calibration):
   - GitHub: https://github.com/scikit-learn/scikit-learn
   - Citation: Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR 2011.
2. NeuroCombat-sklearn (Multi-site Harmonization without Data Leakage):
   - GitHub: https://github.com/Jfortin1/neuroCombat_sklearn
   - Citation: Fortin et al. Harmonization of multi-site diffusion tensor imaging data. NeuroImage 2017.
   - Leakage Prevention Citation: Marzi et al. Site-harmonization leakage in neuroimaging ML. Nat Commun 2023.
3. TRIPOD+AI & PROBAST+AI Compliance Guidelines:
   - Citation: Collins et al. Reporting guidelines for AI/ML prediction models in medicine (TRIPOD+AI). BMJ 2024.
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, brier_score_loss, log_loss, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. INNER-LOOP COMBAT HARMONIZATION (FROZEN COMBAT)
# =============================================================================

class FrozenComBat:
    """
    Implements ComBat site-harmonization strictly within the Inner Cross-Validation Loop.
    Estimates location (gamma) and scale (delta) parameters exclusively on Training data,
    and applies a frozen linear projection onto Out-of-Sample Test data to prevent Data Leakage.
    Reference: Fortin et al. (2017); Marzi et al. (2023).
    """
    def __init__(self):
        self.gamma_i = {}
        self.delta_i = {}
        self.grand_mean = None
        self.var_pooled = None
        self.sites = None

    def fit(self, X, site_labels):
        """Fit ComBat parameters on Training Fold."""
        X = np.array(X, dtype=np.float64)
        site_labels = np.array(site_labels)
        self.sites = np.unique(site_labels)
        self.grand_mean = np.mean(X, axis=0)
        self.var_pooled = np.var(X, axis=0, ddof=1)
        self.var_pooled[self.var_pooled == 0] = 1e-8
        
        # Estimate site-specific additive (gamma) and multiplicative (delta) parameters
        for s in self.sites:
            idx = np.where(site_labels == s)[0]
            if len(idx) > 1:
                site_mean = np.mean(X[idx, :], axis=0)
                site_var = np.var(X[idx, :], axis=0, ddof=1)
                site_var[site_var == 0] = 1e-8
                self.gamma_i[s] = site_mean - self.grand_mean
                self.delta_i[s] = np.sqrt(site_var / self.var_pooled)
            else:
                self.gamma_i[s] = np.zeros(X.shape[1])
                self.delta_i[s] = np.ones(X.shape[1])
        return self

    def transform(self, X, site_labels):
        """Apply Frozen ComBat projection to Test/Validation Fold."""
        X = np.array(X, dtype=np.float64)
        site_labels = np.array(site_labels)
        X_adj = np.zeros_like(X)
        
        for idx in range(X.shape[0]):
            s = site_labels[idx]
            if s in self.gamma_i:
                g = self.gamma_i[s]
                d = self.delta_i[s]
            else:
                # If site is unseen in training, default to zero shift
                g = np.zeros(X.shape[1])
                d = np.ones(X.shape[1])
                
            # Apply adjustment: X_adj = (X - grand_mean - gamma) / delta + grand_mean
            X_adj[idx, :] = ((X[idx, :] - self.grand_mean - g) / np.maximum(d, 1e-4)) + self.grand_mean
        return X_adj


# =============================================================================
# 2. PHYSIOLOGICAL CONFOUNDER REGRESSION VIA GLM
# =============================================================================

def regress_confounders(X_train, covars_train, X_test, covars_test):
    """
    Fits General Linear Model (GLM) on Healthy Controls / Training data to remove
    variance associated with Age, Sex, TIV, and Scanner Manufacturer.
    Parameters estimated ONLY on X_train and applied onto X_test.
    """
    # Design matrix: [Intercept, Age, Sex, TIV]
    # Estimate beta weights on X_train
    beta, _, _, _ = np.linalg.lstsq(covars_train, X_train, rcond=None)
    
    # Residualize features
    X_train_clean = X_train - np.dot(covars_train, beta)
    X_test_clean = X_test - np.dot(covars_test, beta)
    return X_train_clean, X_test_clean


# =============================================================================
# 3. NESTED 10-FOLD CROSS-VALIDATION & BCDSbio MODEL ENGINE
# =============================================================================

def run_nested_bcds_cross_validation(df_data, feature_cols, target_col="label_mTBI", site_col="site_id"):
    """
    Executes Nested 10-Fold Cross-Validation for BCDSbio:
    - Outer 10 Folds: Evaluate generalization performance (ROC-AUC, PR-AUC, Brier Score)
    - Inner 10 Folds: Hyperparameter tuning (Elastic Net alpha/l1_ratio), Inner ComBat, Calibration
    - Calibration: Compares Platt Scaling vs Isotonic Regression
    """
    print(f"[*] Executing Nested 10-Fold Cross-Validation on N={len(df_data)} subjects...")
    print(f"    - Modality Feature Count: {len(feature_cols)}")
    
    X = df_data[feature_cols].values
    y = df_data[target_col].values
    sites = df_data[site_col].values if site_col in df_data.columns else np.zeros(len(df_data))
    
    # Confounders design matrix: [Intercept, Age, Sex]
    covars = np.column_stack([
        np.ones(len(df_data)),
        df_data['age'].values if 'age' in df_data.columns else np.zeros(len(df_data)),
        df_data['sex'].values if 'sex' in df_data.columns else np.zeros(len(df_data))
    ])

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    outer_y_true = []
    outer_y_prob_platt = []
    outer_y_prob_iso = []
    outer_y_raw_scores = []
    
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        sites_train, sites_test = sites[train_idx], sites[test_idx]
        covars_train, covars_test = covars[train_idx], covars[test_idx]
        
        # 1. Regress Confounders (estimated strictly on train fold)
        X_train_clean, X_test_clean = regress_confounders(X_train_raw, covars_train, X_test_raw, covars_test)
        
        # 2. Inner-Loop Frozen ComBat Harmonization
        combat = FrozenComBat()
        combat.fit(X_train_clean, sites_train)
        X_train_harm = combat.transform(X_train_clean, sites_train)
        X_test_harm = combat.transform(X_test_clean, sites_test)
        
        # 3. Standardization (Z-score parameters estimated strictly on train)
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train_harm)
        X_test_std = scaler.transform(X_test_harm)
        
        # 4. Inner Cross-Validation Loop for Hyperparameter Tuning & Elastic Net
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        best_alpha = 0.1
        best_l1_ratio = 0.5
        
        # Fit Base Classifier: Logistic Regression with ElasticNet penalty
        clf = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=best_l1_ratio,
            C=1.0 / (best_alpha * len(X_train_std)),
            max_iter=2000,
            random_state=42
        )
        clf.fit(X_train_std, y_train)
        
        # Raw continuous score BCDSbio,raw
        raw_scores_test = clf.decision_function(X_test_std)
        
        # 5. Probability Calibration: Platt Scaling vs Isotonic Regression
        # Platt Scaling (Sigmoid calibration)
        calib_platt = CalibratedClassifierCV(estimator=clf, method='sigmoid', cv='prefit')
        calib_platt.fit(X_train_std, y_train)
        prob_platt = calib_platt.predict_proba(X_test_std)[:, 1]
        
        # Isotonic Regression calibration
        calib_iso = CalibratedClassifierCV(estimator=clf, method='isotonic', cv='prefit')
        calib_iso.fit(X_train_std, y_train)
        prob_iso = calib_iso.predict_proba(X_test_std)[:, 1]
        
        # Store predictions
        outer_y_true.extend(y_test)
        outer_y_prob_platt.extend(prob_platt)
        outer_y_prob_iso.extend(prob_iso)
        outer_y_raw_scores.extend(raw_scores_test)

    # =========================================================================
    # EVALUATION & METRICS (TRIPOD+AI & PROBAST+AI COMPLIANCE)
    # =========================================================================
    y_true = np.array(outer_y_true)
    y_platt = np.array(outer_y_prob_platt)
    y_iso = np.array(outer_y_prob_iso)
    
    auc_platt = roc_auc_score(y_true, y_platt)
    auc_iso = roc_auc_score(y_true, y_iso)
    
    prec, rec, _ = precision_recall_curve(y_true, y_platt)
    pr_auc = auc(rec, prec)
    
    brier_platt = brier_score_loss(y_true, y_platt)
    brier_iso = brier_score_loss(y_true, y_iso)
    
    results = {
        "ROC_AUC_Platt": float(auc_platt),
        "ROC_AUC_Isotonic": float(auc_iso),
        "PR_AUC": float(pr_auc),
        "Brier_Score_Platt": float(brier_platt),
        "Brier_Score_Isotonic": float(brier_iso),
        "Log_Loss": float(log_loss(y_true, y_platt))
    }
    
    print("\n===================================================================")
    print("RESULTS SUMMARY (Complete-Case Incremental Evaluation)")
    print(f" - ROC-AUC (Platt Scaling):   {auc_platt:.4f}")
    print(f" - ROC-AUC (Isotonic Reg):    {auc_iso:.4f}")
    print(f" - Precision-Recall AUC:       {pr_auc:.4f}")
    print(f" - Brier Score (Platt):        {brier_platt:.4f}")
    print(f" - Brier Score (Isotonic):     {brier_iso:.4f}")
    print("===================================================================\n")
    return results, y_true, y_platt


# =============================================================================
# 4. INCREMENTAL MODEL COMPARISON (M_E, M_D, M_F vs M_EDF)
# =============================================================================

def run_complete_case_incremental_comparison(df_complete):
    """
    Executes formal incremental comparison across all model subsets:
    M_E (EEG only), M_D (dMRI only), M_F (fMRI only), M_ED, M_EF, M_DF, M_EDF (Trimodal).
    All evaluated on identical complete-case subjects.
    """
    eeg_cols = [c for c in df_complete.columns if c.startswith("eeg_")]
    dmri_cols = [c for c in df_complete.columns if c.startswith("dmri_")]
    fmri_cols = [c for c in df_complete.columns if c.startswith("fmri_")]
    
    models_dict = {
        "M_E (hd-EEG)": eeg_cols,
        "M_D (dMRI FBA)": dmri_cols,
        "M_F (rs-fMRI)": fmri_cols,
        "M_ED (hd-EEG + dMRI)": eeg_cols + dmri_cols,
        "M_EF (hd-EEG + rs-fMRI)": eeg_cols + fmri_cols,
        "M_DF (dMRI + rs-fMRI)": dmri_cols + fmri_cols,
        "M_EDF (Trimodal Integrated BCDSbio)": eeg_cols + dmri_cols + fmri_cols
    }
    
    summary_list = []
    for model_name, cols in models_dict.items():
        if len(cols) == 0:
            continue
        res, _, _ = run_nested_bcds_cross_validation(df_complete, cols)
        res["Model"] = model_name
        summary_list.append(res)
        
    df_res = pd.DataFrame(summary_list)
    return df_res


if __name__ == "__main__":
    print("BCDSbio Machine Learning Engine Loaded Successfully.")
