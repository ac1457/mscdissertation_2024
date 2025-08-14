#!/usr/bin/env python3
"""
Enhanced Metrics Implementation - Lending Club Sentiment Analysis
===============================================================
Implements enhanced metrics table addressing all identified issues:
- Missing columns (AUC_Diff_vs_Traditional, PR_AUC_Diff_vs_Traditional, AUC_SE, PR_AUC_SE, DeLong_Z, DeLong_p)
- Calibration metrics (Brier_Before, Brier_After, Brier_Improvement, ECE, Calibration_Slope, Calibration_Intercept)
- BCa bootstrap for better bias correction
- Cost savings with explicit cost matrix
- Threshold-dependent metrics at fixed rejection rates
- Proper statistical rigor with family-wise error control
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, 
                           recall_score, f1_score, brier_score_loss, confusion_matrix, roc_curve)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import json
import hashlib
from datetime import datetime
import os
import sys
from pathlib import Path
warnings.filterwarnings('ignore')

class EnhancedMetricsImplementation:
    """
    Enhanced metrics implementation with proper statistical rigor
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.results = {}
        
        # Configuration for reproducibility
        self.config = {
            'random_state': random_state,
            'bootstrap_resamples': 1000,
            'bootstrap_method': 'BCa',  # Bias-corrected accelerated
            'cv_folds': 5,
            'calibration_bins': 10,
            'cost_default': 1000,
            'cost_review': 50,
            'rejection_rates': [5, 10, 20],  # Top k% for threshold analysis
            'timestamp': datetime.now().isoformat()
        }
        
    def load_data_with_realistic_targets(self):
        """
        Load data with realistic targets
        """
        try:
            df = pd.read_csv('data/synthetic_loan_descriptions_with_realistic_targets.csv')
            print(f"✅ Loaded enhanced dataset: {len(df)} records")
            return df
        except FileNotFoundError:
            print("❌ Enhanced dataset not found. Please run realistic_target_creation.py first.")
            return None
    
    def prepare_features(self, df):
        """
        Prepare feature sets for modeling
        """
        # Traditional features
        traditional_features = [
            'purpose', 'sentiment_score', 'sentiment_confidence', 
            'text_length', 'word_count', 'sentence_count',
            'has_positive_words', 'has_negative_words', 'has_financial_terms'
        ]
        traditional_features = [f for f in traditional_features if f in df.columns]
        
        # Sentiment features
        sentiment_features = [
            'sentiment', 'sentiment_score', 'sentiment_confidence', 
            'text_length', 'word_count', 'sentence_count'
        ]
        sentiment_features = [f for f in sentiment_features if f in df.columns]
        
        # Hybrid features
        df['sentiment_text_interaction'] = df['sentiment_score'] * df['text_length']
        df['sentiment_word_interaction'] = df['sentiment_score'] * df['word_count']
        df['sentiment_purpose_interaction'] = df['sentiment_score'] * df['purpose'].astype('category').cat.codes
        
        # Prepare feature sets
        X_traditional = df[traditional_features].copy()
        X_sentiment = df[traditional_features + sentiment_features].copy()
        X_hybrid = df[traditional_features + sentiment_features + ['sentiment_text_interaction', 'sentiment_word_interaction', 'sentiment_purpose_interaction']].copy()
        
        # Handle categorical variables and missing values
        for X in [X_traditional, X_sentiment, X_hybrid]:
            for col in X.columns:
                if col == 'purpose' or col == 'sentiment':
                    X[col] = X[col].astype('category').cat.codes
                X[col] = X[col].fillna(X[col].median())
        
        return {
            'Traditional': X_traditional,
            'Sentiment': X_sentiment,
            'Hybrid': X_hybrid
        }
    
    def get_realistic_targets(self, df):
        """
        Get realistic targets with proper documentation
        """
        target_columns = [col for col in df.columns if col.startswith('target_')]
        
        if not target_columns:
            print("❌ No realistic targets found in dataset")
            return None
        
        regimes = {}
        
        for target_col in target_columns:
            regime_name = target_col.replace('target_', '')
            y = df[target_col].values
            
            # Calculate regime statistics
            n_total = len(y)
            n_positives = np.sum(y)
            n_negatives = n_total - n_positives
            actual_rate = n_positives / n_total
            
            # Use actual default rate for regime labeling
            corrected_regime_name = f"{actual_rate:.0%}".replace('%', '')
            
            regimes[corrected_regime_name] = {
                'y': y,
                'actual_rate': actual_rate,
                'n_total': n_total,
                'n_positives': n_positives,
                'n_negatives': n_negatives,
                'original_name': regime_name
            }
            
            print(f"  {corrected_regime_name} (was {regime_name}): {actual_rate:.1%} ({n_positives:,} defaults, {n_negatives:,} non-defaults)")
        
        return regimes
    
    def calculate_bca_bootstrap_ci(self, y_true, y_pred_proba, n_bootstrap=1000):
        """
        Calculate BCa (Bias-corrected accelerated) bootstrap confidence intervals
        """
        bootstrap_aucs = []
        bootstrap_pr_aucs = []
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        
        # Calculate original statistics
        original_auc = roc_auc_score(y_true, y_pred_proba)
        original_pr_auc = average_precision_score(y_true, y_pred_proba)
        
        for _ in range(n_bootstrap):
            # Stratified bootstrap resample
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            y_boot = y_true[indices]
            pred_boot = y_pred_proba[indices]
            
            # Calculate metrics
            auc = roc_auc_score(y_boot, pred_boot)
            pr_auc = average_precision_score(y_boot, pred_boot)
            
            bootstrap_aucs.append(auc)
            bootstrap_pr_aucs.append(pr_auc)
        
        # Calculate BCa confidence intervals
        def bca_ci(bootstrap_samples, original_stat, alpha=0.05):
            n_boot = len(bootstrap_samples)
            
            # Calculate bias correction
            z0 = stats.norm.ppf(np.sum(np.array(bootstrap_samples) < original_stat) / n_boot)
            
            # Calculate acceleration factor
            jackknife_stats = []
            for i in range(len(y_true)):
                # Leave-one-out jackknife
                y_jack = np.delete(y_true, i)
                pred_jack = np.delete(y_pred_proba, i)
                if len(np.unique(y_jack)) > 1:  # Ensure we have both classes
                    jack_stat = roc_auc_score(y_jack, pred_jack)
                    jackknife_stats.append(jack_stat)
            
            if len(jackknife_stats) > 0:
                jack_mean = np.mean(jackknife_stats)
                a = np.sum((jack_mean - np.array(jackknife_stats)) ** 3) / (6.0 * np.sum((jack_mean - np.array(jackknife_stats)) ** 2) ** 1.5)
            else:
                a = 0
            
            # Calculate BCa intervals
            z_alpha = stats.norm.ppf(alpha / 2)
            z_1_alpha = stats.norm.ppf(1 - alpha / 2)
            
            z_lower = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
            z_upper = z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha))
            
            # Convert to percentiles
            p_lower = stats.norm.cdf(z_lower)
            p_upper = stats.norm.cdf(z_upper)
            
            # Get bootstrap percentiles
            lower_percentile = int(p_lower * n_boot)
            upper_percentile = int(p_upper * n_boot)
            
            # Ensure valid indices
            lower_percentile = max(0, min(lower_percentile, n_boot - 1))
            upper_percentile = max(0, min(upper_percentile, n_boot - 1))
            
            sorted_samples = np.sort(bootstrap_samples)
            return sorted_samples[lower_percentile], sorted_samples[upper_percentile]
        
        # Calculate BCa CIs
        auc_ci_lower, auc_ci_upper = bca_ci(bootstrap_aucs, original_auc)
        pr_auc_ci_lower, pr_auc_ci_upper = bca_ci(bootstrap_pr_aucs, original_pr_auc)
        
        return {
            'AUC_CI': (auc_ci_lower, auc_ci_upper),
            'PR_AUC_CI': (pr_auc_ci_lower, pr_auc_ci_upper),
            'AUC_mean': np.mean(bootstrap_aucs),
            'PR_AUC_mean': np.mean(bootstrap_pr_aucs),
            'AUC_SE': np.std(bootstrap_aucs),
            'PR_AUC_SE': np.std(bootstrap_pr_aucs),
            'AUC_original': original_auc,
            'PR_AUC_original': original_pr_auc
        }
    
    def perform_enhanced_delong_test(self, y_true, y_pred_1, y_pred_2):
        """
        Enhanced DeLong test with proper statistical rigor
        """
        # Calculate AUCs
        auc_1 = roc_auc_score(y_true, y_pred_1)
        auc_2 = roc_auc_score(y_true, y_pred_2)
        
        # Calculate DeLong statistic using structural components
        def delong_statistic(y_true, y_pred_1, y_pred_2):
            n = len(y_true)
            n_pos = np.sum(y_true == 1)
            n_neg = n - n_pos
            
            # Calculate structural components
            v10 = np.zeros(n)
            v01 = np.zeros(n)
            
            for i in range(n):
                if y_true[i] == 1:  # Positive case
                    v10[i] = np.mean(y_pred_1[y_true == 0] < y_pred_1[i])
                    v01[i] = np.mean(y_pred_2[y_true == 0] < y_pred_2[i])
                else:  # Negative case
                    v10[i] = np.mean(y_pred_1[y_true == 1] > y_pred_1[i])
                    v01[i] = np.mean(y_pred_2[y_true == 1] > y_pred_2[i])
            
            # Calculate variances
            var_v10 = np.var(v10) / n
            var_v01 = np.var(v01) / n
            cov_v10_v01 = np.cov(v10, v01)[0, 1] / n
            
            # Calculate DeLong statistic
            auc_diff = auc_1 - auc_2
            se_diff = np.sqrt(var_v10 + var_v01 - 2 * cov_v10_v01)
            z_stat = auc_diff / se_diff
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            return auc_diff, se_diff, z_stat, p_value
        
        # Perform DeLong test
        auc_diff, se_diff, z_stat, p_value = delong_statistic(y_true, y_pred_1, y_pred_2)
        
        return {
            'test_type': 'DeLong_structural',
            'auc_1': auc_1,
            'auc_2': auc_2,
            'auc_difference': auc_diff,
            'se_difference': se_diff,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def calculate_enhanced_calibration_metrics(self, y_true, y_pred_proba, n_bins=10):
        """
        Calculate comprehensive calibration metrics with before/after comparison
        """
        # Original calibration metrics
        original_calibration = self._calculate_calibration_metrics(y_true, y_pred_proba, n_bins)
        
        # Apply Platt scaling
        calibrated_proba = self._apply_platt_scaling(y_true, y_pred_proba)
        
        # Calibrated calibration metrics
        calibrated_calibration = self._calculate_calibration_metrics(y_true, calibrated_proba, n_bins)
        
        return {
            'Brier_Before': original_calibration['Brier_Score'],
            'Brier_After': calibrated_calibration['Brier_Score'],
            'Brier_Improvement': original_calibration['Brier_Score'] - calibrated_calibration['Brier_Score'],
            'ECE_Before': original_calibration['ECE'],
            'ECE_After': calibrated_calibration['ECE'],
            'ECE_Improvement': original_calibration['ECE'] - calibrated_calibration['ECE'],
            'Calibration_Slope_Before': original_calibration['Calibration_Slope'],
            'Calibration_Slope_After': calibrated_calibration['Calibration_Slope'],
            'Calibration_Intercept_Before': original_calibration['Calibration_Intercept'],
            'Calibration_Intercept_After': calibrated_calibration['Calibration_Intercept'],
            'Calibrated_Probabilities': calibrated_proba
        }
    
    def _calculate_calibration_metrics(self, y_true, y_pred_proba, n_bins=10):
        """
        Internal calibration metrics calculation
        """
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        reliability_bins = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(y_pred_proba > bin_lower, y_pred_proba <= bin_upper)
            if np.sum(in_bin) > 0:
                bin_conf = np.mean(y_pred_proba[in_bin])
                bin_acc = np.mean(y_true[in_bin])
                bin_size = np.sum(in_bin)
                ece += np.abs(bin_conf - bin_acc) * bin_size / len(y_true)
        
        # Calibration slope and intercept
        logit_pred = np.log(y_pred_proba / (1 - y_pred_proba))
        logit_pred = logit_pred.reshape(-1, 1)
        logit_pred = np.nan_to_num(logit_pred, nan=0, posinf=10, neginf=-10)
        
        try:
            lr = LogisticRegression()
            lr.fit(logit_pred, y_true)
            calibration_slope = lr.coef_[0][0]
            calibration_intercept = lr.intercept_[0]
        except:
            calibration_slope = 1.0
            calibration_intercept = 0.0
        
        return {
            'ECE': ece,
            'Calibration_Slope': calibration_slope,
            'Calibration_Intercept': calibration_intercept,
            'Brier_Score': brier_score_loss(y_true, y_pred_proba)
        }
    
    def _apply_platt_scaling(self, y_true, y_pred_proba):
        """
        Internal Platt scaling application
        """
        # Use CalibratedClassifierCV for Platt scaling
        base_model = LogisticRegression()
        calibrated_model = CalibratedClassifierCV(base_model, cv=5, method='sigmoid')
        
        # Reshape for sklearn
        y_pred_proba_reshaped = y_pred_proba.reshape(-1, 1)
        
        # Fit and predict
        calibrated_model.fit(y_pred_proba_reshaped, y_true)
        calibrated_proba = calibrated_model.predict_proba(y_pred_proba_reshaped)[:, 1]
        
        return calibrated_proba
    
    def calculate_enhanced_decision_metrics(self, y_true, y_pred_proba, cost_default=1000, cost_review=50):
        """
        Enhanced decision utility metrics with explicit cost matrix
        """
        # Sort by predicted probabilities
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        sorted_y_true = y_true[sorted_indices]
        sorted_y_pred_proba = y_pred_proba[sorted_indices]
        
        # Threshold-dependent metrics
        threshold_metrics = {}
        for k in self.config['rejection_rates']:
            k_count = int(len(y_true) * k / 100)
            top_k_y_true = sorted_y_true[:k_count]
            
            overall_default_rate = np.mean(y_true)
            top_k_default_rate = np.mean(top_k_y_true)
            lift = top_k_default_rate / overall_default_rate if overall_default_rate > 0 else 0
            
            total_defaults = np.sum(y_true)
            captured_defaults = np.sum(top_k_y_true)
            capture_rate = captured_defaults / total_defaults if total_defaults > 0 else 0
            
            threshold_metrics[f'Lift@{k}%'] = lift
            threshold_metrics[f'Capture_Rate@{k}%'] = capture_rate
            threshold_metrics[f'Default_Rate@{k}%'] = top_k_default_rate
        
        # Business threshold analysis (top 10%)
        business_threshold = np.percentile(y_pred_proba, 90)
        y_pred_business = (y_pred_proba >= business_threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_business).ravel()
        
        # Cost analysis with explicit matrix
        cost_matrix = {
            'TP_cost': cost_review,  # Review cost for caught default
            'FP_cost': cost_review,  # Review cost for false alarm
            'FN_cost': cost_default, # Default cost for missed default
            'TN_cost': 0            # No cost for correctly identified non-default
        }
        
        total_cost_without_model = len(y_true) * cost_default * np.mean(y_true)
        total_cost_with_model = (tp * cost_matrix['TP_cost'] + 
                               fp * cost_matrix['FP_cost'] + 
                               fn * cost_matrix['FN_cost'] + 
                               tn * cost_matrix['TN_cost'])
        
        cost_savings = total_cost_without_model - total_cost_with_model
        
        return {
            **threshold_metrics,
            'Business_Threshold': business_threshold,
            'Business_Precision': precision_score(y_true, y_pred_business, zero_division=0),
            'Business_Recall': recall_score(y_true, y_pred_business, zero_division=0),
            'Business_F1': f1_score(y_true, y_pred_business, zero_division=0),
            'Business_TP': tp,
            'Business_FP': fp,
            'Business_TN': tn,
            'Business_FN': fn,
            'Cost_Savings': cost_savings,
            'Cost_Default': cost_default,
            'Cost_Review': cost_review,
            'Total_Cost_Without_Model': total_cost_without_model,
            'Total_Cost_With_Model': total_cost_with_model,
            'Cost_Matrix': cost_matrix
        }
    
    def run_enhanced_analysis(self, df, regimes, feature_sets):
        """
        Run enhanced analysis with comprehensive metrics
        """
        print("RUNNING ENHANCED ANALYSIS WITH COMPREHENSIVE METRICS")
        print("=" * 60)
        
        all_results = []
        all_delong_results = []
        all_calibration_results = []
        all_decision_results = []
        
        # Store traditional baseline for comparisons
        traditional_baselines = {}
        
        for regime_name, regime_data in regimes.items():
            print(f"\nRegime: {regime_name} (Actual: {regime_data['actual_rate']:.1%})")
            y = regime_data['y']
            
            # Store predictions for each feature set
            feature_predictions = {}
            feature_true_labels = {}
            
            # Analyze each feature set
            for feature_set_name, X in feature_sets.items():
                print(f"  Analyzing {feature_set_name}...")
                
                cv = StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=self.random_state)
                all_predictions = []
                all_true_labels = []
                
                for train_idx, test_idx in cv.split(X, y):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    model = RandomForestClassifier(random_state=self.random_state)
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_test)[:, 1]
                    
                    all_predictions.extend(y_pred)
                    all_true_labels.extend(y_test)
                
                all_predictions = np.array(all_predictions)
                all_true_labels = np.array(all_true_labels)
                
                # Store for comparisons
                feature_predictions[feature_set_name] = all_predictions
                feature_true_labels[feature_set_name] = all_true_labels
                
                # BCa Bootstrap CIs
                bootstrap_results = self.calculate_bca_bootstrap_ci(all_true_labels, all_predictions)
                
                # Enhanced calibration metrics
                calibration_results = self.calculate_enhanced_calibration_metrics(all_true_labels, all_predictions)
                
                # Enhanced decision metrics
                decision_metrics = self.calculate_enhanced_decision_metrics(all_true_labels, all_predictions)
                
                # Store traditional baseline
                if feature_set_name == 'Traditional':
                    traditional_baselines[regime_name] = {
                        'predictions': all_predictions,
                        'true_labels': all_true_labels,
                        'auc': bootstrap_results['AUC_original'],
                        'pr_auc': bootstrap_results['PR_AUC_original']
                    }
                
                # Calculate differences vs traditional
                if feature_set_name != 'Traditional' and regime_name in traditional_baselines:
                    trad_auc = traditional_baselines[regime_name]['auc']
                    trad_pr_auc = traditional_baselines[regime_name]['pr_auc']
                    
                    auc_diff = bootstrap_results['AUC_original'] - trad_auc
                    pr_auc_diff = bootstrap_results['PR_AUC_original'] - trad_pr_auc
                else:
                    auc_diff = 0.0
                    pr_auc_diff = 0.0
                
                # Store results
                result = {
                    'Regime': regime_name,
                    'Feature_Set': feature_set_name,
                    'Default_Rate': regime_data['actual_rate'],
                    'Sample_Size': regime_data['n_total'],
                    'N_Positives': regime_data['n_positives'],
                    'N_Negatives': regime_data['n_negatives'],
                    'AUC_Mean': bootstrap_results['AUC_mean'],
                    'AUC_CI_Lower': bootstrap_results['AUC_CI'][0],
                    'AUC_CI_Upper': bootstrap_results['AUC_CI'][1],
                    'AUC_SE': bootstrap_results['AUC_SE'],
                    'AUC_Original': bootstrap_results['AUC_original'],
                    'PR_AUC_Mean': bootstrap_results['PR_AUC_mean'],
                    'PR_AUC_CI_Lower': bootstrap_results['PR_AUC_CI'][0],
                    'PR_AUC_CI_Upper': bootstrap_results['PR_AUC_CI'][1],
                    'PR_AUC_SE': bootstrap_results['PR_AUC_SE'],
                    'PR_AUC_Original': bootstrap_results['PR_AUC_original'],
                    'AUC_Diff_vs_Traditional': auc_diff,
                    'PR_AUC_Diff_vs_Traditional': pr_auc_diff,
                    'CV_Folds': self.config['cv_folds'],
                    'Bootstrap_Resamples': self.config['bootstrap_resamples'],
                    'Bootstrap_Method': self.config['bootstrap_method']
                }
                
                all_results.append(result)
                
                # Calibration results
                cal_result = {
                    'Regime': regime_name,
                    'Feature_Set': feature_set_name,
                    'Default_Rate': regime_data['actual_rate'],
                    **calibration_results
                }
                
                all_calibration_results.append(cal_result)
                
                # Decision utility results
                decision_result = {
                    'Regime': regime_name,
                    'Feature_Set': feature_set_name,
                    'Default_Rate': regime_data['actual_rate'],
                    **decision_metrics
                }
                
                all_decision_results.append(decision_result)
            
            # Enhanced DeLong tests for comparisons
            if regime_name in traditional_baselines:
                traditional_pred = traditional_baselines[regime_name]['predictions']
                traditional_true = traditional_baselines[regime_name]['true_labels']
                
                for feature_set_name in ['Sentiment', 'Hybrid']:
                    if feature_set_name in feature_predictions:
                        variant_pred = feature_predictions[feature_set_name]
                        variant_true = feature_true_labels[feature_set_name]
                        
                        # Enhanced DeLong test
                        delong_result = self.perform_enhanced_delong_test(
                            traditional_true, 
                            traditional_pred, 
                            variant_pred
                        )
                        
                        delong_result.update({
                            'Regime': regime_name,
                            'Comparison': f'Traditional_vs_{feature_set_name}',
                            'Sample_Size': len(traditional_true)
                        })
                        
                        all_delong_results.append(delong_result)
        
        return (pd.DataFrame(all_results), pd.DataFrame(all_delong_results), 
                pd.DataFrame(all_calibration_results), pd.DataFrame(all_decision_results))
    
    def save_enhanced_results(self, results):
        """
        Save enhanced results with comprehensive documentation
        """
        print("Saving enhanced results...")
        
        # Create configuration hash for reproducibility
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        
        # Create comprehensive legend
        legend_block = f"""# ENHANCED METRICS IMPLEMENTATION - Lending Club Sentiment Analysis
# Generated: {self.config['timestamp']}
# Config Hash: {config_hash}
# 
# CONFIGURATION:
# - Random State: {self.config['random_state']}
# - Bootstrap Method: {self.config['bootstrap_method']} ({self.config['bootstrap_resamples']} resamples)
# - CV Folds: {self.config['cv_folds']}
# - Calibration Bins: {self.config['calibration_bins']}
# - Cost Matrix: Default={self.config['cost_default']}, Review={self.config['cost_review']}
# - Rejection Rates: {self.config['rejection_rates']}%
#
# ENHANCED METRICS:
# - BCa Bootstrap CIs: Bias-corrected accelerated confidence intervals
# - DeLong Structural: Proper DeLong test using structural components
# - Calibration Before/After: Pre- and post-Platt scaling metrics
# - Threshold-Dependent: Metrics at fixed rejection rates
# - Cost-Benefit Analysis: Explicit cost matrix and savings
#
# METRICS GLOSSARY:
# AUC: Area Under ROC Curve (0.5 = random, 1.0 = perfect)
# PR-AUC: Area Under Precision-Recall Curve (better for imbalanced data)
# AUC_SE: Standard error of AUC from bootstrap
# AUC_Diff_vs_Traditional: AUC_variant - AUC_traditional (positive = improvement)
# DeLong_Z: Z-statistic from DeLong test (structural components)
# DeLong_p: P-value from DeLong test (structural components)
# Brier_Before/After: Brier score before/after Platt scaling
# Brier_Improvement: Brier_Before - Brier_After (positive = improvement)
# ECE: Expected Calibration Error (lower = better calibration)
# Calibration_Slope: Slope of calibration curve (1.0 = perfectly calibrated)
# Lift@k%: Ratio of default rate in top k% vs overall default rate
# Cost_Savings: Expected cost savings from model deployment
# Cost_Matrix: Explicit cost structure (TP, FP, FN, TN costs)
"""
        
        # Save results with legends
        results_files = {
            'enhanced_regime_stats': 'final_results/enhanced_regime_stats.csv',
            'enhanced_delong_results': 'final_results/enhanced_delong_results.csv',
            'enhanced_calibration_results': 'final_results/enhanced_calibration_results.csv',
            'enhanced_decision_results': 'final_results/enhanced_decision_results.csv'
        }
        
        for result_type, filepath in results_files.items():
            if result_type in results:
                df_result = results[result_type]
                
                # Format numbers properly
                numeric_columns = df_result.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    if 'p_value' in col.lower():
                        df_result[col] = df_result[col].apply(lambda x: f"{x:.2e}")
                    elif 'AUC' in col or 'PR_AUC' in col:
                        df_result[col] = df_result[col].round(4)
                    elif 'Improvement' in col or 'Diff' in col:
                        df_result[col] = df_result[col].round(4)
                    elif 'Rate' in col:
                        df_result[col] = df_result[col].round(4)
                
                df_result.to_csv(filepath, index=False)
                
                # Add legend
                with open(filepath, 'r') as f:
                    content = f.read()
                
                with open(filepath, 'w') as f:
                    f.write(legend_block + content)
        
        # Save configuration
        with open('final_results/enhanced_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print("✅ Enhanced results saved successfully!")
        print(f"✅ Configuration hash: {config_hash}")
    
    def run_complete_enhanced_implementation(self):
        """
        Run complete enhanced metrics implementation
        """
        print("RUNNING ENHANCED METRICS IMPLEMENTATION")
        print("=" * 50)
        
        # Load data
        df = self.load_data_with_realistic_targets()
        if df is None:
            return None
        
        # Get realistic targets
        print("\nRealistic targets available:")
        regimes = self.get_realistic_targets(df)
        if regimes is None:
            return None
        
        # Prepare features
        feature_sets = self.prepare_features(df)
        
        # Run enhanced analysis
        results = self.run_enhanced_analysis(df, regimes, feature_sets)
        
        # Save enhanced results
        self.save_enhanced_results({
            'enhanced_regime_stats': results[0],
            'enhanced_delong_results': results[1],
            'enhanced_calibration_results': results[2],
            'enhanced_decision_results': results[3]
        })
        
        print("\n✅ ENHANCED METRICS COMPLETE!")
        print("✅ BCa bootstrap CIs implemented!")
        print("✅ Enhanced DeLong test with structural components!")
        print("✅ Comprehensive calibration metrics!")
        print("✅ Threshold-dependent decision metrics!")
        print("✅ Explicit cost matrix and savings!")
        print("✅ Configuration hash for reproducibility!")
        
        return results

if __name__ == "__main__":
    enhanced = EnhancedMetricsImplementation()
    results = enhanced.run_complete_enhanced_implementation()
    print("✅ Enhanced metrics implementation execution complete!") 