#!/usr/bin/env python3
"""
Robust Statistical Analysis for Lending Club Sentiment Analysis
==============================================================
Addresses key methodological issues: proper statistical testing, confidence intervals,
multiple comparison correction, and comprehensive evaluation metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from scipy import stats
from scipy.stats import bootstrap
import warnings
warnings.filterwarnings('ignore')

class RobustStatisticalAnalysis:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.results = {}
        
    def delong_test(self, y_true, y_pred1, y_pred2):
        """
        DeLong test for comparing ROC AUCs
        Returns: statistic, p-value
        """
        from scipy import stats
        
        # Calculate ROC curves
        fpr1, tpr1, _ = roc_curve(y_true, y_pred1)
        fpr2, tpr2, _ = roc_curve(y_true, y_pred2)
        
        # Calculate AUCs
        auc1 = roc_auc_score(y_true, y_pred1)
        auc2 = roc_auc_score(y_true, y_pred2)
        
        # DeLong test implementation
        n1 = sum(y_true == 0)
        n2 = sum(y_true == 1)
        
        # Calculate variance
        var = (auc1 * (1 - auc1) + (n2 - 1) * (auc1 / 2 - auc1**2) + 
               (n1 - 1) * (2 * auc1**2 / (1 + auc1) - auc1**2)) / (n1 * n2)
        
        # Test statistic
        z = (auc1 - auc2) / np.sqrt(var)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return z, p_value
    
    def bootstrap_confidence_interval(self, y_true, y_pred, metric_func, n_bootstrap=1000, confidence=0.95):
        """
        Calculate bootstrap confidence interval for any metric
        """
        def bootstrap_metric(data):
            indices = np.random.choice(len(data[0]), len(data[0]), replace=True)
            return metric_func(data[0][indices], data[1][indices])
        
        data = (y_true, y_pred)
        bootstrap_results = [bootstrap_metric(data) for _ in range(n_bootstrap)]
        
        lower = np.percentile(bootstrap_results, (1 - confidence) / 2 * 100)
        upper = np.percentile(bootstrap_results, (1 + confidence) / 2 * 100)
        
        return lower, upper, np.mean(bootstrap_results)
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate comprehensive evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        
        # KS statistic
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        ks_stat = np.max(tpr - fpr)
        metrics['ks_statistic'] = ks_stat
        
        # Brier score
        metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
        
        # Calibration metrics (manual implementation for older sklearn versions)
        def manual_calibration_curve(y_true, y_pred_proba, n_bins=10):
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            fraction_of_positives = []
            mean_predicted_value = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                mask = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                if mask.sum() > 0:
                    fraction_of_positives.append(y_true[mask].mean())
                    mean_predicted_value.append(y_pred_proba[mask].mean())
                else:
                    fraction_of_positives.append(0)
                    mean_predicted_value.append((bin_lower + bin_upper) / 2)
            
            return np.array(fraction_of_positives), np.array(mean_predicted_value)
        
        fraction_of_positives, mean_predicted_value = manual_calibration_curve(y_true, y_pred_proba, n_bins=10)
        metrics['calibration_error'] = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        # Lift metrics
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        top_10_percent = int(0.1 * len(y_true))
        top_10_indices = sorted_indices[:top_10_percent]
        lift_at_10 = np.mean(y_true[top_10_indices]) / np.mean(y_true)
        metrics['lift_at_10'] = lift_at_10
        
        return metrics
    
    def permutation_test(self, y_true, y_pred_traditional, y_pred_sentiment, n_permutations=1000):
        """
        Permutation test to assess if sentiment features provide real signal
        """
        original_diff = roc_auc_score(y_true, y_pred_sentiment) - roc_auc_score(y_true, y_pred_traditional)
        
        permuted_diffs = []
        for _ in range(n_permutations):
            # Shuffle sentiment predictions
            shuffled_sentiment = np.random.permutation(y_pred_sentiment)
            diff = roc_auc_score(y_true, shuffled_sentiment) - roc_auc_score(y_true, y_pred_traditional)
            permuted_diffs.append(diff)
        
        # Calculate p-value
        p_value = np.mean(np.array(permuted_diffs) >= original_diff)
        
        return original_diff, p_value, permuted_diffs
    
    def multiple_comparison_correction(self, p_values, method='fdr_bh'):
        """
        Apply multiple comparison correction
        """
        from statsmodels.stats.multitest import multipletests
        
        if method == 'fdr_bh':
            rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        elif method == 'bonferroni':
            rejected, p_corrected, _, _ = multipletests(p_values, method='bonferroni')
        else:
            raise ValueError("Method must be 'fdr_bh' or 'bonferroni'")
        
        return rejected, p_corrected
    
    def temporal_split_evaluation(self, X, y, dates, test_size=0.2):
        """
        Temporal split evaluation to simulate production conditions
        """
        # Sort by dates
        sorted_indices = np.argsort(dates)
        X_sorted = X.iloc[sorted_indices]
        y_sorted = y.iloc[sorted_indices]
        
        # Split temporally
        split_point = int(len(X_sorted) * (1 - test_size))
        X_train = X_sorted.iloc[:split_point]
        X_test = X_sorted.iloc[split_point:]
        y_train = y_sorted.iloc[:split_point]
        y_test = y_sorted.iloc[split_point:]
        
        return X_train, X_test, y_train, y_test
    
    def feature_importance_analysis(self, model, feature_names, X, y):
        """
        Analyze feature importance and potential redundancy
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return None
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Calculate correlation matrix for sentiment features
        sentiment_features = [f for f in feature_names if 'sentiment' in f.lower()]
        if sentiment_features:
            sentiment_corr = X[sentiment_features].corr()
        else:
            sentiment_corr = None
        
        return importance_df, sentiment_corr
    
    def calibration_analysis(self, y_true, y_pred_proba, model_name):
        """
        Comprehensive calibration analysis
        """
        # Calculate calibration curve (manual implementation)
        def manual_calibration_curve(y_true, y_pred_proba, n_bins=10):
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            fraction_of_positives = []
            mean_predicted_value = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                mask = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                if mask.sum() > 0:
                    fraction_of_positives.append(y_true[mask].mean())
                    mean_predicted_value.append(y_pred_proba[mask].mean())
                else:
                    fraction_of_positives.append(0)
                    mean_predicted_value.append((bin_lower + bin_upper) / 2)
            
            return np.array(fraction_of_positives), np.array(mean_predicted_value)
        
        fraction_of_positives, mean_predicted_value = manual_calibration_curve(y_true, y_pred_proba, n_bins=10)
        
        # Fit isotonic calibration
        from sklearn.isotonic import IsotonicRegression
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(y_pred_proba, y_true)
        y_calibrated = iso_reg.predict(y_pred_proba)
        
        # Calculate calibrated metrics
        calibrated_brier = brier_score_loss(y_true, y_calibrated)
        
        # Create calibration plot
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'{model_name} (Original)')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Plot - {model_name}')
        plt.legend()
        plt.grid(True)
        
        # Plot calibrated probabilities
        plt.subplot(1, 2, 2)
        plt.hist(y_pred_proba[y_true == 0], bins=20, alpha=0.5, label='Non-default', density=True)
        plt.hist(y_pred_proba[y_true == 1], bins=20, alpha=0.5, label='Default', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title(f'Probability Distribution - {model_name}')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'calibration_analysis_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'original_brier': brier_score_loss(y_true, y_pred_proba),
            'calibrated_brier': calibrated_brier,
            'calibration_error': np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        }
    
    def generate_robust_results_table(self, results_dict):
        """
        Generate comprehensive results table with proper statistical reporting
        """
        table_data = []
        
        for model_name, model_results in results_dict.items():
            for variant, metrics in model_results.items():
                row = {
                    'Model': model_name,
                    'Variant': variant,
                    'AUC': f"{metrics['auc']:.4f}",
                    'AUC_CI': f"({metrics['auc_ci_lower']:.4f}, {metrics['auc_ci_upper']:.4f})",
                    'PR-AUC': f"{metrics['pr_auc']:.4f}",
                    'KS': f"{metrics['ks_statistic']:.4f}",
                    'Brier': f"{metrics['brier_score']:.4f}",
                    'Lift@10%': f"{metrics['lift_at_10']:.2f}",
                    'Calibration_Error': f"{metrics['calibration_error']:.4f}"
                }
                
                if 'delta_auc' in metrics:
                    row['ΔAUC'] = f"{metrics['delta_auc']:.4f}"
                    row['p_value'] = f"{metrics['p_value']:.4f}"
                    row['p_adj'] = f"{metrics['p_adj']:.4f}"
                
                table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def run_comprehensive_evaluation(self, X_traditional, X_sentiment, X_hybrid, y, sample_info=None):
        """
        Run comprehensive evaluation addressing all methodological issues
        """
        print("COMPREHENSIVE STATISTICAL EVALUATION")
        print("=" * 60)
        
        # 1. Dataset transparency
        print("1. DATASET TRANSPARENCY")
        print("-" * 30)
        default_rate = np.mean(y)
        print(f"Default rate: {default_rate:.3f}")
        if default_rate > 0.3:
            print("WARNING: High default rate suggests possible sampling bias")
            print("Recommendation: Disclose sampling methodology and implications")
        
        if sample_info:
            print(f"Sample size: {len(y):,}")
            print(f"Time period: {sample_info.get('time_period', 'Not specified')}")
        
        # 2. Comprehensive metrics calculation
        print("\n2. COMPREHENSIVE METRICS CALCULATION")
        print("-" * 30)
        
        models = {
            'RandomForest': RandomForestClassifier(random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state),
            'LogisticRegression': LogisticRegression(random_state=self.random_state)
        }
        
        all_results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            model_results = {}
            
            # Train and evaluate each variant
            for variant_name, X in [('Traditional', X_traditional), 
                                   ('Sentiment', X_sentiment), 
                                   ('Hybrid', X_hybrid)]:
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state, stratify=y
                )
                
                # Train model
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate comprehensive metrics
                metrics = self.calculate_comprehensive_metrics(y_test, 
                                                             model.predict(X_test), 
                                                             y_pred_proba)
                
                # Bootstrap confidence intervals
                auc_ci_lower, auc_ci_upper, _ = self.bootstrap_confidence_interval(
                    y_test, y_pred_proba, roc_auc_score
                )
                metrics['auc_ci_lower'] = auc_ci_lower
                metrics['auc_ci_upper'] = auc_ci_upper
                
                model_results[variant_name] = metrics
                
                print(f"  {variant_name}: AUC = {metrics['auc']:.4f} (95% CI: {auc_ci_lower:.4f}-{auc_ci_upper:.4f})")
            
            all_results[model_name] = model_results
        
        # 3. Statistical testing with multiple comparison correction
        print("\n3. STATISTICAL TESTING")
        print("-" * 30)
        
        p_values = []
        comparisons = []
        
        for model_name, model_results in all_results.items():
            # Traditional vs Sentiment
            trad_auc = model_results['Traditional']['auc']
            sent_auc = model_results['Sentiment']['auc']
            delta_auc = sent_auc - trad_auc
            
            # DeLong test
            _, p_value = self.delong_test(y_test, 
                                         model_results['Traditional']['predictions'],
                                         model_results['Sentiment']['predictions'])
            
            p_values.append(p_value)
            comparisons.append(f"{model_name}_Trad_vs_Sent")
            
            # Traditional vs Hybrid
            hybrid_auc = model_results['Hybrid']['auc']
            delta_auc_hybrid = hybrid_auc - trad_auc
            
            _, p_value_hybrid = self.delong_test(y_test,
                                                model_results['Traditional']['predictions'],
                                                model_results['Hybrid']['predictions'])
            
            p_values.append(p_value_hybrid)
            comparisons.append(f"{model_name}_Trad_vs_Hybrid")
        
        # Multiple comparison correction
        rejected, p_corrected = self.multiple_comparison_correction(p_values)
        
        print("Multiple comparison correction (Benjamini-Hochberg):")
        for i, (comp, p_orig, p_adj, is_rejected) in enumerate(zip(comparisons, p_values, p_corrected, rejected)):
            significance = "***" if is_rejected else ""
            print(f"  {comp}: p = {p_orig:.4f}, p_adj = {p_adj:.4f} {significance}")
        
        # 4. Permutation test for sentiment signal
        print("\n4. PERMUTATION TEST FOR SENTIMENT SIGNAL")
        print("-" * 30)
        
        # Use RandomForest results for permutation test
        rf_results = all_results['RandomForest']
        original_diff, p_perm, _ = self.permutation_test(
            y_test, 
            rf_results['Traditional']['predictions'],
            rf_results['Sentiment']['predictions']
        )
        
        print(f"Original AUC difference: {original_diff:.4f}")
        print(f"Permutation test p-value: {p_perm:.4f}")
        print(f"Sentiment provides {'significant' if p_perm < 0.05 else 'non-significant'} signal")
        
        # 5. Generate comprehensive results table
        print("\n5. COMPREHENSIVE RESULTS TABLE")
        print("-" * 30)
        
        results_table = self.generate_robust_results_table(all_results)
        print(results_table.to_string(index=False))
        
        # Save results
        results_table.to_csv('comprehensive_statistical_results.csv', index=False)
        
        # 6. Generate recommendations
        print("\n6. METHODOLOGICAL RECOMMENDATIONS")
        print("-" * 30)
        
        recommendations = []
        
        # Check for high default rate
        if default_rate > 0.3:
            recommendations.append("Disclose sampling methodology and implications for external validity")
        
        # Check for inconsistent results
        inconsistent_models = []
        for model_name, results in all_results.items():
            trad_auc = results['Traditional']['auc']
            sent_auc = results['Sentiment']['auc']
            hybrid_auc = results['Hybrid']['auc']
            
            if hybrid_auc < sent_auc:
                inconsistent_models.append(model_name)
        
        if inconsistent_models:
            recommendations.append(f"Investigate feature crowding in {', '.join(inconsistent_models)}")
        
        # Check statistical significance
        significant_comparisons = sum(rejected)
        if significant_comparisons < len(p_values) * 0.5:
            recommendations.append("Moderate statistical significance - consider larger sample size")
        
        for rec in recommendations:
            print(f"• {rec}")
        
        return all_results, results_table, recommendations

if __name__ == "__main__":
    # Example usage
    analyzer = RobustStatisticalAnalysis()
    print("Robust Statistical Analysis Module Ready")
    print("Use run_comprehensive_evaluation() for full analysis") 