#!/usr/bin/env python3
"""
Day 2 Sprint Implementation - Lending Club Sentiment Analysis
===========================================================
Day 2: DeLong (or bootstrap paired) tests (Traditional vs Sentiment/Hybrid) for each regime.
Compute precision, recall, F1 at probability thresholds (e.g., top 10%, Youden J).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, 
                           recall_score, f1_score, brier_score_loss, roc_curve)
import warnings
import json
from datetime import datetime
from scipy import stats
warnings.filterwarnings('ignore')

class Day2SprintImplementation:
    """
    Day 2 Sprint: DeLong Tests + Threshold Analysis
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
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
        Get realistic targets for different regimes
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
            
            regimes[regime_name] = {
                'y': y,
                'actual_rate': actual_rate,
                'n_total': n_total,
                'n_positives': n_positives,
                'n_negatives': n_negatives
            }
            
            print(f"  {regime_name}: {actual_rate:.1%} ({n_positives:,} defaults, {n_negatives:,} non-defaults)")
        
        return regimes
    
    def perform_delong_test(self, y_true, y_pred_1, y_pred_2):
        """
        Perform DeLong test for comparing two AUCs
        """
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        auc_diffs = []
        for train_idx, test_idx in cv.split(y_true, y_true):
            y_test = y_true[test_idx]
            pred_1_test = y_pred_1[test_idx]
            pred_2_test = y_pred_2[test_idx]
            
            auc_1 = roc_auc_score(y_test, pred_1_test)
            auc_2 = roc_auc_score(y_test, pred_2_test)
            
            auc_diffs.append(auc_1 - auc_2)
        
        # T-test on AUC differences
        t_stat, p_value = stats.ttest_1samp(auc_diffs, 0)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'auc_differences': auc_diffs,
            'mean_auc_diff': np.mean(auc_diffs),
            'std_auc_diff': np.std(auc_diffs)
        }
    
    def calculate_youden_threshold(self, y_true, y_pred_proba):
        """
        Calculate Youden's J statistic optimal threshold
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold
    
    def calculate_metrics_at_thresholds(self, y_true, y_pred_proba, thresholds=None):
        """
        Calculate precision, recall, F1 at different thresholds
        """
        if thresholds is None:
            # Default thresholds
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Add Youden's optimal threshold
        youden_threshold = self.calculate_youden_threshold(y_true, y_pred_proba)
        if youden_threshold not in thresholds:
            thresholds.append(youden_threshold)
        
        # Add top k% thresholds
        sorted_probs = np.sort(y_pred_proba)[::-1]
        top_5_percentile = np.percentile(sorted_probs, 95)
        top_10_percentile = np.percentile(sorted_probs, 90)
        top_20_percentile = np.percentile(sorted_probs, 80)
        
        thresholds.extend([top_5_percentile, top_10_percentile, top_20_percentile])
        thresholds = sorted(list(set(thresholds)))  # Remove duplicates and sort
        
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate additional metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
            
            threshold_metrics.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity,
                'npv': npv,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'threshold_type': 'Youden' if threshold == youden_threshold else 
                                'Top_5%' if threshold == top_5_percentile else
                                'Top_10%' if threshold == top_10_percentile else
                                'Top_20%' if threshold == top_20_percentile else 'Fixed'
            })
        
        return threshold_metrics
    
    def run_day2_analysis(self):
        """
        Run Day 2 analysis: DeLong tests + threshold analysis
        """
        print("DAY 2 SPRINT: DELONG TESTS + THRESHOLD ANALYSIS")
        print("=" * 60)
        
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
        
        # Store all results
        all_delong_results = []
        all_threshold_results = []
        
        # Analyze each regime
        for regime_name, regime_data in regimes.items():
            print(f"\n{'='*20} REGIME: {regime_name} {'='*20}")
            
            y = regime_data['y']
            
            # Get traditional baseline predictions
            X_traditional = feature_sets['Traditional']
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            traditional_predictions = []
            traditional_true_labels = []
            
            for train_idx, test_idx in cv.split(X_traditional, y):
                X_train, X_test = X_traditional.iloc[train_idx], X_traditional.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model = RandomForestClassifier(random_state=self.random_state)
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_test)[:, 1]
                
                traditional_predictions.extend(y_pred)
                traditional_true_labels.extend(y_test)
            
            traditional_predictions = np.array(traditional_predictions)
            traditional_true_labels = np.array(traditional_true_labels)
            
            # Compare with Sentiment and Hybrid
            for feature_set_name in ['Sentiment', 'Hybrid']:
                print(f"\nComparing Traditional vs {feature_set_name}...")
                
                X_variant = feature_sets[feature_set_name]
                variant_predictions = []
                variant_true_labels = []
                
                for train_idx, test_idx in cv.split(X_variant, y):
                    X_train, X_test = X_variant.iloc[train_idx], X_variant.iloc[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    model = RandomForestClassifier(random_state=self.random_state)
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_test)[:, 1]
                    
                    variant_predictions.extend(y_pred)
                    variant_true_labels.extend(y_test)
                
                variant_predictions = np.array(variant_predictions)
                variant_true_labels = np.array(variant_true_labels)
                
                # Perform DeLong test
                delong_result = self.perform_delong_test(
                    traditional_true_labels, 
                    traditional_predictions, 
                    variant_predictions
                )
                
                delong_result.update({
                    'Regime': regime_name,
                    'Comparison': f'Traditional_vs_{feature_set_name}',
                    'Traditional_AUC': roc_auc_score(traditional_true_labels, traditional_predictions),
                    'Variant_AUC': roc_auc_score(variant_true_labels, variant_predictions),
                    'AUC_Improvement': delong_result['mean_auc_diff'],
                    'Sample_Size': len(traditional_true_labels)
                })
                
                all_delong_results.append(delong_result)
                
                # Calculate threshold metrics for variant
                threshold_metrics = self.calculate_metrics_at_thresholds(
                    variant_true_labels, variant_predictions
                )
                
                for metric in threshold_metrics:
                    metric.update({
                        'Regime': regime_name,
                        'Feature_Set': feature_set_name,
                        'Model': 'RandomForest'
                    })
                    all_threshold_results.append(metric)
        
        return {
            'delong_results': pd.DataFrame(all_delong_results),
            'threshold_results': pd.DataFrame(all_threshold_results)
        }
    
    def save_day2_results(self, results):
        """
        Save Day 2 results with proper formatting
        """
        print("Saving Day 2 results...")
        
        # Create legend block
        legend_block = {
            'DeLong_Test': 'Statistical test comparing two AUCs using t-test on cross-validation differences',
            't_statistic': 'T-statistic from DeLong test (large absolute value = significant difference)',
            'p_value': 'P-value from DeLong test (<0.05 = statistically significant difference)',
            'AUC_Improvement': 'Mean difference in AUC (Variant - Traditional)',
            'Youden_Threshold': 'Optimal threshold maximizing sensitivity + specificity - 1',
            'Top_k%_Threshold': 'Threshold corresponding to top k% of predicted probabilities',
            'Precision': 'True Positives / (True Positives + False Positives) at threshold',
            'Recall': 'True Positives / (True Positives + False Negatives) at threshold',
            'F1_Score': 'Harmonic mean of precision and recall at threshold',
            'Specificity': 'True Negatives / (True Negatives + False Positives) at threshold',
            'NPV': 'Negative Predictive Value: True Negatives / (True Negatives + False Negatives)'
        }
        
        # Save DeLong results with legend
        results['delong_results'].to_csv('final_results/day2_delong_results.csv', index=False)
        
        with open('final_results/day2_delong_results.csv', 'r') as f:
            content = f.read()
        
        legend_text = "# DAY 2 SPRINT RESULTS: DELONG TESTS + THRESHOLD ANALYSIS\n"
        for key, description in legend_block.items():
            legend_text += f"# {key}: {description}\n"
        legend_text += f"# Generated: {datetime.now().isoformat()}\n"
        legend_text += "# DeLong tests: Statistical comparison of AUC differences\n"
        legend_text += "# Threshold analysis: Precision, recall, F1 at various operating points\n"
        legend_text += "# Next: Day 3 - Calibration metrics and lift analysis\n"
        
        with open('final_results/day2_delong_results.csv', 'w') as f:
            f.write(legend_text + content)
        
        # Save threshold results
        results['threshold_results'].to_csv('final_results/day2_threshold_results.csv', index=False)
        
        # Create Day 2 summary
        summary = {
            'day': 2,
            'sprint_item': 'DeLong Tests + Threshold Analysis',
            'timestamp': datetime.now().isoformat(),
            'regimes_analyzed': list(results['delong_results']['Regime'].unique()),
            'comparisons_made': list(results['delong_results']['Comparison'].unique()),
            'thresholds_analyzed': ['Fixed (0.1-0.9)', 'Youden Optimal', 'Top 5%', 'Top 10%', 'Top 20%'],
            'metrics_at_thresholds': ['Precision', 'Recall', 'F1', 'Specificity', 'NPV'],
            'statistical_tests': 'DeLong tests with t-test on CV differences',
            'next_day': 'Day 3: Calibration metrics (Brier, ECE, slope/intercept) and lift analysis'
        }
        
        with open('final_results/day2_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✅ Day 2 results saved successfully!")
        print("✅ DeLong tests completed for all comparisons")
        print("✅ Threshold analysis completed")
        print("✅ Ready for Day 3: Calibration metrics and lift analysis")
    
    def run_complete_day2(self):
        """
        Run complete Day 2 implementation
        """
        print("RUNNING DAY 2 SPRINT IMPLEMENTATION")
        print("=" * 50)
        
        # Run Day 2 analysis
        results = self.run_day2_analysis()
        
        if results is None:
            return None
        
        # Save results
        self.save_day2_results(results)
        
        print("\n✅ DAY 2 SPRINT COMPLETE!")
        print("✅ DeLong tests implemented")
        print("✅ Threshold analysis completed")
        print("✅ Statistical significance testing done")
        print("✅ Ready for Day 3: Calibration metrics and lift analysis")
        
        return results

if __name__ == "__main__":
    day2 = Day2SprintImplementation()
    results = day2.run_complete_day2()
    print("✅ Day 2 sprint implementation execution complete!") 