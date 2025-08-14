#!/usr/bin/env python3
"""
Day 1 Sprint Implementation - Lending Club Sentiment Analysis
===========================================================
Day 1: Implement stratified K-fold (k=5) CV for realistic regimes; 
collect per-fold AUC, PR-AUC. Bootstrap (1k resamples) per regime → CIs.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, 
                           recall_score, f1_score, brier_score_loss)
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

class Day1SprintImplementation:
    """
    Day 1 Sprint: Stratified K-fold CV + Bootstrap CIs
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
    
    def calculate_bootstrap_ci(self, y_true, y_pred_proba, n_bootstrap=1000):
        """
        Calculate bootstrap confidence intervals
        """
        bootstrap_aucs = []
        bootstrap_pr_aucs = []
        bootstrap_briers = []
        bootstrap_precisions = []
        bootstrap_recalls = []
        bootstrap_f1s = []
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        
        for _ in range(n_bootstrap):
            # Bootstrap resample
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            y_boot = y_true[indices]
            pred_boot = y_pred_proba[indices]
            
            # Calculate metrics
            auc = roc_auc_score(y_boot, pred_boot)
            pr_auc = average_precision_score(y_boot, pred_boot)
            brier = brier_score_loss(y_boot, pred_boot)
            
            # For classification metrics, need predictions
            y_pred = (pred_boot >= 0.5).astype(int)
            precision = precision_score(y_boot, y_pred, zero_division=0)
            recall = recall_score(y_boot, y_pred, zero_division=0)
            f1 = f1_score(y_boot, y_pred, zero_division=0)
            
            bootstrap_aucs.append(auc)
            bootstrap_pr_aucs.append(pr_auc)
            bootstrap_briers.append(brier)
            bootstrap_precisions.append(precision)
            bootstrap_recalls.append(recall)
            bootstrap_f1s.append(f1)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_aucs, 2.5)
        ci_upper = np.percentile(bootstrap_aucs, 97.5)
        
        pr_ci_lower = np.percentile(bootstrap_pr_aucs, 2.5)
        pr_ci_upper = np.percentile(bootstrap_pr_aucs, 97.5)
        
        brier_ci_lower = np.percentile(bootstrap_briers, 2.5)
        brier_ci_upper = np.percentile(bootstrap_briers, 97.5)
        
        precision_ci_lower = np.percentile(bootstrap_precisions, 2.5)
        precision_ci_upper = np.percentile(bootstrap_precisions, 97.5)
        
        recall_ci_lower = np.percentile(bootstrap_recalls, 2.5)
        recall_ci_upper = np.percentile(bootstrap_recalls, 97.5)
        
        f1_ci_lower = np.percentile(bootstrap_f1s, 2.5)
        f1_ci_upper = np.percentile(bootstrap_f1s, 97.5)
        
        return {
            'AUC_CI': (ci_lower, ci_upper),
            'PR_AUC_CI': (pr_ci_lower, pr_ci_upper),
            'Brier_CI': (brier_ci_lower, brier_ci_upper),
            'Precision_CI': (precision_ci_lower, precision_ci_upper),
            'Recall_CI': (recall_ci_lower, recall_ci_upper),
            'F1_CI': (f1_ci_lower, f1_ci_upper),
            'AUC_mean': np.mean(bootstrap_aucs),
            'PR_AUC_mean': np.mean(bootstrap_pr_aucs),
            'Brier_mean': np.mean(bootstrap_briers),
            'Precision_mean': np.mean(bootstrap_precisions),
            'Recall_mean': np.mean(bootstrap_recalls),
            'F1_mean': np.mean(bootstrap_f1s)
        }
    
    def perform_stratified_kfold_cv(self, X, y, cv_folds=5):
        """
        Perform stratified K-fold cross-validation with comprehensive metrics
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        models = {
            'RandomForest': RandomForestClassifier(random_state=self.random_state),
            'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=1000)
        }
        
        cv_results = {}
        
        for model_name, model in models.items():
            print(f"  Cross-validating {model_name}...")
            
            # Per-fold metrics
            fold_aucs = []
            fold_pr_aucs = []
            fold_briers = []
            fold_precisions = []
            fold_recalls = []
            fold_f1s = []
            all_predictions = []
            all_true_labels = []
            
            # Detailed fold results
            fold_details = []
            
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                auc = roc_auc_score(y_test, y_pred_proba)
                pr_auc = average_precision_score(y_test, y_pred_proba)
                brier = brier_score_loss(y_test, y_pred_proba)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Store fold results
                fold_aucs.append(auc)
                fold_pr_aucs.append(pr_auc)
                fold_briers.append(brier)
                fold_precisions.append(precision)
                fold_recalls.append(recall)
                fold_f1s.append(f1)
                all_predictions.extend(y_pred_proba)
                all_true_labels.extend(y_test)
                
                # Store detailed fold information
                fold_details.append({
                    'fold': fold + 1,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'train_positives': np.sum(y_train),
                    'test_positives': np.sum(y_test),
                    'auc': auc,
                    'pr_auc': pr_auc,
                    'brier': brier,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
            
            # Bootstrap confidence intervals
            bootstrap_results = self.calculate_bootstrap_ci(all_true_labels, all_predictions)
            
            cv_results[model_name] = {
                'AUC_mean': np.mean(fold_aucs),
                'AUC_std': np.std(fold_aucs),
                'AUC_folds': fold_aucs,
                'AUC_CI': bootstrap_results['AUC_CI'],
                'PR_AUC_mean': np.mean(fold_pr_aucs),
                'PR_AUC_std': np.std(fold_pr_aucs),
                'PR_AUC_folds': fold_pr_aucs,
                'PR_AUC_CI': bootstrap_results['PR_AUC_CI'],
                'Brier_mean': np.mean(fold_briers),
                'Brier_std': np.std(fold_briers),
                'Brier_folds': fold_briers,
                'Brier_CI': bootstrap_results['Brier_CI'],
                'Precision_mean': np.mean(fold_precisions),
                'Precision_std': np.std(fold_precisions),
                'Precision_folds': fold_precisions,
                'Precision_CI': bootstrap_results['Precision_CI'],
                'Recall_mean': np.mean(fold_recalls),
                'Recall_std': np.std(fold_recalls),
                'Recall_folds': fold_recalls,
                'Recall_CI': bootstrap_results['Recall_CI'],
                'F1_mean': np.mean(fold_f1s),
                'F1_std': np.std(fold_f1s),
                'F1_folds': fold_f1s,
                'F1_CI': bootstrap_results['F1_CI'],
                'fold_details': fold_details
            }
        
        return cv_results
    
    def run_day1_analysis(self):
        """
        Run Day 1 analysis: Stratified K-fold CV + Bootstrap CIs
        """
        print("DAY 1 SPRINT: STRATIFIED K-FOLD CV + BOOTSTRAP CIs")
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
        all_results = []
        all_fold_details = []
        
        # Analyze each regime
        for regime_name, regime_data in regimes.items():
            print(f"\n{'='*20} REGIME: {regime_name} {'='*20}")
            
            y = regime_data['y']
            
            # Analyze each feature set
            for feature_set_name, X in feature_sets.items():
                print(f"\nAnalyzing {feature_set_name} features...")
                
                # Perform stratified K-fold CV
                cv_results = self.perform_stratified_kfold_cv(X, y)
                
                # Store results for each model
                for model_name, results in cv_results.items():
                    # Main results
                    result = {
                        'Regime': regime_name,
                        'Feature_Set': feature_set_name,
                        'Model': model_name,
                        'Default_Rate': regime_data['actual_rate'],
                        'Sample_Size': regime_data['n_total'],
                        'Positives': regime_data['n_positives'],
                        'Negatives': regime_data['n_negatives'],
                        'AUC_Mean': results['AUC_mean'],
                        'AUC_Std': results['AUC_std'],
                        'AUC_CI_Lower': results['AUC_CI'][0],
                        'AUC_CI_Upper': results['AUC_CI'][1],
                        'PR_AUC_Mean': results['PR_AUC_mean'],
                        'PR_AUC_Std': results['PR_AUC_std'],
                        'PR_AUC_CI_Lower': results['PR_AUC_CI'][0],
                        'PR_AUC_CI_Upper': results['PR_AUC_CI'][1],
                        'Brier_Mean': results['Brier_mean'],
                        'Brier_Std': results['Brier_std'],
                        'Brier_CI_Lower': results['Brier_CI'][0],
                        'Brier_CI_Upper': results['Brier_CI'][1],
                        'Precision_Mean': results['Precision_mean'],
                        'Precision_Std': results['Precision_std'],
                        'Precision_CI_Lower': results['Precision_CI'][0],
                        'Precision_CI_Upper': results['Precision_CI'][1],
                        'Recall_Mean': results['Recall_mean'],
                        'Recall_Std': results['Recall_std'],
                        'Recall_CI_Lower': results['Recall_CI'][0],
                        'Recall_CI_Upper': results['Recall_CI'][1],
                        'F1_Mean': results['F1_mean'],
                        'F1_Std': results['F1_std'],
                        'F1_CI_Lower': results['F1_CI'][0],
                        'F1_CI_Upper': results['F1_CI'][1],
                        'Feature_Count': X.shape[1],
                        'CV_Folds': 5,
                        'Bootstrap_Resamples': 1000
                    }
                    
                    all_results.append(result)
                    
                    # Fold details
                    for fold_detail in results['fold_details']:
                        fold_detail.update({
                            'Regime': regime_name,
                            'Feature_Set': feature_set_name,
                            'Model': model_name
                        })
                        all_fold_details.append(fold_detail)
        
        return {
            'comprehensive_results': pd.DataFrame(all_results),
            'fold_details': pd.DataFrame(all_fold_details)
        }
    
    def save_day1_results(self, results):
        """
        Save Day 1 results with proper formatting
        """
        print("Saving Day 1 results...")
        
        # Create legend block
        legend_block = {
            'AUC': 'Area Under ROC Curve (0.5 = random, 1.0 = perfect)',
            'PR_AUC': 'Area Under Precision-Recall Curve (better for imbalanced data)',
            'Brier_Score': 'Mean squared error of probability predictions (0 = perfect, 1 = worst)',
            'Precision': 'True Positives / (True Positives + False Positives)',
            'Recall': 'True Positives / (True Positives + False Negatives)',
            'F1_Score': 'Harmonic mean of precision and recall',
            'Bootstrap_CI': '95% confidence interval from 1000 bootstrap resamples',
            'CV_Folds': '5-fold stratified cross-validation',
            'Statistical_Testing': 'DeLong tests and paired bootstrap tests pending for Day 2'
        }
        
        # Save main results with legend
        results['comprehensive_results'].to_csv('final_results/day1_comprehensive_results.csv', index=False)
        
        # Add legend to CSV
        with open('final_results/day1_comprehensive_results.csv', 'r') as f:
            content = f.read()
        
        legend_text = "# DAY 1 SPRINT RESULTS: STRATIFIED K-FOLD CV + BOOTSTRAP CIs\n"
        for key, description in legend_block.items():
            legend_text += f"# {key}: {description}\n"
        legend_text += f"# Generated: {datetime.now().isoformat()}\n"
        legend_text += "# Statistical testing pending - DeLong tests scheduled for Day 2\n"
        legend_text += "# Bootstrap CIs: 95% confidence intervals from 1000 resamples\n"
        legend_text += "# CV Folds: 5-fold stratified cross-validation\n"
        
        with open('final_results/day1_comprehensive_results.csv', 'w') as f:
            f.write(legend_text + content)
        
        # Save fold details
        results['fold_details'].to_csv('final_results/day1_fold_details.csv', index=False)
        
        # Create Day 1 summary
        summary = {
            'day': 1,
            'sprint_item': 'Stratified K-fold CV + Bootstrap CIs',
            'timestamp': datetime.now().isoformat(),
            'regimes_analyzed': list(results['comprehensive_results']['Regime'].unique()),
            'models_analyzed': list(results['comprehensive_results']['Model'].unique()),
            'feature_sets_analyzed': list(results['comprehensive_results']['Feature_Set'].unique()),
            'cv_folds': 5,
            'bootstrap_resamples': 1000,
            'metrics_calculated': ['AUC', 'PR_AUC', 'Brier', 'Precision', 'Recall', 'F1'],
            'confidence_intervals': '95% bootstrap CIs for all metrics',
            'next_day': 'Day 2: DeLong tests and precision/recall at thresholds'
        }
        
        with open('final_results/day1_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✅ Day 1 results saved successfully!")
        print("✅ Bootstrap CIs calculated for all metrics")
        print("✅ Stratified K-fold CV completed")
        print("✅ Ready for Day 2: DeLong tests and threshold analysis")
    
    def run_complete_day1(self):
        """
        Run complete Day 1 implementation
        """
        print("RUNNING DAY 1 SPRINT IMPLEMENTATION")
        print("=" * 50)
        
        # Run Day 1 analysis
        results = self.run_day1_analysis()
        
        if results is None:
            return None
        
        # Save results
        self.save_day1_results(results)
        
        print("\n✅ DAY 1 SPRINT COMPLETE!")
        print("✅ Stratified K-fold CV implemented")
        print("✅ Bootstrap CIs calculated")
        print("✅ Comprehensive metrics collected")
        print("✅ Ready for Day 2: DeLong tests and threshold analysis")
        
        return results

if __name__ == "__main__":
    day1 = Day1SprintImplementation()
    results = day1.run_complete_day1()
    print("✅ Day 1 sprint implementation execution complete!") 