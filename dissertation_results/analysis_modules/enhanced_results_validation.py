#!/usr/bin/env python3
"""
Enhanced Results Validation Module - Lending Club Sentiment Analysis
===================================================================
Adds statistical validation, confidence intervals, and DeLong tests to the fixed results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EnhancedResultsValidation:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def load_fixed_results(self):
        """
        Load the fixed results and validate calculations
        """
        print("LOADING AND VALIDATING FIXED RESULTS")
        print("=" * 50)
        
        # Load the fixed results
        results_df = pd.read_csv('fixed_results_with_proper_target.csv')
        print(f"Loaded results: {len(results_df)} model-variant combinations")
        
        # Validate AUC_Improvement calculations
        print(f"\nVALIDATING AUC_IMPROVEMENT CALCULATIONS:")
        print("-" * 45)
        
        for model in results_df['Model'].unique():
            model_results = results_df[results_df['Model'] == model]
            trad_auc = model_results[model_results['Variant'] == 'Traditional']['AUC'].iloc[0]
            
            print(f"\n{model} (Traditional baseline: {trad_auc:.6f}):")
            
            for _, row in model_results.iterrows():
                if row['Variant'] != 'Traditional':
                    # Recompute improvement
                    auc_improvement = row['AUC'] - trad_auc
                    improvement_percent = (auc_improvement / trad_auc) * 100
                    
                    # Validate against stored values
                    stored_improvement = row['AUC_Improvement']
                    stored_percent = row['Improvement_Percent']
                    
                    print(f"  {row['Variant']}:")
                    print(f"    AUC: {row['AUC']:.6f}")
                    print(f"    Computed improvement: {auc_improvement:.6f} ({improvement_percent:.4f}%)")
                    print(f"    Stored improvement: {stored_improvement:.6f} ({stored_percent:.4f}%)")
                    
                    # Check if they match
                    if abs(auc_improvement - stored_improvement) < 1e-10:
                        print(f"    ✅ Validation: PASS")
                    else:
                        print(f"    ❌ Validation: FAIL")
        
        return results_df
    
    def delong_test(self, y_true, y_pred1, y_pred2):
        """
        Perform DeLong test for comparing two AUCs
        """
        # Calculate AUCs
        auc1 = roc_auc_score(y_true, y_pred1)
        auc2 = roc_auc_score(y_true, y_pred2)
        
        # Calculate ROC curves
        fpr1, tpr1, _ = roc_curve(y_true, y_pred1)
        fpr2, tpr2, _ = roc_curve(y_true, y_pred2)
        
        # DeLong test implementation
        n1 = sum(y_true == 0)
        n2 = sum(y_true == 1)
        
        # Calculate variance
        var = (auc1 * (1 - auc1) + (n2 - 1) * (auc1 / 2 - auc1**2) + 
               (n1 - 1) * (2 * auc1**2 / (1 + auc1) - auc1**2)) / (n1 * n2)
        
        # Test statistic
        z = (auc1 - auc2) / np.sqrt(var)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return z, p_value, auc1, auc2
    
    def bootstrap_confidence_interval(self, y_true, y_pred_proba, metric_func, n_bootstrap=1000, confidence=0.95):
        """
        Calculate bootstrap confidence interval
        """
        n_samples = len(y_true)
        bootstrap_metrics = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
            y_pred_boot = y_pred_proba[indices]
            try:
                metric = metric_func(y_true_boot, y_pred_boot)
                bootstrap_metrics.append(metric)
            except:
                continue
        
        if len(bootstrap_metrics) == 0:
            return (0, 0)
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_metrics, alpha/2 * 100)
        upper = np.percentile(bootstrap_metrics, (1 - alpha/2) * 100)
        
        return (lower, upper)
    
    def train_models_and_get_predictions(self, df):
        """
        Train models and get predictions for statistical testing
        """
        print(f"\nTRAINING MODELS FOR STATISTICAL TESTING")
        print("=" * 45)
        
        # Prepare target
        y = df['target']
        
        # Prepare feature sets
        traditional_features = [
            'loan_amnt', 'annual_inc', 'dti', 'emp_length', 'fico_score',
            'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
            'revol_bal', 'revol_util', 'total_acc', 'home_ownership', 'purpose',
            'age', 'credit_history_length', 'num_credit_cards', 'mortgage_accounts',
            'auto_loans', 'student_loans', 'other_loans', 'bankruptcy_count',
            'foreclosure_count', 'tax_liens', 'collections_count', 'charge_offs'
        ]
        
        sentiment_features = [
            'sentiment_score', 'sentiment_confidence',
            'text_length', 'word_count'
        ]
        
        # Handle categorical sentiment
        if 'sentiment' in df.columns:
            sentiment_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
            df['sentiment_numeric'] = df['sentiment'].map(sentiment_mapping)
            sentiment_features.append('sentiment_numeric')
        
        # Create feature sets
        available_traditional = [f for f in traditional_features if f in df.columns]
        available_sentiment = [f for f in sentiment_features if f in df.columns]
        
        X_traditional = df[available_traditional].copy()
        X_sentiment = df[available_traditional + available_sentiment].copy()
        
        # Create hybrid features
        X_hybrid = X_sentiment.copy()
        if 'sentiment_score' in X_hybrid.columns:
            if 'dti' in X_hybrid.columns:
                X_hybrid['sentiment_dti_interaction'] = X_hybrid['sentiment_score'] * X_hybrid['dti']
            if 'fico_score' in X_hybrid.columns:
                X_hybrid['sentiment_fico_interaction'] = X_hybrid['sentiment_score'] * X_hybrid['fico_score']
            if 'annual_inc' in X_hybrid.columns:
                X_hybrid['sentiment_income_interaction'] = X_hybrid['sentiment_score'] * X_hybrid['annual_inc']
        
        # Split data
        X_train_trad, X_test_trad, y_train, y_test = train_test_split(
            X_traditional, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        X_train_sent, X_test_sent, _, _ = train_test_split(
            X_sentiment, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        X_train_hyb, X_test_hyb, _, _ = train_test_split(
            X_hybrid, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Train models and get predictions
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=self.random_state),
            'LogisticRegression': LogisticRegression(random_state=self.random_state)
        }
        
        predictions = {}
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            # Traditional
            model.fit(X_train_trad, y_train)
            y_pred_trad = model.predict_proba(X_test_trad)[:, 1]
            predictions[f"{model_name}_Traditional"] = y_pred_trad
            
            # Sentiment
            model.fit(X_train_sent, y_train)
            y_pred_sent = model.predict_proba(X_test_sent)[:, 1]
            predictions[f"{model_name}_Sentiment"] = y_pred_sent
            
            # Hybrid
            model.fit(X_train_hyb, y_train)
            y_pred_hyb = model.predict_proba(X_test_hyb)[:, 1]
            predictions[f"{model_name}_Hybrid"] = y_pred_hyb
        
        return predictions, y_test
    
    def perform_statistical_validation(self, predictions, y_test):
        """
        Perform comprehensive statistical validation
        """
        print(f"\nPERFORMING STATISTICAL VALIDATION")
        print("=" * 40)
        
        validation_results = []
        
        for model in ['RandomForest', 'XGBoost', 'LogisticRegression']:
            print(f"\n{model} statistical tests:")
            
            # Get predictions
            y_pred_trad = predictions[f"{model}_Traditional"]
            y_pred_sent = predictions[f"{model}_Sentiment"]
            y_pred_hyb = predictions[f"{model}_Hybrid"]
            
            # Calculate AUCs
            auc_trad = roc_auc_score(y_test, y_pred_trad)
            auc_sent = roc_auc_score(y_test, y_pred_sent)
            auc_hyb = roc_auc_score(y_test, y_pred_hyb)
            
            # Bootstrap confidence intervals
            ci_trad = self.bootstrap_confidence_interval(y_test, y_pred_trad, roc_auc_score)
            ci_sent = self.bootstrap_confidence_interval(y_test, y_pred_sent, roc_auc_score)
            ci_hyb = self.bootstrap_confidence_interval(y_test, y_pred_hyb, roc_auc_score)
            
            # DeLong tests
            z_sent, p_sent, _, _ = self.delong_test(y_test, y_pred_trad, y_pred_sent)
            z_hyb, p_hyb, _, _ = self.delong_test(y_test, y_pred_trad, y_pred_hyb)
            
            print(f"  Traditional vs Sentiment: z={z_sent:.3f}, p={p_sent:.4f}")
            print(f"  Traditional vs Hybrid: z={z_hyb:.3f}, p={p_hyb:.4f}")
            
            # Store results
            validation_results.append({
                'Model': model,
                'Variant': 'Traditional',
                'AUC': auc_trad,
                'AUC_CI_Lower': ci_trad[0],
                'AUC_CI_Upper': ci_trad[1]
            })
            
            validation_results.append({
                'Model': model,
                'Variant': 'Sentiment',
                'AUC': auc_sent,
                'AUC_CI_Lower': ci_sent[0],
                'AUC_CI_Upper': ci_sent[1],
                'DeLong_p_vs_Traditional': p_sent
            })
            
            validation_results.append({
                'Model': model,
                'Variant': 'Hybrid',
                'AUC': auc_hyb,
                'AUC_CI_Lower': ci_hyb[0],
                'AUC_CI_Upper': ci_hyb[1],
                'DeLong_p_vs_Traditional': p_hyb
            })
        
        return validation_results
    
    def generate_enhanced_results_table(self, original_results, validation_results):
        """
        Generate enhanced results table with statistical validation
        """
        print(f"\nGENERATING ENHANCED RESULTS TABLE")
        print("=" * 40)
        
        # Merge original results with validation results
        enhanced_results = []
        
        for _, orig_row in original_results.iterrows():
            # Find corresponding validation result
            val_row = None
            for v_row in validation_results:
                if v_row['Model'] == orig_row['Model'] and v_row['Variant'] == orig_row['Variant']:
                    val_row = v_row
                    break
            
            if val_row:
                enhanced_row = {
                    'Model': orig_row['Model'],
                    'Variant': orig_row['Variant'],
                    'AUC': orig_row['AUC'],
                    'AUC_CI': f"({val_row['AUC_CI_Lower']:.4f}, {val_row['AUC_CI_Upper']:.4f})",
                    'Features': orig_row['Features'],
                    'AUC_Improvement': orig_row['AUC_Improvement'],
                    'Improvement_Percent': orig_row['Improvement_Percent'],
                    'DeLong_p_vs_Traditional': val_row.get('DeLong_p_vs_Traditional', np.nan),
                    'Significance': ''
                }
                
                # Add significance flags
                if 'DeLong_p_vs_Traditional' in val_row and not pd.isna(val_row['DeLong_p_vs_Traditional']):
                    p_val = val_row['DeLong_p_vs_Traditional']
                    if p_val < 0.001:
                        enhanced_row['Significance'] = '***'
                    elif p_val < 0.01:
                        enhanced_row['Significance'] = '**'
                    elif p_val < 0.05:
                        enhanced_row['Significance'] = '*'
                
                enhanced_results.append(enhanced_row)
        
        enhanced_df = pd.DataFrame(enhanced_results)
        
        # Save enhanced results
        enhanced_df.to_csv('enhanced_results_with_validation.csv', index=False)
        
        print(f"Enhanced results saved to 'enhanced_results_with_validation.csv'")
        
        # Display summary
        print(f"\nENHANCED RESULTS SUMMARY:")
        print("-" * 30)
        for model in enhanced_df['Model'].unique():
            model_results = enhanced_df[enhanced_df['Model'] == model]
            print(f"\n{model}:")
            for _, row in model_results.iterrows():
                if row['Variant'] == 'Traditional':
                    print(f"  {row['Variant']}: AUC = {row['AUC']:.4f} {row['AUC_CI']}")
                else:
                    improvement_text = f" (+{row['AUC_Improvement']:.4f}, +{row['Improvement_Percent']:.1f}%)"
                    significance_text = f" {row['Significance']}" if row['Significance'] else ""
                    p_text = f" [p={row['DeLong_p_vs_Traditional']:.4f}]" if not pd.isna(row['DeLong_p_vs_Traditional']) else ""
                    print(f"  {row['Variant']}: AUC = {row['AUC']:.4f} {row['AUC_CI']}{improvement_text}{significance_text}{p_text}")
        
        return enhanced_df
    
    def run_complete_validation(self):
        """
        Run complete enhanced validation
        """
        print("ENHANCED RESULTS VALIDATION - COMPLETE ANALYSIS")
        print("=" * 60)
        
        # Step 1: Load and validate fixed results
        original_results = self.load_fixed_results()
        
        # Step 2: Load data and train models
        df = pd.read_csv('comprehensive_results/data/comprehensive_dataset.csv')
        df['target'] = df['loan_status'].astype(int)
        
        predictions, y_test = self.train_models_and_get_predictions(df)
        
        # Step 3: Perform statistical validation
        validation_results = self.perform_statistical_validation(predictions, y_test)
        
        # Step 4: Generate enhanced results table
        enhanced_results = self.generate_enhanced_results_table(original_results, validation_results)
        
        print(f"\n" + "=" * 60)
        print("ENHANCED VALIDATION COMPLETE")
        print("=" * 60)
        
        return enhanced_results

if __name__ == "__main__":
    validation = EnhancedResultsValidation()
    results = validation.run_complete_validation()
    
    if results is not None:
        print(f"\n✅ ENHANCED VALIDATION COMPLETED SUCCESSFULLY")
        print("Check 'enhanced_results_with_validation.csv' for validated results.")
    else:
        print(f"\n❌ ENHANCED VALIDATION FAILED")
        print("Please check the error messages above.") 