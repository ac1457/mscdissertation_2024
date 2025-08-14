#!/usr/bin/env python3
"""
Permutation and Lift Validation Module - Lending Club Sentiment Analysis
=======================================================================
Addresses critical missing analyses: permutation tests, lift analysis, leakage checks, and practical significance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PermutationAndLiftValidation:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def check_for_leakage(self, df):
        """
        Check for potential data leakage - post-outcome or target-derived features
        """
        print("CHECKING FOR DATA LEAKAGE")
        print("=" * 40)
        
        # List of features that could indicate leakage
        potential_leakage_features = [
            'loan_status', 'default', 'target', 'outcome', 'result',
            'payment_status', 'charge_off_date', 'last_payment_date',
            'next_payment_date', 'days_past_due', 'collection_status'
        ]
        
        # Check for target-derived features
        leakage_found = []
        for feature in potential_leakage_features:
            if feature in df.columns:
                leakage_found.append(feature)
                print(f"⚠️  POTENTIAL LEAKAGE: {feature} found in dataset")
        
        # Check for features that might be post-origination
        post_origination_indicators = [
            'payment_', 'charge_off', 'collection', 'late_fee',
            'recovery', 'settlement', 'write_off'
        ]
        
        for indicator in post_origination_indicators:
            matching_features = [col for col in df.columns if indicator in col.lower()]
            if matching_features:
                print(f"⚠️  POST-ORIGINATION FEATURES: {matching_features}")
                leakage_found.extend(matching_features)
        
        if not leakage_found:
            print("✅ NO OBVIOUS LEAKAGE DETECTED")
            print("✅ All features appear to be pre-origination")
        else:
            print(f"❌ POTENTIAL LEAKAGE FEATURES: {leakage_found}")
        
        return leakage_found
    
    def permutation_test(self, y_true, y_pred_traditional, y_pred_sentiment, n_permutations=1000):
        """
        Perform permutation test to validate sentiment signal is not random
        """
        print(f"\nPERFORMING PERMUTATION TEST")
        print("=" * 35)
        
        # Calculate observed difference
        auc_traditional = roc_auc_score(y_true, y_pred_traditional)
        auc_sentiment = roc_auc_score(y_true, y_pred_sentiment)
        observed_diff = auc_sentiment - auc_traditional
        
        print(f"Observed AUC difference: {observed_diff:.6f}")
        print(f"Traditional AUC: {auc_traditional:.6f}")
        print(f"Sentiment AUC: {auc_sentiment:.6f}")
        
        # Permutation test
        permuted_diffs = []
        for i in range(n_permutations):
            # Shuffle sentiment predictions
            y_pred_sentiment_permuted = np.random.permutation(y_pred_sentiment)
            
            # Calculate permuted AUC
            auc_sentiment_permuted = roc_auc_score(y_true, y_pred_sentiment_permuted)
            permuted_diff = auc_sentiment_permuted - auc_traditional
            permuted_diffs.append(permuted_diff)
        
        # Calculate p-value
        p_value = np.mean(np.array(permuted_diffs) >= observed_diff)
        
        print(f"Permutation p-value: {p_value:.6f}")
        print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")
        
        return observed_diff, p_value, permuted_diffs
    
    def calculate_lift_metrics(self, y_true, y_pred_proba, decile=10):
        """
        Calculate lift metrics for business value assessment
        """
        # Sort by predicted probability
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        y_true_sorted = y_true.iloc[sorted_indices] if hasattr(y_true, 'iloc') else y_true[sorted_indices]
        
        # Calculate decile boundaries
        n_samples = len(y_true)
        decile_size = n_samples // decile
        
        # Calculate lift for each decile
        lift_values = []
        default_rates = []
        
        for i in range(decile):
            start_idx = i * decile_size
            end_idx = (i + 1) * decile_size if i < decile - 1 else n_samples
            
            decile_true = y_true_sorted[start_idx:end_idx]
            decile_default_rate = np.mean(decile_true)
            overall_default_rate = np.mean(y_true)
            
            lift = decile_default_rate / overall_default_rate if overall_default_rate > 0 else 0
            lift_values.append(lift)
            default_rates.append(decile_default_rate)
        
        # Focus on top decile (highest risk)
        top_decile_lift = lift_values[0]
        top_decile_default_rate = default_rates[0]
        
        return {
            'top_decile_lift': top_decile_lift,
            'top_decile_default_rate': top_decile_default_rate,
            'overall_default_rate': np.mean(y_true),
            'lift_values': lift_values,
            'default_rates': default_rates
        }
    
    def calculate_brier_score(self, y_true, y_pred_proba):
        """
        Calculate Brier score for calibration assessment
        """
        return brier_score_loss(y_true, y_pred_proba)
    
    def univariate_auc_analysis(self, df, y):
        """
        Compare univariate AUC of strongest tabular feature vs sentiment
        """
        print(f"\nUNIVARIATE AUC ANALYSIS")
        print("=" * 30)
        
        # Exclude target and non-numeric columns
        exclude_cols = ['target', 'loan_status', 'sentiment']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Calculate univariate AUC for each feature
        univariate_aucs = {}
        for col in feature_cols:
            try:
                auc = roc_auc_score(y, df[col])
                univariate_aucs[col] = auc
            except:
                continue
        
        # Sort by AUC
        sorted_features = sorted(univariate_aucs.items(), key=lambda x: x[1], reverse=True)
        
        print("Top 10 strongest univariate features:")
        for i, (feature, auc) in enumerate(sorted_features[:10]):
            print(f"  {i+1:2d}. {feature}: {auc:.4f}")
        
        # Compare with sentiment features
        sentiment_features = ['sentiment_score', 'sentiment_confidence', 'sentiment_numeric']
        print(f"\nSentiment feature univariate AUCs:")
        for feature in sentiment_features:
            if feature in df.columns:
                auc = roc_auc_score(y, df[feature])
                print(f"  {feature}: {auc:.4f}")
        
        return sorted_features
    
    def train_models_and_validate(self, df):
        """
        Train models and perform comprehensive validation
        """
        print("COMPREHENSIVE VALIDATION ANALYSIS")
        print("=" * 45)
        
        # Step 1: Check for leakage
        leakage_features = self.check_for_leakage(df)
        
        # Step 2: Prepare data
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
        
        # Train models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=self.random_state),
            'LogisticRegression': LogisticRegression(random_state=self.random_state)
        }
        
        validation_results = []
        
        for model_name, model in models.items():
            print(f"\n{model_name} VALIDATION:")
            print("-" * 25)
            
            # Traditional model
            model.fit(X_train_trad, y_train)
            y_pred_trad = model.predict_proba(X_test_trad)[:, 1]
            auc_trad = roc_auc_score(y_test, y_pred_trad)
            brier_trad = self.calculate_brier_score(y_test, y_pred_trad)
            lift_trad = self.calculate_lift_metrics(y_test, y_pred_trad)
            
            # Sentiment model
            model.fit(X_train_sent, y_train)
            y_pred_sent = model.predict_proba(X_test_sent)[:, 1]
            auc_sent = roc_auc_score(y_test, y_pred_sent)
            brier_sent = self.calculate_brier_score(y_test, y_pred_sent)
            lift_sent = self.calculate_lift_metrics(y_test, y_pred_sent)
            
            # Hybrid model
            model.fit(X_train_hyb, y_train)
            y_pred_hyb = model.predict_proba(X_test_hyb)[:, 1]
            auc_hyb = roc_auc_score(y_test, y_pred_hyb)
            brier_hyb = self.calculate_brier_score(y_test, y_pred_hyb)
            lift_hyb = self.calculate_lift_metrics(y_test, y_pred_hyb)
            
            # Permutation test for sentiment
            perm_diff, perm_p, _ = self.permutation_test(y_test, y_pred_trad, y_pred_sent)
            
            # Store results
            validation_results.append({
                'Model': model_name,
                'Variant': 'Traditional',
                'AUC': auc_trad,
                'Brier': brier_trad,
                'Top_Decile_Lift': lift_trad['top_decile_lift'],
                'Top_Decile_Default_Rate': lift_trad['top_decile_default_rate']
            })
            
            validation_results.append({
                'Model': model_name,
                'Variant': 'Sentiment',
                'AUC': auc_sent,
                'Brier': brier_sent,
                'Top_Decile_Lift': lift_sent['top_decile_lift'],
                'Top_Decile_Default_Rate': lift_sent['top_decile_default_rate'],
                'Permutation_p': perm_p
            })
            
            validation_results.append({
                'Model': model_name,
                'Variant': 'Hybrid',
                'AUC': auc_hyb,
                'Brier': brier_hyb,
                'Top_Decile_Lift': lift_hyb['top_decile_lift'],
                'Top_Decile_Default_Rate': lift_hyb['top_decile_default_rate']
            })
            
            print(f"  Traditional: AUC={auc_trad:.4f}, Brier={brier_trad:.4f}, Lift@10%={lift_trad['top_decile_lift']:.2f}")
            print(f"  Sentiment: AUC={auc_sent:.4f}, Brier={brier_sent:.4f}, Lift@10%={lift_sent['top_decile_lift']:.2f}, Perm_p={perm_p:.4f}")
            print(f"  Hybrid: AUC={auc_hyb:.4f}, Brier={brier_hyb:.4f}, Lift@10%={lift_hyb['top_decile_lift']:.2f}")
        
        # Step 3: Univariate analysis
        univariate_results = self.univariate_auc_analysis(df, y)
        
        return validation_results, univariate_results, leakage_features
    
    def generate_comprehensive_report(self, validation_results, univariate_results, leakage_features):
        """
        Generate comprehensive validation report
        """
        print(f"\nGENERATING COMPREHENSIVE VALIDATION REPORT")
        print("=" * 50)
        
        # Create results DataFrame
        results_df = pd.DataFrame(validation_results)
        
        # Add improvement calculations
        for model in results_df['Model'].unique():
            model_results = results_df[results_df['Model'] == model]
            trad_row = model_results[model_results['Variant'] == 'Traditional'].iloc[0]
            
            for idx, row in model_results.iterrows():
                if row['Variant'] != 'Traditional':
                    results_df.loc[idx, 'AUC_Improvement'] = row['AUC'] - trad_row['AUC']
                    results_df.loc[idx, 'Brier_Improvement'] = trad_row['Brier'] - row['Brier']
                    results_df.loc[idx, 'Lift_Improvement'] = row['Top_Decile_Lift'] - trad_row['Top_Decile_Lift']
        
        # Fill NaN values
        results_df['AUC_Improvement'] = results_df['AUC_Improvement'].fillna(0)
        results_df['Brier_Improvement'] = results_df['Brier_Improvement'].fillna(0)
        results_df['Lift_Improvement'] = results_df['Lift_Improvement'].fillna(0)
        
        # Save comprehensive results
        results_df.to_csv('comprehensive_validation_results.csv', index=False)
        
        # Generate summary
        print(f"\nCOMPREHENSIVE VALIDATION SUMMARY:")
        print("-" * 40)
        
        print(f"LEAKAGE CHECK:")
        if not leakage_features:
            print("  ✅ NO LEAKAGE DETECTED - All features pre-origination")
        else:
            print(f"  ⚠️  POTENTIAL LEAKAGE: {leakage_features}")
        
        print(f"\nPERMUTATION TESTS:")
        for _, row in results_df.iterrows():
            if row['Variant'] == 'Sentiment' and 'Permutation_p' in row:
                print(f"  {row['Model']} Sentiment: p={row['Permutation_p']:.4f} {'(Significant)' if row['Permutation_p'] < 0.05 else '(Not Significant)'}")
        
        print(f"\nLIFT ANALYSIS (Top Decile):")
        for _, row in results_df.iterrows():
            print(f"  {row['Model']} {row['Variant']}: Lift={row['Top_Decile_Lift']:.2f}, Default Rate={row['Top_Decile_Default_Rate']:.3f}")
        
        print(f"\nBRIER SCORE IMPROVEMENTS:")
        for _, row in results_df.iterrows():
            if row['Variant'] != 'Traditional':
                print(f"  {row['Model']} {row['Variant']}: Brier improvement={row['Brier_Improvement']:.4f}")
        
        print(f"\nTOP UNIVARIATE FEATURES:")
        for i, (feature, auc) in enumerate(univariate_results[:5]):
            print(f"  {i+1}. {feature}: AUC={auc:.4f}")
        
        return results_df
    
    def run_complete_validation(self):
        """
        Run complete validation analysis
        """
        print("COMPREHENSIVE VALIDATION - CRITICAL ANALYSIS")
        print("=" * 55)
        
        # Load data
        df = pd.read_csv('comprehensive_results/data/comprehensive_dataset.csv')
        df['target'] = df['loan_status'].astype(int)
        
        # Run comprehensive validation
        validation_results, univariate_results, leakage_features = self.train_models_and_validate(df)
        
        # Generate report
        results_df = self.generate_comprehensive_report(validation_results, univariate_results, leakage_features)
        
        print(f"\n" + "=" * 55)
        print("COMPREHENSIVE VALIDATION COMPLETE")
        print("=" * 55)
        
        return results_df

if __name__ == "__main__":
    validation = PermutationAndLiftValidation()
    results = validation.run_complete_validation()
    
    if results is not None:
        print(f"\n✅ COMPREHENSIVE VALIDATION COMPLETED SUCCESSFULLY")
        print("Check 'comprehensive_validation_results.csv' for detailed results.")
    else:
        print(f"\n❌ COMPREHENSIVE VALIDATION FAILED")
        print("Please check the error messages above.") 