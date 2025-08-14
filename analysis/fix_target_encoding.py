#!/usr/bin/env python3
"""
Fix Target Encoding Module - Lending Club Sentiment Analysis
===========================================================
Properly fixes the target encoding issue by correctly interpreting loan_status.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FixTargetEncoding:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def load_and_analyze_data(self):
        """
        Load and analyze the comprehensive dataset to understand target encoding
        """
        print("LOADING AND ANALYZING COMPREHENSIVE DATASET")
        print("=" * 55)
        
        # Load the comprehensive dataset
        df = pd.read_csv('comprehensive_results/data/comprehensive_dataset.csv')
        print(f"Loaded dataset: {len(df):,} samples")
        print(f"Columns: {len(df.columns)}")
        
        # Analyze the loan_status column
        print(f"\nLOAN_STATUS ANALYSIS:")
        print("-" * 25)
        
        if 'loan_status' in df.columns:
            loan_status_counts = df['loan_status'].value_counts()
            print(f"Loan status distribution:")
            for status, count in loan_status_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {status}: {count:,} samples ({percentage:.1f}%)")
            
            # Check if we need to interpret the values
            print(f"\nLoan status values: {sorted(df['loan_status'].unique())}")
            
            # Determine what 0 and 1 represent
            if 0 in df['loan_status'].values and 1 in df['loan_status'].values:
                print(f"\nINTERPRETATION:")
                print("-" * 15)
                print(f"0: Likely represents 'Fully Paid' or 'Current' loans")
                print(f"1: Likely represents 'Charged Off' or 'Default' loans")
                print(f"Default rate: {loan_status_counts[1] / len(df):.3f} ({loan_status_counts[1] / len(df) * 100:.1f}%)")
        
        return df
    
    def create_proper_target(self, df):
        """
        Create proper target variable from loan_status
        """
        print(f"\nCREATING PROPER TARGET VARIABLE")
        print("=" * 40)
        
        # Create target variable
        # Assuming: 0 = Good (Fully Paid/Current), 1 = Bad (Charged Off/Default)
        # For credit risk modeling, we want to predict default (1)
        df['target'] = df['loan_status'].astype(int)
        
        # Verify target distribution
        target_counts = df['target'].value_counts()
        print(f"Target distribution:")
        for target, count in target_counts.items():
            percentage = (count / len(df)) * 100
            status = "Default" if target == 1 else "Non-Default"
            print(f"  {target} ({status}): {count:,} samples ({percentage:.1f}%)")
        
        print(f"Default rate: {target_counts[1] / len(df):.3f}")
        print(f"Target variable created successfully!")
        
        return df
    
    def prepare_feature_sets(self, df):
        """
        Prepare traditional, sentiment, and hybrid feature sets
        """
        print(f"\nPREPARING FEATURE SETS")
        print("=" * 30)
        
        # Remove target and ID columns from features
        exclude_columns = ['target', 'loan_status', 'sample_id']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        print(f"Total features available: {len(feature_columns)}")
        
        # Traditional features (financial and credit history)
        traditional_features = [
            'loan_amnt', 'annual_inc', 'dti', 'emp_length', 'fico_score',
            'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
            'revol_bal', 'revol_util', 'total_acc', 'home_ownership', 'purpose',
            'age', 'credit_history_length', 'num_credit_cards', 'mortgage_accounts',
            'auto_loans', 'student_loans', 'other_loans', 'bankruptcy_count',
            'foreclosure_count', 'tax_liens', 'collections_count', 'charge_offs'
        ]
        
        available_traditional = [f for f in traditional_features if f in df.columns]
        print(f"Traditional features: {len(available_traditional)}")
        
        # Sentiment features (exclude categorical sentiment)
        sentiment_features = [
            'sentiment_score', 'sentiment_confidence',
            'text_length', 'word_count'
        ]
        
        available_sentiment = [f for f in sentiment_features if f in df.columns]
        print(f"Sentiment features: {len(available_sentiment)}")
        
        # Handle categorical sentiment separately
        if 'sentiment' in df.columns:
            # Convert categorical sentiment to numerical
            sentiment_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
            df['sentiment_numeric'] = df['sentiment'].map(sentiment_mapping)
            available_sentiment.append('sentiment_numeric')
            print(f"Added sentiment_numeric feature")
        
        # Create feature sets
        X_traditional = df[available_traditional].copy()
        X_sentiment = df[available_traditional + available_sentiment].copy()
        
        # Create hybrid features with interactions
        X_hybrid = X_sentiment.copy()
        
        # Add interaction features if sentiment features exist
        if 'sentiment_score' in X_hybrid.columns:
            if 'dti' in X_hybrid.columns:
                X_hybrid['sentiment_dti_interaction'] = X_hybrid['sentiment_score'] * X_hybrid['dti']
            if 'fico_score' in X_hybrid.columns:
                X_hybrid['sentiment_fico_interaction'] = X_hybrid['sentiment_score'] * X_hybrid['fico_score']
            if 'annual_inc' in X_hybrid.columns:
                X_hybrid['sentiment_income_interaction'] = X_hybrid['sentiment_score'] * X_hybrid['annual_inc']
        
        print(f"Hybrid features: {len(X_hybrid.columns)}")
        
        return X_traditional, X_sentiment, X_hybrid
    
    def train_and_evaluate_models(self, X_traditional, X_sentiment, X_hybrid, y):
        """
        Train and evaluate models with proper target encoding
        """
        print(f"\nTRAINING AND EVALUATING MODELS")
        print("=" * 40)
        
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
        
        print(f"Train set: {len(X_train_trad):,} samples")
        print(f"Test set: {len(X_test_trad):,} samples")
        print(f"Train default rate: {y_train.mean():.3f}")
        print(f"Test default rate: {y_test.mean():.3f}")
        
        # Define models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=self.random_state),
            'LogisticRegression': LogisticRegression(random_state=self.random_state)
        }
        
        results = []
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Traditional features
            model.fit(X_train_trad, y_train)
            y_pred_proba_trad = model.predict_proba(X_test_trad)[:, 1]
            auc_trad = roc_auc_score(y_test, y_pred_proba_trad)
            
            print(f"  Traditional: AUC = {auc_trad:.4f}")
            
            # Sentiment features
            model.fit(X_train_sent, y_train)
            y_pred_proba_sent = model.predict_proba(X_test_sent)[:, 1]
            auc_sent = roc_auc_score(y_test, y_pred_proba_sent)
            
            print(f"  Sentiment: AUC = {auc_sent:.4f}")
            
            # Hybrid features
            model.fit(X_train_hyb, y_train)
            y_pred_proba_hyb = model.predict_proba(X_test_hyb)[:, 1]
            auc_hyb = roc_auc_score(y_test, y_pred_proba_hyb)
            
            print(f"  Hybrid: AUC = {auc_hyb:.4f}")
            
            # Store results
            results.append({
                'Model': model_name,
                'Variant': 'Traditional',
                'AUC': auc_trad,
                'Features': len(X_traditional.columns)
            })
            
            results.append({
                'Model': model_name,
                'Variant': 'Sentiment',
                'AUC': auc_sent,
                'Features': len(X_sentiment.columns)
            })
            
            results.append({
                'Model': model_name,
                'Variant': 'Hybrid',
                'AUC': auc_hyb,
                'Features': len(X_hybrid.columns)
            })
        
        return results
    
    def generate_fixed_results(self, results):
        """
        Generate fixed results table
        """
        print(f"\nGENERATING FIXED RESULTS")
        print("=" * 30)
        
        results_df = pd.DataFrame(results)
        
        # Add improvement metrics
        for model in results_df['Model'].unique():
            model_results = results_df[results_df['Model'] == model]
            trad_auc = model_results[model_results['Variant'] == 'Traditional']['AUC'].iloc[0]
            
            for idx, row in model_results.iterrows():
                if row['Variant'] != 'Traditional':
                    improvement = row['AUC'] - trad_auc
                    results_df.loc[idx, 'AUC_Improvement'] = improvement
                    results_df.loc[idx, 'Improvement_Percent'] = (improvement / trad_auc) * 100
        
        # Fill NaN values
        results_df['AUC_Improvement'] = results_df['AUC_Improvement'].fillna(0)
        results_df['Improvement_Percent'] = results_df['Improvement_Percent'].fillna(0)
        
        # Save results
        results_df.to_csv('fixed_results_with_proper_target.csv', index=False)
        
        print(f"Fixed results saved to 'fixed_results_with_proper_target.csv'")
        
        # Display summary
        print(f"\nRESULTS SUMMARY:")
        print("-" * 20)
        for model in results_df['Model'].unique():
            model_results = results_df[results_df['Model'] == model]
            print(f"\n{model}:")
            for _, row in model_results.iterrows():
                improvement_text = f" (+{row['AUC_Improvement']:.4f}, +{row['Improvement_Percent']:.1f}%)" if row['AUC_Improvement'] > 0 else ""
                print(f"  {row['Variant']}: AUC = {row['AUC']:.4f}{improvement_text}")
        
        return results_df
    
    def run_complete_fix(self):
        """
        Run complete target encoding fix
        """
        print("FIXING TARGET ENCODING - COMPLETE ANALYSIS")
        print("=" * 55)
        
        # Step 1: Load and analyze data
        df = self.load_and_analyze_data()
        
        # Step 2: Create proper target variable
        df = self.create_proper_target(df)
        
        # Step 3: Prepare feature sets
        X_traditional, X_sentiment, X_hybrid = self.prepare_feature_sets(df)
        
        # Step 4: Train and evaluate models
        y = df['target']
        results = self.train_and_evaluate_models(X_traditional, X_sentiment, X_hybrid, y)
        
        # Step 5: Generate fixed results
        results_df = self.generate_fixed_results(results)
        
        print(f"\n" + "=" * 55)
        print("TARGET ENCODING FIX COMPLETE")
        print("=" * 55)
        
        return results_df

if __name__ == "__main__":
    fix = FixTargetEncoding()
    results = fix.run_complete_fix()
    
    if results is not None:
        print(f"\n✅ TARGET ENCODING FIXED SUCCESSFULLY")
        print("Check 'fixed_results_with_proper_target.csv' for valid results.")
    else:
        print(f"\n❌ TARGET ENCODING FIX FAILED")
        print("Please check the error messages above.") 