#!/usr/bin/env python3
"""
Fix Leakage and Rebalance Analysis - Lending Club Sentiment Analysis
===================================================================
Addresses critical issues: data leakage, artificial balancing, and industry contextualization.
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

class FixLeakageAndRebalance:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def identify_and_remove_leakage(self, df):
        """
        Identify and remove all leakage features
        """
        print("IDENTIFYING AND REMOVING DATA LEAKAGE")
        print("=" * 50)
        
        # List of leakage features to remove (excluding target variables)
        leakage_features = [
            'charge_offs', 'collections_count',  # Post-origination features
            'payment_status', 'charge_off_date', 'last_payment_date',
            'next_payment_date', 'days_past_due', 'collection_status',
            'recovery', 'settlement', 'write_off', 'late_fee'
        ]
        
        # Find actual leakage features in dataset
        found_leakage = []
        for feature in leakage_features:
            if feature in df.columns:
                found_leakage.append(feature)
                print(f"❌ REMOVING LEAKAGE: {feature}")
        
        # Remove leakage features (but keep target variables)
        df_clean = df.drop(columns=found_leakage, errors='ignore')
        
        print(f"\nRemoved {len(found_leakage)} leakage features")
        print(f"Remaining features: {len(df_clean.columns)}")
        
        # Verify target is still present
        if 'target' not in df_clean.columns and 'loan_status' not in df_clean.columns:
            print("❌ ERROR: Target variable not found! Need to recreate from original data.")
            return None
        
        print(f"✅ Target variable preserved: {'loan_status' if 'loan_status' in df_clean.columns else 'target'}")
        
        return df_clean, found_leakage
    
    def create_realistic_default_rates(self, df, target_col='loan_status'):
        """
        Create realistic default rates by downsampling the majority class
        """
        print(f"\nCREATING REALISTIC DEFAULT RATES")
        print("=" * 40)
        
        # Get original distribution
        original_counts = df[target_col].value_counts()
        print(f"Original distribution:")
        for value, count in original_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {value}: {count:,} samples ({percentage:.1f}%)")
        
        # Define realistic default rates to test
        realistic_rates = [0.05, 0.10, 0.15]  # 5%, 10%, 15%
        
        balanced_datasets = {}
        
        for rate in realistic_rates:
            print(f"\nCreating dataset with {rate*100:.0f}% default rate...")
            
            # Separate classes
            default_samples = df[df[target_col] == 1]
            non_default_samples = df[df[target_col] == 0]
            
            # Calculate target counts
            n_default = int(len(df) * rate)
            n_non_default = len(df) - n_default
            
            # Downsample majority class (non-default)
            if len(non_default_samples) > n_non_default:
                non_default_downsampled = non_default_samples.sample(
                    n=n_non_default, random_state=self.random_state
                )
            else:
                non_default_downsampled = non_default_samples
            
            # Upsample minority class (default) if needed
            if len(default_samples) < n_default:
                # Repeat samples to reach target count
                repeat_factor = int(np.ceil(n_default / len(default_samples)))
                default_upsampled = pd.concat([default_samples] * repeat_factor)
                default_upsampled = default_upsampled.head(n_default)
            else:
                # Downsample if too many
                default_upsampled = default_samples.sample(
                    n=n_default, random_state=self.random_state
                )
            
            # Combine datasets
            balanced_df = pd.concat([non_default_downsampled, default_upsampled])
            balanced_df = balanced_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            
            # Verify distribution
            new_counts = balanced_df[target_col].value_counts()
            new_rate = new_counts[1] / len(balanced_df)
            print(f"  Achieved default rate: {new_rate:.3f} ({new_rate*100:.1f}%)")
            
            balanced_datasets[f"{rate*100:.0f}%_default"] = balanced_df
        
        return balanced_datasets
    
    def prepare_feature_sets(self, df):
        """
        Prepare traditional, sentiment, and hybrid feature sets (no leakage)
        """
        print(f"\nPREPARING FEATURE SETS (NO LEAKAGE)")
        print("=" * 45)
        
        # Traditional features (pre-origination only)
        traditional_features = [
            'loan_amnt', 'annual_inc', 'dti', 'emp_length', 'fico_score',
            'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
            'revol_bal', 'revol_util', 'total_acc', 'home_ownership', 'purpose',
            'age', 'credit_history_length', 'num_credit_cards', 'mortgage_accounts',
            'auto_loans', 'student_loans', 'other_loans', 'bankruptcy_count',
            'foreclosure_count', 'tax_liens'
        ]
        
        # Sentiment features
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
        
        print(f"Traditional features: {len(available_traditional)}")
        print(f"Sentiment features: {len(available_sentiment)}")
        
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
        
        print(f"Hybrid features: {len(X_hybrid.columns)}")
        
        return X_traditional, X_sentiment, X_hybrid
    
    def train_and_evaluate_models(self, X_traditional, X_sentiment, X_hybrid, y, dataset_name):
        """
        Train and evaluate models with proper validation
        """
        print(f"\nTRAINING MODELS - {dataset_name}")
        print("=" * 40)
        
        # Ensure all feature sets have the same number of samples
        assert len(X_traditional) == len(X_sentiment) == len(X_hybrid) == len(y), "Feature sets must have same number of samples"
        
        # Split data once and use same indices for all feature sets
        indices = np.arange(len(y))
        train_indices, test_indices = train_test_split(
            indices, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Split all feature sets using same indices
        X_train_trad = X_traditional.iloc[train_indices]
        X_test_trad = X_traditional.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
        
        X_train_sent = X_sentiment.iloc[train_indices]
        X_test_sent = X_sentiment.iloc[test_indices]
        
        X_train_hyb = X_hybrid.iloc[train_indices]
        X_test_hyb = X_hybrid.iloc[test_indices]
        
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
            brier_trad = brier_score_loss(y_test, y_pred_proba_trad)
            
            print(f"  Traditional: AUC = {auc_trad:.4f}, Brier = {brier_trad:.4f}")
            
            # Sentiment features
            model.fit(X_train_sent, y_train)
            y_pred_proba_sent = model.predict_proba(X_test_sent)[:, 1]
            auc_sent = roc_auc_score(y_test, y_pred_proba_sent)
            brier_sent = brier_score_loss(y_test, y_pred_proba_sent)
            
            print(f"  Sentiment: AUC = {auc_sent:.4f}, Brier = {brier_sent:.4f}")
            
            # Hybrid features
            model.fit(X_train_hyb, y_train)
            y_pred_proba_hyb = model.predict_proba(X_test_hyb)[:, 1]
            auc_hyb = roc_auc_score(y_test, y_pred_proba_hyb)
            brier_hyb = brier_score_loss(y_test, y_pred_proba_hyb)
            
            print(f"  Hybrid: AUC = {auc_hyb:.4f}, Brier = {brier_hyb:.4f}")
            
            # Store results
            results.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Variant': 'Traditional',
                'AUC': auc_trad,
                'Brier': brier_trad,
                'Default_Rate': y_test.mean(),
                'Features': len(X_traditional.columns)
            })
            
            results.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Variant': 'Sentiment',
                'AUC': auc_sent,
                'Brier': brier_sent,
                'Default_Rate': y_test.mean(),
                'Features': len(X_sentiment.columns),
                'AUC_Improvement': auc_sent - auc_trad,
                'Brier_Improvement': brier_trad - brier_sent
            })
            
            results.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Variant': 'Hybrid',
                'AUC': auc_hyb,
                'Brier': brier_hyb,
                'Default_Rate': y_test.mean(),
                'Features': len(X_hybrid.columns),
                'AUC_Improvement': auc_hyb - auc_trad,
                'Brier_Improvement': brier_trad - brier_hyb
            })
        
        return results
    
    def industry_benchmark_comparison(self, results_df):
        """
        Compare results to industry standards
        """
        print(f"\nINDUSTRY BENCHMARK COMPARISON")
        print("=" * 40)
        
        # Industry benchmarks (from literature)
        industry_benchmarks = {
            'Credit_Card_Models': {'AUC': 0.65, 'Description': 'Typical credit card default models'},
            'Personal_Loan_Models': {'AUC': 0.70, 'Description': 'Personal loan default models'},
            'Mortgage_Models': {'AUC': 0.75, 'Description': 'Mortgage default models'},
            'Commercial_Lending': {'AUC': 0.68, 'Description': 'Commercial lending models'}
        }
        
        print("Industry Benchmark AUCs:")
        for benchmark, info in industry_benchmarks.items():
            print(f"  {benchmark}: {info['AUC']:.2f} - {info['Description']}")
        
        # Compare current results
        print(f"\nCurrent Results Comparison:")
        for dataset in results_df['Dataset'].unique():
            dataset_results = results_df[results_df['Dataset'] == dataset]
            best_auc = dataset_results['AUC'].max()
            print(f"\n{dataset}:")
            print(f"  Best AUC: {best_auc:.4f}")
            
            for benchmark, info in industry_benchmarks.items():
                gap = info['AUC'] - best_auc
                percentage_gap = (gap / info['AUC']) * 100
                print(f"  Gap to {benchmark}: {gap:.3f} ({percentage_gap:.1f}%)")
        
        return industry_benchmarks
    
    def generate_fixed_results_report(self, all_results, industry_benchmarks):
        """
        Generate comprehensive report of fixed results
        """
        print(f"\nGENERATING FIXED RESULTS REPORT")
        print("=" * 40)
        
        # Create comprehensive results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Fill NaN values
        results_df['AUC_Improvement'] = results_df['AUC_Improvement'].fillna(0)
        results_df['Brier_Improvement'] = results_df['Brier_Improvement'].fillna(0)
        
        # Save results
        results_df.to_csv('fixed_results_no_leakage.csv', index=False)
        
        # Generate summary by dataset
        print(f"\nFIXED RESULTS SUMMARY:")
        print("-" * 30)
        
        for dataset in results_df['Dataset'].unique():
            dataset_results = results_df[results_df['Dataset'] == dataset]
            print(f"\n{dataset} (Default Rate: {dataset_results['Default_Rate'].iloc[0]:.1%}):")
            
            for model in dataset_results['Model'].unique():
                model_results = dataset_results[dataset_results['Model'] == model]
                print(f"  {model}:")
                
                for _, row in model_results.iterrows():
                    if row['Variant'] == 'Traditional':
                        print(f"    {row['Variant']}: AUC = {row['AUC']:.4f}, Brier = {row['Brier']:.4f}")
                    else:
                        improvement_text = f" (+{row['AUC_Improvement']:.4f})" if row['AUC_Improvement'] > 0 else f" ({row['AUC_Improvement']:.4f})"
                        print(f"    {row['Variant']}: AUC = {row['AUC']:.4f}{improvement_text}, Brier = {row['Brier']:.4f}")
        
        return results_df
    
    def run_complete_fix(self):
        """
        Run complete fix for leakage and rebalancing
        """
        print("FIXING LEAKAGE AND REBALANCING - COMPLETE ANALYSIS")
        print("=" * 60)
        
        # Step 1: Load data
        df = pd.read_csv('comprehensive_results/data/comprehensive_dataset.csv')
        
        # Step 2: Fix leakage
        df_clean, leakage_features = self.identify_and_remove_leakage(df)
        if df_clean is None:
            print("❌ Failed to fix leakage - target variable removed")
            return None
        
        # Step 3: Create realistic default rates
        balanced_datasets = self.create_realistic_default_rates(df_clean)
        
        # Step 4: Prepare feature sets
        X_traditional, X_sentiment, X_hybrid = self.prepare_feature_sets(df_clean)
        
        # Step 5: Train and evaluate on all datasets
        all_results = []
        
        for dataset_name, balanced_df in balanced_datasets.items():
            y = balanced_df['loan_status'] if 'loan_status' in balanced_df.columns else balanced_df['target']
            
            # Prepare feature sets from the balanced dataset
            X_trad_balanced, X_sent_balanced, X_hyb_balanced = self.prepare_feature_sets(balanced_df)
            
            results = self.train_and_evaluate_models(X_trad_balanced, X_sent_balanced, X_hyb_balanced, y, dataset_name)
            all_results.extend(results)
        
        # Step 6: Industry comparison
        results_df = pd.DataFrame(all_results)
        industry_benchmarks = self.industry_benchmark_comparison(results_df)
        
        # Step 7: Generate report
        final_results = self.generate_fixed_results_report(all_results, industry_benchmarks)
        
        print(f"\n" + "=" * 60)
        print("LEAKAGE FIX AND REBALANCING COMPLETE")
        print("=" * 60)
        
        return final_results, industry_benchmarks, leakage_features

if __name__ == "__main__":
    fix = FixLeakageAndRebalance()
    results = fix.run_complete_fix()
    
    if results is not None:
        print(f"\n✅ LEAKAGE FIX AND REBALANCING COMPLETED SUCCESSFULLY")
        print("Check 'fixed_results_no_leakage.csv' for clean results.")
    else:
        print(f"\n❌ LEAKAGE FIX AND REBALANCING FAILED")
        print("Please check the error messages above.") 