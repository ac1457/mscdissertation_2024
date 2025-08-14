#!/usr/bin/env python3
"""
Critical Diagnostics Module - Lending Club Sentiment Analysis
===========================================================
Implements sanity checks and univariate analysis to identify root causes of near-random performance.
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

class CriticalDiagnostics:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def verify_target_integrity(self, df):
        """
        Verify target integrity and check for label misalignment
        """
        print("VERIFYING TARGET INTEGRITY")
        print("=" * 40)
        
        # Check target variable
        if 'default' in df.columns:
            y = df['default']
            target_name = 'default'
        elif 'loan_status' in df.columns:
            y = (df['loan_status'] == 'Charged Off').astype(int)
            target_name = 'loan_status (Charged Off)'
        else:
            print("❌ ERROR: No target variable found")
            return None, None
        
        print(f"Target variable: {target_name}")
        print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
        print(f"Default rate: {y.mean():.4f}")
        print(f"Unique values: {np.unique(y)}")
        
        # Check for single class
        if len(np.unique(y)) < 2:
            print("❌ CRITICAL: Target has only one class - cannot compute AUC")
            return None, None
        
        # Check for reasonable default rate
        if y.mean() < 0.01 or y.mean() > 0.99:
            print("⚠️  WARNING: Unrealistic default rate - possible data issue")
        
        # Check for row ordering consistency
        print(f"\nRow ordering check:")
        print(f"  Total rows: {len(df)}")
        print(f"  Target sum: {y.sum()}")
        print(f"  Target mean: {y.mean():.4f}")
        
        # Check for any missing values in target
        missing_target = y.isnull().sum() if hasattr(y, 'isnull') else 0
        print(f"  Missing target values: {missing_target}")
        
        return df, y
    
    def univariate_separation_analysis(self, df, y):
        """
        Compute univariate AUC for individual features to check separability
        """
        print("\nUNIVARIATE SEPARATION ANALYSIS")
        print("=" * 40)
        
        # Select numerical features for univariate analysis
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and ID columns
        if 'default' in numerical_features:
            numerical_features.remove('default')
        if 'loan_status' in numerical_features:
            numerical_features.remove('loan_status')
        if 'sample_id' in numerical_features:
            numerical_features.remove('sample_id')
        
        print(f"Analyzing {len(numerical_features)} numerical features")
        
        univariate_results = []
        
        for feature in numerical_features:
            try:
                # Handle missing values
                feature_data = df[feature].fillna(df[feature].median())
                
                # Skip if all values are the same
                if feature_data.std() == 0:
                    continue
                
                # Compute AUC
                auc = roc_auc_score(y, feature_data)
                
                # Compute correlation with target
                correlation = np.corrcoef(feature_data, y)[0, 1]
                
                univariate_results.append({
                    'Feature': feature,
                    'AUC': auc,
                    'Correlation': correlation,
                    'Mean': feature_data.mean(),
                    'Std': feature_data.std(),
                    'Missing': df[feature].isnull().sum() if hasattr(df[feature], 'isnull') else 0
                })
                
                print(f"  {feature}: AUC = {auc:.4f}, Corr = {correlation:.4f}")
                
            except Exception as e:
                print(f"  ⚠️  Error with {feature}: {e}")
                continue
        
        # Sort by AUC
        univariate_df = pd.DataFrame(univariate_results)
        univariate_df = univariate_df.sort_values('AUC', ascending=False)
        
        print(f"\nTop 10 features by AUC:")
        for _, row in univariate_df.head(10).iterrows():
            print(f"  {row['Feature']}: AUC = {row['AUC']:.4f}, Corr = {row['Correlation']:.4f}")
        
        print(f"\nBottom 10 features by AUC:")
        for _, row in univariate_df.tail(10).iterrows():
            print(f"  {row['Feature']}: AUC = {row['AUC']:.4f}, Corr = {row['Correlation']:.4f}")
        
        # Check if any features have meaningful separation
        meaningful_features = univariate_df[univariate_df['AUC'] > 0.55]
        print(f"\nFeatures with AUC > 0.55: {len(meaningful_features)}")
        
        if len(meaningful_features) == 0:
            print("❌ CRITICAL: No features show meaningful separation (AUC > 0.55)")
            print("   This suggests a fundamental data issue")
        
        return univariate_df
    
    def check_feature_variance(self, df):
        """
        Check feature variance and identify zero-variance features
        """
        print("\nFEATURE VARIANCE ANALYSIS")
        print("=" * 35)
        
        # Check numerical features
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        variance_report = []
        
        for feature in numerical_features:
            feature_data = df[feature].fillna(df[feature].median())
            variance = feature_data.var()
            std = feature_data.std()
            
            variance_report.append({
                'Feature': feature,
                'Variance': variance,
                'Std': std,
                'Min': feature_data.min(),
                'Max': feature_data.max(),
                'Missing': df[feature].isnull().sum() if hasattr(df[feature], 'isnull') else 0
            })
        
        variance_df = pd.DataFrame(variance_report)
        variance_df = variance_df.sort_values('Variance', ascending=False)
        
        # Identify zero-variance features
        zero_variance = variance_df[variance_df['Variance'] == 0]
        low_variance = variance_df[variance_df['Variance'] < 0.01]
        
        print(f"Zero-variance features: {len(zero_variance)}")
        if len(zero_variance) > 0:
            print("  Features to remove:")
            for _, row in zero_variance.iterrows():
                print(f"    {row['Feature']}")
        
        print(f"Low-variance features (var < 0.01): {len(low_variance)}")
        if len(low_variance) > 0:
            print("  Features to investigate:")
            for _, row in low_variance.iterrows():
                print(f"    {row['Feature']}: var = {row['Variance']:.6f}")
        
        return variance_df
    
    def correlation_analysis(self, df, y):
        """
        Analyze correlations between features and target
        """
        print("\nCORRELATION ANALYSIS")
        print("=" * 25)
        
        # Select numerical features
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target columns
        if 'default' in numerical_features:
            numerical_features.remove('default')
        if 'loan_status' in numerical_features:
            numerical_features.remove('loan_status')
        
        # Compute correlations with target
        correlations = []
        for feature in numerical_features:
            feature_data = df[feature].fillna(df[feature].median())
            corr = np.corrcoef(feature_data, y)[0, 1]
            correlations.append(abs(corr))
        
        # Create correlation DataFrame
        corr_df = pd.DataFrame({
            'Feature': numerical_features,
            'Abs_Correlation': correlations
        }).sort_values('Abs_Correlation', ascending=False)
        
        print(f"Top 10 features by absolute correlation:")
        for _, row in corr_df.head(10).iterrows():
            print(f"  {row['Feature']}: |corr| = {row['Abs_Correlation']:.4f}")
        
        print(f"\nBottom 10 features by absolute correlation:")
        for _, row in corr_df.tail(10).iterrows():
            print(f"  {row['Feature']}: |corr| = {row['Abs_Correlation']:.4f}")
        
        # Check for meaningful correlations
        meaningful_corr = corr_df[corr_df['Abs_Correlation'] > 0.1]
        print(f"\nFeatures with |correlation| > 0.1: {len(meaningful_corr)}")
        
        if len(meaningful_corr) == 0:
            print("❌ CRITICAL: No features show meaningful correlation with target")
        
        return corr_df
    
    def temporal_split_analysis(self, df, y):
        """
        Perform temporal split analysis to check for data leakage
        """
        print("\nTEMPORAL SPLIT ANALYSIS")
        print("=" * 30)
        
        # Check if we have temporal information
        temporal_features = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d']
        available_temporal = [f for f in temporal_features if f in df.columns]
        
        if not available_temporal:
            print("⚠️  No temporal features found - using random split")
            return None
        
        print(f"Available temporal features: {available_temporal}")
        
        # Use the first available temporal feature
        temporal_feature = available_temporal[0]
        print(f"Using temporal feature: {temporal_feature}")
        
        # Convert to datetime if possible
        try:
            df[temporal_feature] = pd.to_datetime(df[temporal_feature], errors='coerce')
            df = df.dropna(subset=[temporal_feature])
            
            # Sort by temporal feature
            df_sorted = df.sort_values(temporal_feature).reset_index(drop=True)
            
            # Create temporal split (80% train, 20% test)
            split_point = int(0.8 * len(df_sorted))
            train_df = df_sorted.iloc[:split_point]
            test_df = df_sorted.iloc[split_point:]
            
            print(f"Temporal split:")
            print(f"  Train: {len(train_df)} samples ({train_df[temporal_feature].min()} to {train_df[temporal_feature].max()})")
            print(f"  Test: {len(test_df)} samples ({test_df[temporal_feature].min()} to {test_df[temporal_feature].max()})")
            print(f"  Train default rate: {train_df['default'].mean():.4f}")
            print(f"  Test default rate: {test_df['default'].mean():.4f}")
            
            return train_df, test_df
            
        except Exception as e:
            print(f"⚠️  Error with temporal split: {e}")
            return None
    
    def permutation_test_analysis(self, df, y):
        """
        Perform permutation tests to validate feature importance
        """
        print("\nPERMUTATION TEST ANALYSIS")
        print("=" * 35)
        
        # Select top features by univariate AUC
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'default' in numerical_features:
            numerical_features.remove('default')
        if 'loan_status' in numerical_features:
            numerical_features.remove('loan_status')
        
        # Compute univariate AUC for all features
        feature_aucs = []
        for feature in numerical_features:
            feature_data = df[feature].fillna(df[feature].median())
            if feature_data.std() > 0:
                auc = roc_auc_score(y, feature_data)
                feature_aucs.append((feature, auc))
        
        # Sort by AUC and take top 5
        feature_aucs.sort(key=lambda x: x[1], reverse=True)
        top_features = feature_aucs[:5]
        
        print(f"Testing top 5 features with permutation tests:")
        
        permutation_results = []
        
        for feature, original_auc in top_features:
            print(f"\n  {feature}: Original AUC = {original_auc:.4f}")
            
            # Perform permutation test
            n_permutations = 1000
            permuted_aucs = []
            
            for _ in range(n_permutations):
                # Shuffle the feature values
                shuffled_feature = np.random.permutation(df[feature].fillna(df[feature].median()))
                permuted_auc = roc_auc_score(y, shuffled_feature)
                permuted_aucs.append(permuted_auc)
            
            # Calculate p-value
            p_value = np.mean(np.array(permuted_aucs) >= original_auc)
            
            print(f"    Permutation p-value: {p_value:.4f}")
            print(f"    Permuted AUC range: {min(permuted_aucs):.4f} - {max(permuted_aucs):.4f}")
            
            permutation_results.append({
                'Feature': feature,
                'Original_AUC': original_auc,
                'Permutation_p': p_value,
                'Significant': p_value < 0.05
            })
        
        return pd.DataFrame(permutation_results)
    
    def generate_diagnostic_report(self, target_integrity, univariate_results, variance_results, correlation_results, permutation_results):
        """
        Generate comprehensive diagnostic report
        """
        print("\nGENERATING DIAGNOSTIC REPORT")
        print("=" * 35)
        
        report = f"""
CRITICAL DIAGNOSTIC REPORT
==========================

TARGET INTEGRITY:
-----------------
Status: {'✅ PASS' if target_integrity else '❌ FAIL'}
Default rate: {target_integrity.get('default_rate', 'N/A') if target_integrity else 'N/A'}

UNIVARIATE SEPARATION:
----------------------
Total features analyzed: {len(univariate_results) if univariate_results is not None else 0}
Features with AUC > 0.55: {len(univariate_results[univariate_results['AUC'] > 0.55]) if univariate_results is not None else 0}
Features with AUC > 0.60: {len(univariate_results[univariate_results['AUC'] > 0.60]) if univariate_results is not None else 0}

Top 5 features by AUC:
{univariate_results.head(5)[['Feature', 'AUC', 'Correlation']].to_string() if univariate_results is not None else 'N/A'}

FEATURE VARIANCE:
-----------------
Zero-variance features: {len(variance_results[variance_results['Variance'] == 0]) if variance_results is not None else 0}
Low-variance features: {len(variance_results[variance_results['Variance'] < 0.01]) if variance_results is not None else 0}

CORRELATION ANALYSIS:
---------------------
Features with |corr| > 0.1: {len(correlation_results[correlation_results['Abs_Correlation'] > 0.1]) if correlation_results is not None else 0}
Features with |corr| > 0.2: {len(correlation_results[correlation_results['Abs_Correlation'] > 0.2]) if correlation_results is not None else 0}

PERMUTATION TESTS:
------------------
Features with significant permutation tests: {len(permutation_results[permutation_results['Significant']]) if permutation_results is not None else 0}

CRITICAL FINDINGS:
------------------
"""
        
        # Add critical findings
        if univariate_results is not None and len(univariate_results[univariate_results['AUC'] > 0.55]) == 0:
            report += "❌ CRITICAL: No features show meaningful separation (AUC > 0.55)\n"
            report += "   This indicates a fundamental data issue\n\n"
        
        if correlation_results is not None and len(correlation_results[correlation_results['Abs_Correlation'] > 0.1]) == 0:
            report += "❌ CRITICAL: No features show meaningful correlation with target\n"
            report += "   This suggests target/feature misalignment\n\n"
        
        if variance_results is not None and len(variance_results[variance_results['Variance'] == 0]) > 0:
            report += "⚠️  WARNING: Zero-variance features detected\n"
            report += "   These should be removed from analysis\n\n"
        
        report += """
RECOMMENDATIONS:
----------------
1. If no features show AUC > 0.55: Investigate target definition and data quality
2. If no features show correlation > 0.1: Check for label/feature misalignment
3. Remove zero-variance features before modeling
4. Consider temporal split to avoid data leakage
5. Implement proper feature engineering for better separability

NEXT STEPS:
-----------
1. Fix target integrity issues if identified
2. Remove problematic features (zero/low variance)
3. Implement proper feature engineering
4. Use temporal split for evaluation
5. Re-run analysis with corrected data
        """
        
        # Save report
        with open('critical_diagnostic_report.txt', 'w') as f:
            f.write(report)
        
        print("Diagnostic report saved to 'critical_diagnostic_report.txt'")
        return report
    
    def run_complete_diagnostics(self):
        """
        Run complete diagnostic analysis
        """
        print("CRITICAL DIAGNOSTICS - COMPLETE ANALYSIS")
        print("=" * 50)
        
        # Load data
        try:
            df = pd.read_csv('comprehensive_results/data/comprehensive_dataset.csv')
            print(f"Loaded dataset: {len(df):,} samples")
        except FileNotFoundError:
            print("Creating synthetic data for demonstration...")
            df = self.create_synthetic_data(10000)
        
        # Step 1: Verify target integrity
        df, y = self.verify_target_integrity(df)
        if df is None:
            print("❌ Target integrity check failed - stopping diagnostics")
            return False
        
        target_integrity = {
            'default_rate': y.mean(),
            'target_distribution': pd.Series(y).value_counts().to_dict(),
            'unique_values': np.unique(y)
        }
        
        # Step 2: Univariate separation analysis
        univariate_results = self.univariate_separation_analysis(df, y)
        
        # Step 3: Feature variance analysis
        variance_results = self.check_feature_variance(df)
        
        # Step 4: Correlation analysis
        correlation_results = self.correlation_analysis(df, y)
        
        # Step 5: Temporal split analysis
        temporal_results = self.temporal_split_analysis(df, y)
        
        # Step 6: Permutation test analysis
        permutation_results = self.permutation_test_analysis(df, y)
        
        # Step 7: Generate diagnostic report
        self.generate_diagnostic_report(target_integrity, univariate_results, variance_results, correlation_results, permutation_results)
        
        print("\n" + "=" * 50)
        print("CRITICAL DIAGNOSTICS COMPLETE")
        print("=" * 50)
        
        return True
    
    def create_synthetic_data(self, n_samples=10000):
        """Create synthetic data for testing"""
        np.random.seed(self.random_state)
        
        data = {
            'loan_amnt': np.random.lognormal(9.6, 0.6, n_samples),
            'annual_inc': np.random.lognormal(11.2, 0.7, n_samples),
            'dti': np.random.gamma(2.5, 7, n_samples),
            'emp_length': np.random.choice([0, 2, 5, 8, 10], n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1]),
            'fico_score': np.random.normal(710, 45, n_samples),
            'delinq_2yrs': np.random.poisson(0.4, n_samples),
            'inq_last_6mths': np.random.poisson(1.1, n_samples),
            'open_acc': np.random.poisson(11, n_samples),
            'pub_rec': np.random.poisson(0.2, n_samples),
            'revol_bal': np.random.lognormal(8.8, 1.1, n_samples),
            'revol_util': np.random.beta(2.2, 2.8, n_samples) * 100,
            'total_acc': np.random.poisson(22, n_samples),
            'home_ownership': np.random.choice([0, 1, 2], n_samples, p=[0.48, 0.42, 0.1]),
            'purpose': np.random.choice(range(6), n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Realistic bounds
        df['loan_amnt'] = np.clip(df['loan_amnt'], 1000, 40000)
        df['annual_inc'] = np.clip(df['annual_inc'], 25000, 300000)
        df['dti'] = np.clip(df['dti'], 0, 45)
        df['fico_score'] = np.clip(df['fico_score'], 620, 850)
        df['revol_util'] = np.clip(df['revol_util'], 0, 100)
        
        # Create realistic target
        df['default'] = np.random.binomial(1, 0.3, n_samples)
        
        return df

if __name__ == "__main__":
    diagnostics = CriticalDiagnostics()
    success = diagnostics.run_complete_diagnostics()
    
    if success:
        print("\n✅ CRITICAL DIAGNOSTICS COMPLETED SUCCESSFULLY")
        print("Check the generated report for detailed analysis.")
    else:
        print("\n❌ CRITICAL DIAGNOSTICS FAILED")
        print("Please check the error messages above.") 