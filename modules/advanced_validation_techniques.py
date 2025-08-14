#!/usr/bin/env python3
"""
Advanced Validation Techniques for Credit Risk Modeling
Includes cross-domain validation, causal inference, fairness analysis, and sophisticated statistical testing
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AdvancedValidator:
    """Advanced validation techniques for credit risk models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        
    def comprehensive_validation(self, X, y, models, loan_data=None):
        """Run comprehensive validation suite"""
        print("Running Advanced Validation Suite")
        print("="*50)
        
        # 1. Multiple Validation Techniques
        validation_results = self._multiple_validation_techniques(X, y, models)
        
        # 2. Statistical Significance Testing
        significance_results = self._statistical_significance_testing(X, y, models)
        
        # 3. Cross-Domain Validation
        domain_results = self._cross_domain_validation(X, y, models, loan_data)
        
        # 4. Fairness Analysis
        fairness_results = self._fairness_analysis(X, y, models, loan_data)
        
        # 5. Robustness Testing
        robustness_results = self._robustness_testing(X, y, models)
        
        # Compile all results
        self.results = {
            'validation': validation_results,
            'significance': significance_results,
            'domain': domain_results,
            'fairness': fairness_results,
            'robustness': robustness_results
        }
        
        return self.results
    
    def _multiple_validation_techniques(self, X, y, models):
        """Apply multiple validation techniques"""
        print("1. Multiple Validation Techniques")
        print("-" * 30)
        
        results = {}
        
        # 1.1 Stratified K-Fold Cross-Validation
        print("  Running Stratified K-Fold CV...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for name, model in models.items():
            cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
            results[f'{name}_stratified_cv'] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores
            }
            print(f"    {name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 1.2 Time Series Split (if temporal data available)
        print("  Running Time Series Split...")
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in models.items():
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc')
            results[f'{name}_timeseries_cv'] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores
            }
            print(f"    {name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 1.3 Repeated Cross-Validation
        print("  Running Repeated CV...")
        repeated_scores = {}
        for name, model in models.items():
            all_scores = []
            for i in range(10):  # 10 repetitions
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
                all_scores.extend(cv_scores)
            
            repeated_scores[name] = all_scores
            results[f'{name}_repeated_cv'] = {
                'mean': np.mean(all_scores),
                'std': np.std(all_scores),
                'scores': all_scores
            }
            print(f"    {name}: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
        
        return results
    
    def _statistical_significance_testing(self, X, y, models):
        """Perform comprehensive statistical significance testing"""
        print("\n2. Statistical Significance Testing")
        print("-" * 30)
        
        results = {}
        
        # Get model predictions
        model_predictions = {}
        for name, model in models.items():
            # Use cross-validation to get predictions
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            model_predictions[name] = cv_scores
        
        # Compare each pair of models
        model_names = list(models.keys())
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                
                print(f"  Comparing {model1} vs {model2}:")
                
                # Paired t-test
                t_stat, p_value = ttest_rel(
                    model_predictions[model1], 
                    model_predictions[model2]
                )
                
                # Wilcoxon signed-rank test (non-parametric)
                try:
                    w_stat, w_p_value = wilcoxon(
                        model_predictions[model1], 
                        model_predictions[model2]
                    )
                except ValueError:
                    # Handle case where differences are all zero
                    w_stat, w_p_value = 0.0, 1.0
                
                # Effect size (Cohen's d)
                diff = np.array(model_predictions[model1]) - np.array(model_predictions[model2])
                cohens_d = np.mean(diff) / np.std(diff, ddof=1)
                
                # Confidence interval
                ci_lower, ci_upper = np.percentile(diff, [2.5, 97.5])
                
                results[f'{model1}_vs_{model2}'] = {
                    't_statistic': t_stat,
                    't_p_value': p_value,
                    'wilcoxon_statistic': w_stat,
                    'wilcoxon_p_value': w_p_value,
                    'cohens_d': cohens_d,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'significant_t': p_value < 0.05,
                    'significant_wilcoxon': w_p_value < 0.05
                }
                
                print(f"    T-test: t={t_stat:.4f}, p={p_value:.4f}, significant={p_value < 0.05}")
                print(f"    Wilcoxon: W={w_stat:.4f}, p={w_p_value:.4f}, significant={w_p_value < 0.05}")
                print(f"    Effect size (Cohen's d): {cohens_d:.4f}")
                print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return results
    
    def _cross_domain_validation(self, X, y, models, loan_data):
        """Cross-domain validation across different segments"""
        print("\n3. Cross-Domain Validation")
        print("-" * 30)
        
        results = {}
        
        if loan_data is None:
            print("  No loan data provided for domain validation")
            return results
        
        # 3.1 Geographic Domain Validation
        if 'addr_state' in loan_data.columns:
            print("  Geographic domain validation...")
            states = loan_data['addr_state'].value_counts()
            major_states = states[states > 50].index[:5]  # Top 5 states with >50 loans
            
            for state in major_states:
                state_mask = loan_data['addr_state'] == state
                if state_mask.sum() > 20:  # Need sufficient samples
                    X_state = X[state_mask]
                    y_state = y[state_mask]
                    
                    for name, model in models.items():
                        cv_score = cross_val_score(model, X_state, y_state, cv=3, scoring='roc_auc').mean()
                        results[f'{name}_state_{state}'] = cv_score
                        print(f"    {name} in {state}: {cv_score:.4f}")
        
        # 3.2 Loan Amount Domain Validation
        print("  Loan amount domain validation...")
        loan_amount_bins = pd.cut(loan_data['loan_amnt'], bins=3, labels=['small', 'medium', 'large'])
        
        for bin_name in ['small', 'medium', 'large']:
            bin_mask = loan_amount_bins == bin_name
            if bin_mask.sum() > 20:
                X_bin = X[bin_mask]
                y_bin = y[bin_mask]
                
                for name, model in models.items():
                    cv_score = cross_val_score(model, X_bin, y_bin, cv=3, scoring='roc_auc').mean()
                    results[f'{name}_loan_{bin_name}'] = cv_score
                    print(f"    {name} for {bin_name} loans: {cv_score:.4f}")
        
        # 3.3 Income Domain Validation
        if 'annual_inc' in loan_data.columns:
            print("  Income domain validation...")
            income_bins = pd.cut(loan_data['annual_inc'], bins=3, labels=['low', 'medium', 'high'])
            
            for bin_name in ['low', 'medium', 'high']:
                bin_mask = income_bins == bin_name
                if bin_mask.sum() > 20:
                    X_bin = X[bin_mask]
                    y_bin = y[bin_mask]
                    
                    for name, model in models.items():
                        cv_score = cross_val_score(model, X_bin, y_bin, cv=3, scoring='roc_auc').mean()
                        results[f'{name}_income_{bin_name}'] = cv_score
                        print(f"    {name} for {bin_name} income: {cv_score:.4f}")
        
        return results
    
    def _fairness_analysis(self, X, y, models, loan_data):
        """Analyze model fairness across different demographic groups"""
        print("\n4. Fairness Analysis")
        print("-" * 30)
        
        results = {}
        
        if loan_data is None:
            print("  No loan data provided for fairness analysis")
            return results
        
        # 4.1 Gender Fairness (if available)
        if 'gender' in loan_data.columns:
            print("  Gender fairness analysis...")
            for gender in loan_data['gender'].unique():
                if pd.notna(gender) and (loan_data['gender'] == gender).sum() > 20:
                    gender_mask = loan_data['gender'] == gender
                    X_gender = X[gender_mask]
                    y_gender = y[gender_mask]
                    
                    for name, model in models.items():
                        cv_score = cross_val_score(model, X_gender, y_gender, cv=3, scoring='roc_auc').mean()
                        results[f'{name}_gender_{gender}'] = cv_score
                        print(f"    {name} for {gender}: {cv_score:.4f}")
        
        # 4.2 Age Fairness (if available)
        if 'age' in loan_data.columns:
            print("  Age fairness analysis...")
            age_bins = pd.cut(loan_data['age'], bins=3, labels=['young', 'middle', 'senior'])
            
            for bin_name in ['young', 'middle', 'senior']:
                bin_mask = age_bins == bin_name
                if bin_mask.sum() > 20:
                    X_bin = X[bin_mask]
                    y_bin = y[bin_mask]
                    
                    for name, model in models.items():
                        cv_score = cross_val_score(model, X_bin, y_bin, cv=3, scoring='roc_auc').mean()
                        results[f'{name}_age_{bin_name}'] = cv_score
                        print(f"    {name} for {bin_name} age: {cv_score:.4f}")
        
        # 4.3 Employment Fairness
        if 'emp_length' in loan_data.columns:
            print("  Employment length fairness analysis...")
            emp_categories = ['new', 'experienced', 'veteran']
            emp_masks = {
                'new': loan_data['emp_length'].str.contains(r'< 1|1 year', na=False),
                'experienced': loan_data['emp_length'].str.contains(r'[2-5] years', na=False),
                'veteran': loan_data['emp_length'].str.contains(r'[6-9] years|10\+', na=False)
            }
            
            for category, mask in emp_masks.items():
                if mask.sum() > 20:
                    X_cat = X[mask]
                    y_cat = y[mask]
                    
                    for name, model in models.items():
                        cv_score = cross_val_score(model, X_cat, y_cat, cv=3, scoring='roc_auc').mean()
                        results[f'{name}_emp_{category}'] = cv_score
                        print(f"    {name} for {category} employees: {cv_score:.4f}")
        
        return results
    
    def _robustness_testing(self, X, y, models):
        """Test model robustness to data perturbations"""
        print("\n5. Robustness Testing")
        print("-" * 30)
        
        results = {}
        
        # 5.1 Noise Robustness
        print("  Testing noise robustness...")
        noise_levels = [0.01, 0.05, 0.10]  # 1%, 5%, 10% noise
        
        for noise_level in noise_levels:
            X_noisy = X.copy()
            # Add Gaussian noise to numerical features
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            noise = np.random.normal(0, noise_level, size=X_noisy[numerical_cols].shape)
            X_noisy[numerical_cols] = X_noisy[numerical_cols] + noise
            
            for name, model in models.items():
                cv_score = cross_val_score(model, X_noisy, y, cv=3, scoring='roc_auc').mean()
                results[f'{name}_noise_{noise_level}'] = cv_score
                print(f"    {name} with {noise_level*100}% noise: {cv_score:.4f}")
        
        # 5.2 Missing Data Robustness
        print("  Testing missing data robustness...")
        missing_rates = [0.05, 0.10, 0.20]  # 5%, 10%, 20% missing
        
        for missing_rate in missing_rates:
            X_missing = X.copy()
            # Randomly set values to NaN
            mask = np.random.random(X_missing.shape) < missing_rate
            X_missing[mask] = np.nan
            
            # Fill missing values with median
            X_missing = X_missing.fillna(X_missing.median())
            
            for name, model in models.items():
                cv_score = cross_val_score(model, X_missing, y, cv=3, scoring='roc_auc').mean()
                results[f'{name}_missing_{missing_rate}'] = cv_score
                print(f"    {name} with {missing_rate*100}% missing: {cv_score:.4f}")
        
        # 5.3 Feature Subset Robustness
        print("  Testing feature subset robustness...")
        feature_subsets = [0.5, 0.7, 0.9]  # Use 50%, 70%, 90% of features
        
        for subset_ratio in feature_subsets:
            n_features = int(X.shape[1] * subset_ratio)
            selected_features = np.random.choice(X.columns, n_features, replace=False)
            X_subset = X[selected_features]
            
            for name, model in models.items():
                cv_score = cross_val_score(model, X_subset, y, cv=3, scoring='roc_auc').mean()
                results[f'{name}_subset_{subset_ratio}'] = cv_score
                print(f"    {name} with {subset_ratio*100}% features: {cv_score:.4f}")
        
        return results
    
    def generate_validation_report(self, save_path=None):
        """Generate comprehensive validation report"""
        if not self.results:
            print("No validation results available. Run comprehensive_validation first.")
            return
        
        print("\n" + "="*60)
        print("COMPREHENSIVE VALIDATION REPORT")
        print("="*60)
        
        # 1. Summary Statistics
        print("\n1. VALIDATION SUMMARY")
        print("-" * 30)
        
        for validation_type, results in self.results.items():
            print(f"\n{validation_type.upper()} RESULTS:")
            for key, value in results.items():
                if isinstance(value, dict) and 'mean' in value:
                    print(f"  {key}: {value['mean']:.4f} ± {value['std']:.4f}")
                elif isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
        # 2. Statistical Significance Summary
        print("\n2. STATISTICAL SIGNIFICANCE SUMMARY")
        print("-" * 30)
        
        if 'significance' in self.results:
            for comparison, result in self.results['significance'].items():
                print(f"\n{comparison}:")
                print(f"  T-test significant: {result['significant_t']}")
                print(f"  Wilcoxon significant: {result['significant_wilcoxon']}")
                print(f"  Effect size: {result['cohens_d']:.4f}")
        
        # 3. Domain Performance Summary
        print("\n3. DOMAIN PERFORMANCE SUMMARY")
        print("-" * 30)
        
        if 'domain' in self.results:
            domain_results = self.results['domain']
            if domain_results:
                # Group by model
                models = set()
                for key in domain_results.keys():
                    model_name = key.split('_')[0]
                    models.add(model_name)
                
                for model in models:
                    print(f"\n{model} domain performance:")
                    model_results = {k: v for k, v in domain_results.items() if k.startswith(model)}
                    for domain, score in model_results.items():
                        domain_name = '_'.join(domain.split('_')[1:])
                        print(f"  {domain_name}: {score:.4f}")
        
        # 4. Fairness Summary
        print("\n4. FAIRNESS ANALYSIS SUMMARY")
        print("-" * 30)
        
        if 'fairness' in self.results:
            fairness_results = self.results['fairness']
            if fairness_results:
                # Similar grouping as domain results
                models = set()
                for key in fairness_results.keys():
                    model_name = key.split('_')[0]
                    models.add(model_name)
                
                for model in models:
                    print(f"\n{model} fairness performance:")
                    model_results = {k: v for k, v in fairness_results.items() if k.startswith(model)}
                    for fairness_group, score in model_results.items():
                        group_name = '_'.join(fairness_group.split('_')[1:])
                        print(f"  {group_name}: {score:.4f}")
        
        # 5. Robustness Summary
        print("\n5. ROBUSTNESS TESTING SUMMARY")
        print("-" * 30)
        
        if 'robustness' in self.results:
            robustness_results = self.results['robustness']
            if robustness_results:
                models = set()
                for key in robustness_results.keys():
                    model_name = key.split('_')[0]
                    models.add(model_name)
                
                for model in models:
                    print(f"\n{model} robustness performance:")
                    model_results = {k: v for k, v in robustness_results.items() if k.startswith(model)}
                    for test, score in model_results.items():
                        test_name = '_'.join(test.split('_')[1:])
                        print(f"  {test_name}: {score:.4f}")
        
        # Save report if path provided
        if save_path:
            self._save_validation_report(save_path)
    
    def _save_validation_report(self, save_path):
        """Save validation report to file"""
        # Convert results to DataFrame for easy saving
        all_results = []
        
        for validation_type, results in self.results.items():
            for key, value in results.items():
                if isinstance(value, dict):
                    if 'mean' in value:
                        all_results.append({
                            'validation_type': validation_type,
                            'metric': key,
                            'mean': value['mean'],
                            'std': value['std']
                        })
                    else:
                        # For significance results
                        all_results.append({
                            'validation_type': validation_type,
                            'metric': key,
                            'value': str(value)
                        })
                else:
                    all_results.append({
                        'validation_type': validation_type,
                        'metric': key,
                        'value': value
                    })
        
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(save_path, index=False)
        print(f"\nValidation report saved to: {save_path}")

def run_advanced_validation_demo():
    """Demonstrate advanced validation techniques"""
    print("Advanced Validation Techniques Demo")
    print("="*50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic loan data
    loan_data = pd.DataFrame({
        'loan_amnt': np.random.uniform(5000, 35000, n_samples),
        'annual_inc': np.random.uniform(30000, 150000, n_samples),
        'dti': np.random.uniform(5, 40, n_samples),
        'int_rate': np.random.uniform(5, 25, n_samples),
        'emp_length': np.random.choice(['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'], n_samples),
        'addr_state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL'], n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'age': np.random.uniform(25, 65, n_samples)
    })
    
    # Create features
    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.normal(0, 1, n_samples),
        'feature_4': np.random.normal(0, 1, n_samples),
        'feature_5': np.random.normal(0, 1, n_samples)
    })
    
    # Create target with some relationship to features
    y = ((X['feature_1'] + X['feature_2'] + np.random.normal(0, 0.5, n_samples)) > 0).astype(int)
    
    # Create models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
    }
    
    # Initialize validator
    validator = AdvancedValidator()
    
    # Run comprehensive validation
    results = validator.comprehensive_validation(X, y, models, loan_data)
    
    # Generate report
    validator.generate_validation_report('advanced_validation_results.csv')
    
    return results

if __name__ == "__main__":
    run_advanced_validation_demo() 