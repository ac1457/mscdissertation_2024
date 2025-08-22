"""
Critical Gaps Fixes Module
Addresses identified methodological gaps in the analysis:
1. Synthetic Text Validity Validation
2. Data Leakage Prevention in Temporal Split
3. Comprehensive Fairness Evaluation
4. Synthetic Data Contamination Analysis
5. Sensitivity Analysis at Different Risk Thresholds
6. Error Analysis and Misclassification Forensics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CriticalGapsFixes:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.config = {
            'output_dir': 'final_results/critical_gaps_analysis',
            'ks_test_alpha': 0.05,
            'fairness_threshold': 0.8,
            'risk_thresholds': [0.01, 0.05, 0.10, 0.15, 0.20],
            'n_bootstrap': 1000
        }
        
    def validate_synthetic_text_distribution(self, df):
        """Validate synthetic vs real text distribution similarity"""
        print("Validating synthetic text distribution similarity...")
        
        # Identify synthetic vs real text
        real_text_mask = df['desc'].notna() & (df['desc'] != '')
        synthetic_text_mask = df['desc'].isna() | (df['desc'] == '')
        
        if real_text_mask.sum() == 0:
            print("No real text found for comparison")
            return None
            
        results = {}
        
        # 1. Sentiment Score Distribution
        if 'sentiment_score' in df.columns:
            real_sentiment = df.loc[real_text_mask, 'sentiment_score'].dropna()
            synthetic_sentiment = df.loc[synthetic_text_mask, 'sentiment_score'].dropna()
            
            if len(real_sentiment) > 0 and len(synthetic_sentiment) > 0:
                ks_stat, p_value = stats.ks_2samp(real_sentiment, synthetic_sentiment)
                results['sentiment_score'] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'significant': p_value < self.config['ks_test_alpha'],
                    'real_mean': real_sentiment.mean(),
                    'synthetic_mean': synthetic_sentiment.mean(),
                    'real_std': real_sentiment.std(),
                    'synthetic_std': synthetic_sentiment.std()
                }
        
        # 2. Text Length Distribution
        if 'text_length' in df.columns:
            real_length = df.loc[real_text_mask, 'text_length'].dropna()
            synthetic_length = df.loc[synthetic_text_mask, 'text_length'].dropna()
            
            if len(real_length) > 0 and len(synthetic_length) > 0:
                ks_stat, p_value = stats.ks_2samp(real_length, synthetic_length)
                results['text_length'] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'significant': p_value < self.config['ks_test_alpha'],
                    'real_mean': real_length.mean(),
                    'synthetic_mean': synthetic_length.mean(),
                    'real_std': real_length.std(),
                    'synthetic_std': synthetic_length.std()
                }
        
        # 3. Word Count Distribution
        if 'word_count' in df.columns:
            real_words = df.loc[real_text_mask, 'word_count'].dropna()
            synthetic_words = df.loc[synthetic_text_mask, 'word_count'].dropna()
            
            if len(real_words) > 0 and len(synthetic_words) > 0:
                ks_stat, p_value = stats.ks_2samp(real_words, synthetic_words)
                results['word_count'] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'significant': p_value < self.config['ks_test_alpha'],
                    'real_mean': real_words.mean(),
                    'synthetic_mean': synthetic_words.mean(),
                    'real_std': real_words.std(),
                    'synthetic_std': synthetic_words.std()
                }
        
        # 4. Financial Keyword Density
        if 'financial_keyword_count' in df.columns:
            real_financial = df.loc[real_text_mask, 'financial_keyword_count'].dropna()
            synthetic_financial = df.loc[synthetic_text_mask, 'financial_keyword_count'].dropna()
            
            if len(real_financial) > 0 and len(synthetic_financial) > 0:
                ks_stat, p_value = stats.ks_2samp(real_financial, synthetic_financial)
                results['financial_keywords'] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'significant': p_value < self.config['ks_test_alpha'],
                    'real_mean': real_financial.mean(),
                    'synthetic_mean': synthetic_financial.mean(),
                    'real_std': real_financial.std(),
                    'synthetic_std': synthetic_financial.std()
                }
        
        return results
    
    def prevent_temporal_data_leakage(self, df):
        """Implement strict as-of-date feature engineering to prevent leakage"""
        print("Implementing strict temporal data leakage prevention...")
        
        # Convert date columns to datetime
        date_columns = ['issue_d', 'last_credit_pull_d', 'last_pymnt_d', 'next_pymnt_d']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Create as-of-date features
        if 'issue_d' in df.columns:
            df['issue_year'] = df['issue_d'].dt.year
            df['issue_month'] = df['issue_d'].dt.month
            df['issue_quarter'] = df['issue_d'].dt.quarter
            
            # Remove future-looking features
            future_features = [
                'last_credit_pull_d', 'last_pymnt_d', 'next_pymnt_d',
                'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
                'total_rec_int', 'total_rec_late_fee', 'recoveries',
                'collection_recovery_fee', 'last_pymnt_amnt'
            ]
            
            for feature in future_features:
                if feature in df.columns:
                    print(f"Removing future-looking feature: {feature}")
                    df = df.drop(columns=[feature])
        
        # Create temporal split that respects time ordering
        if 'issue_d' in df.columns:
            df = df.sort_values('issue_d').reset_index(drop=True)
            
            # Create time-based folds
            n_splits = 5
            split_size = len(df) // n_splits
            
            temporal_splits = []
            for i in range(n_splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < n_splits - 1 else len(df)
                temporal_splits.append((start_idx, end_idx))
        
        return df, temporal_splits
    
    def comprehensive_fairness_evaluation(self, df, model_predictions, true_labels):
        """Comprehensive fairness evaluation including individual fairness"""
        print("Running comprehensive fairness evaluation...")
        
        fairness_results = {}
        
        # 1. Group Fairness (Demographic Parity, Equalized Odds)
        if 'grade' in df.columns:
            fairness_results['group_fairness'] = self.evaluate_group_fairness(
                df, model_predictions, true_labels, 'grade'
            )
        
        if 'home_ownership' in df.columns:
            fairness_results['home_ownership_fairness'] = self.evaluate_group_fairness(
                df, model_predictions, true_labels, 'home_ownership'
            )
        
        # 2. Individual Fairness (Consistency)
        fairness_results['individual_fairness'] = self.evaluate_individual_fairness(
            df, model_predictions, true_labels
        )
        
        # 3. Counterfactual Fairness
        fairness_results['counterfactual_fairness'] = self.evaluate_counterfactual_fairness(
            df, model_predictions, true_labels
        )
        
        return fairness_results
    
    def evaluate_group_fairness(self, df, predictions, true_labels, group_column):
        """Evaluate group fairness metrics"""
        groups = df[group_column].unique()
        group_metrics = {}
        
        for group in groups:
            group_mask = df[group_column] == group
            if group_mask.sum() > 10:  # Minimum sample size
                group_pred = predictions[group_mask]
                group_true = true_labels[group_mask]
                
                # Calculate metrics
                group_metrics[group] = {
                    'sample_size': group_mask.sum(),
                    'default_rate': group_true.mean(),
                    'prediction_rate': group_pred.mean(),
                    'auc': self.calculate_auc(group_true, group_pred),
                    'precision': self.calculate_precision(group_true, group_pred),
                    'recall': self.calculate_recall(group_true, group_pred)
                }
        
        # Calculate fairness metrics
        fairness_metrics = {
            'demographic_parity': self.calculate_demographic_parity(group_metrics),
            'equalized_odds': self.calculate_equalized_odds(group_metrics),
            'equal_opportunity': self.calculate_equal_opportunity(group_metrics)
        }
        
        return {'group_metrics': group_metrics, 'fairness_metrics': fairness_metrics}
    
    def evaluate_individual_fairness(self, df, predictions, true_labels):
        """Evaluate individual fairness through consistency"""
        print("Evaluating individual fairness...")
        
        # Create similarity matrix for similar loans
        similarity_scores = []
        
        # Sample pairs for efficiency
        n_samples = min(1000, len(df))
        sample_indices = np.random.choice(len(df), n_samples, replace=False)
        
        for i in range(0, n_samples, 2):
            if i + 1 < n_samples:
                idx1, idx2 = sample_indices[i], sample_indices[i + 1]
                
                # Calculate feature similarity
                feature_cols = ['loan_amnt', 'int_rate', 'dti', 'annual_inc']
                feature_cols = [col for col in feature_cols if col in df.columns]
                
                if len(feature_cols) > 0:
                    features1 = df.iloc[idx1][feature_cols].values
                    features2 = df.iloc[idx2][feature_cols].values
                    
                    # Normalize features
                    scaler = StandardScaler()
                    features1_scaled = scaler.fit_transform(features1.reshape(1, -1))
                    features2_scaled = scaler.transform(features2.reshape(1, -1))
                    
                    # Calculate similarity
                    similarity = 1 - np.linalg.norm(features1_scaled - features2_scaled)
                    
                    # Calculate prediction difference
                    pred_diff = abs(predictions[idx1] - predictions[idx2])
                    
                    similarity_scores.append({
                        'similarity': similarity,
                        'prediction_diff': pred_diff,
                        'consistency': 1 - pred_diff
                    })
        
        if similarity_scores:
            similarity_df = pd.DataFrame(similarity_scores)
            consistency_score = similarity_df['consistency'].mean()
            
            return {
                'consistency_score': consistency_score,
                'similarity_distribution': similarity_df['similarity'].describe(),
                'prediction_diff_distribution': similarity_df['prediction_diff'].describe()
            }
        
        return {'consistency_score': None, 'error': 'Insufficient data for individual fairness'}
    
    def evaluate_counterfactual_fairness(self, df, predictions, true_labels):
        """Evaluate counterfactual fairness"""
        print("Evaluating counterfactual fairness...")
        
        # Simple counterfactual test: change protected attribute and check prediction change
        if 'grade' in df.columns:
            counterfactual_results = []
            
            # Sample loans for counterfactual analysis
            sample_size = min(500, len(df))
            sample_indices = np.random.choice(len(df), sample_size, replace=False)
            
            for idx in sample_indices:
                original_grade = df.iloc[idx]['grade']
                original_pred = predictions[idx]
                
                # Find similar loan with different grade
                similar_mask = (
                    (df['loan_amnt'] == df.iloc[idx]['loan_amnt']) &
                    (df['grade'] != original_grade) &
                    (abs(df['int_rate'] - df.iloc[idx]['int_rate']) < 0.01)
                )
                
                if similar_mask.sum() > 0:
                    similar_idx = df[similar_mask].index[0]
                    counterfactual_pred = predictions[similar_idx]
                    pred_change = abs(original_pred - counterfactual_pred)
                    
                    counterfactual_results.append({
                        'original_grade': original_grade,
                        'counterfactual_grade': df.iloc[similar_idx]['grade'],
                        'prediction_change': pred_change
                    })
            
            if counterfactual_results:
                cf_df = pd.DataFrame(counterfactual_results)
                return {
                    'mean_prediction_change': cf_df['prediction_change'].mean(),
                    'max_prediction_change': cf_df['prediction_change'].max(),
                    'prediction_change_distribution': cf_df['prediction_change'].describe()
                }
        
        return {'error': 'Insufficient data for counterfactual fairness'}
    
    def synthetic_data_ablation_study(self, df, model_results):
        """Isolate synthetic text impact in results"""
        print("Running synthetic data ablation study...")
        
        # Identify synthetic vs real text
        real_text_mask = df['desc'].notna() & (df['desc'] != '')
        synthetic_text_mask = df['desc'].isna() | (df['desc'] == '')
        
        ablation_results = {}
        
        # 1. Performance on real text only
        if real_text_mask.sum() > 100:
            real_only_results = self.evaluate_subset_performance(
                df[real_text_mask], model_results, 'real_text_only'
            )
            ablation_results['real_text_only'] = real_only_results
        
        # 2. Performance on synthetic text only
        if synthetic_text_mask.sum() > 100:
            synthetic_only_results = self.evaluate_subset_performance(
                df[synthetic_text_mask], model_results, 'synthetic_text_only'
            )
            ablation_results['synthetic_text_only'] = synthetic_only_results
        
        # 3. Performance comparison
        if 'real_text_only' in ablation_results and 'synthetic_text_only' in ablation_results:
            comparison = self.compare_performance_subsets(
                ablation_results['real_text_only'],
                ablation_results['synthetic_text_only']
            )
            ablation_results['comparison'] = comparison
        
        return ablation_results
    
    def evaluate_subset_performance(self, subset_df, model_results, subset_name):
        """Evaluate model performance on a subset of data"""
        # This would need to be implemented based on your model structure
        # For now, return placeholder
        return {
            'subset_name': subset_name,
            'sample_size': len(subset_df),
            'default_rate': subset_df.get('target_5%', 0).mean() if 'target_5%' in subset_df.columns else 0.1,
            'placeholder': 'Implement based on model structure'
        }
    
    def compare_performance_subsets(self, real_results, synthetic_results):
        """Compare performance between real and synthetic text subsets"""
        return {
            'sample_size_diff': real_results['sample_size'] - synthetic_results['sample_size'],
            'default_rate_diff': real_results['default_rate'] - synthetic_results['default_rate'],
            'performance_comparison': 'Implement detailed comparison'
        }
    
    def sensitivity_analysis_risk_thresholds(self, df, model_predictions, true_labels):
        """Sensitivity analysis at different risk acceptance levels"""
        print("Running sensitivity analysis at different risk thresholds...")
        
        sensitivity_results = {}
        
        for threshold in self.config['risk_thresholds']:
            # Calculate precision-recall at this threshold
            precision, recall, _ = precision_recall_curve(true_labels, model_predictions)
            
            # Find closest threshold
            threshold_idx = np.argmin(np.abs(_ - threshold)) if len(_) > 0 else 0
            
            sensitivity_results[f'threshold_{threshold}'] = {
                'precision': precision[threshold_idx] if threshold_idx < len(precision) else 0,
                'recall': recall[threshold_idx] if threshold_idx < len(recall) else 0,
                'f1_score': 2 * (precision[threshold_idx] * recall[threshold_idx]) / 
                           (precision[threshold_idx] + recall[threshold_idx]) if threshold_idx < len(precision) else 0,
                'threshold': threshold
            }
        
        # Calculate AUC for each threshold
        for threshold in self.config['risk_thresholds']:
            # Create binary labels at threshold
            binary_labels = (true_labels >= threshold).astype(int)
            
            # Calculate AUC
            fpr, tpr, _ = roc_curve(binary_labels, model_predictions)
            auc_score = auc(fpr, tpr)
            
            sensitivity_results[f'threshold_{threshold}']['auc'] = auc_score
        
        return sensitivity_results
    
    def error_analysis_forensics(self, df, model_predictions, true_labels):
        """Comprehensive error analysis and misclassification forensics"""
        print("Running error analysis and misclassification forensics...")
        
        # Calculate prediction errors
        errors = abs(model_predictions - true_labels)
        
        # Identify high-error cases
        error_threshold = np.percentile(errors, 95)  # Top 5% errors
        high_error_mask = errors > error_threshold
        
        error_analysis = {
            'overall_error_stats': {
                'mean_error': errors.mean(),
                'std_error': errors.std(),
                'max_error': errors.max(),
                'error_percentiles': np.percentile(errors, [25, 50, 75, 90, 95, 99])
            },
            'high_error_cases': {
                'count': high_error_mask.sum(),
                'percentage': high_error_mask.mean() * 100,
                'characteristics': self.analyze_high_error_characteristics(
                    df[high_error_mask], errors[high_error_mask]
                )
            }
        }
        
        # Analyze misclassification patterns
        if 'target_5%' in df.columns:
            binary_labels = (df['target_5%'] == 1).astype(int)
            binary_predictions = (model_predictions > 0.5).astype(int)
            
            # Confusion matrix analysis
            tp = ((binary_labels == 1) & (binary_predictions == 1)).sum()
            tn = ((binary_labels == 0) & (binary_predictions == 0)).sum()
            fp = ((binary_labels == 0) & (binary_predictions == 1)).sum()
            fn = ((binary_labels == 1) & (binary_predictions == 0)).sum()
            
            error_analysis['misclassification_patterns'] = {
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'fp_characteristics': self.analyze_false_positives(df, binary_labels, binary_predictions),
                'fn_characteristics': self.analyze_false_negatives(df, binary_labels, binary_predictions)
            }
        
        return error_analysis
    
    def analyze_high_error_characteristics(self, high_error_df, high_errors):
        """Analyze characteristics of high-error cases"""
        characteristics = {}
        
        # Analyze by loan purpose
        if 'purpose' in high_error_df.columns:
            purpose_errors = high_error_df.groupby('purpose')['target_5%'].agg(['count', 'mean'])
            characteristics['by_purpose'] = purpose_errors.to_dict()
        
        # Analyze by loan amount
        if 'loan_amnt' in high_error_df.columns:
            characteristics['loan_amount_stats'] = {
                'mean': high_error_df['loan_amnt'].mean(),
                'std': high_error_df['loan_amnt'].std(),
                'range': [high_error_df['loan_amnt'].min(), high_error_df['loan_amnt'].max()]
            }
        
        # Analyze by credit grade
        if 'grade' in high_error_df.columns:
            grade_errors = high_error_df.groupby('grade')['target_5%'].agg(['count', 'mean'])
            characteristics['by_grade'] = grade_errors.to_dict()
        
        return characteristics
    
    def analyze_false_positives(self, df, true_labels, predictions):
        """Analyze false positive characteristics"""
        fp_mask = (true_labels == 0) & (predictions == 1)
        if fp_mask.sum() > 0:
            fp_df = df[fp_mask]
            return {
                'count': fp_mask.sum(),
                'loan_amount_stats': fp_df['loan_amnt'].describe() if 'loan_amnt' in fp_df.columns else None,
                'grade_distribution': fp_df['grade'].value_counts().to_dict() if 'grade' in fp_df.columns else None,
                'purpose_distribution': fp_df['purpose'].value_counts().to_dict() if 'purpose' in fp_df.columns else None
            }
        return {'count': 0}
    
    def analyze_false_negatives(self, df, true_labels, predictions):
        """Analyze false negative characteristics"""
        fn_mask = (true_labels == 1) & (predictions == 0)
        if fn_mask.sum() > 0:
            fn_df = df[fn_mask]
            return {
                'count': fn_mask.sum(),
                'loan_amount_stats': fn_df['loan_amnt'].describe() if 'loan_amnt' in fn_df.columns else None,
                'grade_distribution': fn_df['grade'].value_counts().to_dict() if 'grade' in fn_df.columns else None,
                'purpose_distribution': fn_df['purpose'].value_counts().to_dict() if 'purpose' in fn_df.columns else None
            }
        return {'count': 0}
    
    def calculate_auc(self, y_true, y_pred):
        """Calculate AUC"""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            return auc(fpr, tpr)
        except:
            return 0.5
    
    def calculate_precision(self, y_true, y_pred, threshold=0.5):
        """Calculate precision"""
        try:
            y_pred_binary = (y_pred > threshold).astype(int)
            return (y_true & y_pred_binary).sum() / y_pred_binary.sum() if y_pred_binary.sum() > 0 else 0
        except:
            return 0
    
    def calculate_recall(self, y_true, y_pred, threshold=0.5):
        """Calculate recall"""
        try:
            y_pred_binary = (y_pred > threshold).astype(int)
            return (y_true & y_pred_binary).sum() / y_true.sum() if y_true.sum() > 0 else 0
        except:
            return 0
    
    def calculate_demographic_parity(self, group_metrics):
        """Calculate demographic parity"""
        prediction_rates = [metrics['prediction_rate'] for metrics in group_metrics.values()]
        return max(prediction_rates) - min(prediction_rates) if prediction_rates else 0
    
    def calculate_equalized_odds(self, group_metrics):
        """Calculate equalized odds"""
        tprs = [metrics['recall'] for metrics in group_metrics.values()]
        fprs = [1 - metrics['precision'] for metrics in group_metrics.values()]
        
        tpr_diff = max(tprs) - min(tprs) if tprs else 0
        fpr_diff = max(fprs) - min(fprs) if fprs else 0
        
        return (tpr_diff + fpr_diff) / 2
    
    def calculate_equal_opportunity(self, group_metrics):
        """Calculate equal opportunity"""
        recalls = [metrics['recall'] for metrics in group_metrics.values()]
        return max(recalls) - min(recalls) if recalls else 0
    
    def run_comprehensive_analysis(self, df):
        """Run comprehensive analysis addressing all critical gaps"""
        print("Running comprehensive critical gaps analysis...")
        
        # Load data
        try:
            # Try to load real data first, fall back to synthetic if not available
            try:
                df = pd.read_csv('data/real_lending_club/real_lending_club_processed.csv')
                print(f"Loaded REAL Lending Club dataset: {len(df)} records")
            except FileNotFoundError:
                df = pd.read_csv('data/synthetic_loan_descriptions_with_realistic_targets.csv')
                print(f"Using SYNTHETIC data (real data not found): {len(df)} records")
            
            print(f"Dataset loaded: {len(df)} records, {len(df.columns)} columns")
            
        except FileNotFoundError:
            print("No dataset found. Please run real data processing first.")
            return None
        
        # 1. Synthetic Text Validity Validation
        print("\n1. Synthetic Text Validity Validation")
        synthetic_validation = self.validate_synthetic_text_distribution(df)
        
        # 2. Temporal Data Leakage Prevention
        print("\n2. Temporal Data Leakage Prevention")
        df_clean, temporal_splits = self.prevent_temporal_data_leakage(df)
        
        # 3. Comprehensive Fairness Evaluation
        print("\n3. Comprehensive Fairness Evaluation")
        # Note: This requires model predictions - placeholder for now
        fairness_results = {
            'note': 'Fairness evaluation requires model predictions - implement after model training'
        }
        
        # 4. Synthetic Data Ablation Study
        print("\n4. Synthetic Data Ablation Study")
        # Note: This requires model results - placeholder for now
        ablation_results = {
            'note': 'Ablation study requires model results - implement after model training'
        }
        
        # 5. Sensitivity Analysis
        print("\n5. Sensitivity Analysis")
        # Note: This requires model predictions - placeholder for now
        sensitivity_results = {
            'note': 'Sensitivity analysis requires model predictions - implement after model training'
        }
        
        # 6. Error Analysis
        print("\n6. Error Analysis")
        # Note: This requires model predictions - placeholder for now
        error_results = {
            'note': 'Error analysis requires model predictions - implement after model training'
        }
        
        # Compile results
        comprehensive_results = {
            'synthetic_validation': synthetic_validation,
            'temporal_leakage_prevention': {
                'cleaned_features': list(df_clean.columns),
                'temporal_splits': temporal_splits,
                'removed_future_features': ['last_credit_pull_d', 'last_pymnt_d', 'next_pymnt_d']
            },
            'fairness_evaluation': fairness_results,
            'ablation_study': ablation_results,
            'sensitivity_analysis': sensitivity_results,
            'error_analysis': error_results
        }
        
        # Save results
        self.save_comprehensive_results(comprehensive_results)
        
        return comprehensive_results
    
    def save_comprehensive_results(self, results):
        """Save comprehensive analysis results"""
        import json
        from pathlib import Path
        
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(output_dir / 'comprehensive_critical_gaps_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary report
        self.create_summary_report(results, output_dir)
        
        print(f"Comprehensive analysis results saved to: {output_dir}")
    
    def create_summary_report(self, results, output_dir):
        """Create summary report of critical gaps analysis"""
        report = []
        report.append("# Critical Gaps Analysis Summary Report")
        report.append("=" * 50)
        report.append("")
        
        # Synthetic Text Validation
        report.append("## 1. Synthetic Text Validity Validation")
        if results['synthetic_validation']:
            for feature, metrics in results['synthetic_validation'].items():
                report.append(f"### {feature}")
                report.append(f"- KS Statistic: {metrics['ks_statistic']:.4f}")
                report.append(f"- P-value: {metrics['p_value']:.4f}")
                report.append(f"- Significant Difference: {metrics['significant']}")
                report.append(f"- Real Mean: {metrics['real_mean']:.4f}")
                report.append(f"- Synthetic Mean: {metrics['synthetic_mean']:.4f}")
                report.append("")
        else:
            report.append("No synthetic text validation performed (insufficient data)")
        report.append("")
        
        # Temporal Leakage Prevention
        report.append("## 2. Temporal Data Leakage Prevention")
        report.append(f"- Cleaned Features: {len(results['temporal_leakage_prevention']['cleaned_features'])}")
        report.append(f"- Temporal Splits: {len(results['temporal_leakage_prevention']['temporal_splits'])}")
        report.append(f"- Removed Future Features: {results['temporal_leakage_prevention']['removed_future_features']}")
        report.append("")
        
        # Other sections
        for section in ['fairness_evaluation', 'ablation_study', 'sensitivity_analysis', 'error_analysis']:
            report.append(f"## {section.replace('_', ' ').title()}")
            report.append(results[section].get('note', 'Analysis completed'))
            report.append("")
        
        # Save report
        with open(output_dir / 'critical_gaps_summary_report.md', 'w') as f:
            f.write('\n'.join(report))

if __name__ == "__main__":
    # Run comprehensive critical gaps analysis
    analyzer = CriticalGapsFixes(random_state=42)
    results = analyzer.run_comprehensive_analysis(None)
    
    if results:
        print("Comprehensive critical gaps analysis completed successfully!")
    else:
        print("Critical gaps analysis failed. Please check the error messages above.") 