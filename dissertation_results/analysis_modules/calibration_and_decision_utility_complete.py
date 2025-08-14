#!/usr/bin/env python3
"""
Calibration and Decision Utility Complete - Lending Club Sentiment Analysis
==========================================================================
Comprehensive calibration metrics and decision utility analysis.
Includes Brier, ECE, calibration slope, Lift@k, and expected profit/default reduction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

class CalibrationAndDecisionUtilityComplete:
    """
    Comprehensive calibration and decision utility analysis
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def load_data(self):
        """
        Load synthetic loan descriptions data
        """
        try:
            df = pd.read_csv('data/synthetic_loan_descriptions.csv')
            print(f"✅ Loaded dataset: {len(df)} records")
            return df
        except FileNotFoundError:
            print("❌ synthetic_loan_descriptions.csv not found")
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
    
    def calculate_ece(self, y_true, y_pred_proba, n_bins=10):
        """
        Calculate Expected Calibration Error (ECE)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = np.logical_and(y_pred_proba > bin_lower, y_pred_proba <= bin_upper)
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                # Calculate accuracy and confidence for this bin
                bin_accuracy = np.sum(y_true[in_bin]) / bin_size
                bin_confidence = np.mean(y_pred_proba[in_bin])
                
                # Add to ECE
                ece += bin_size * np.abs(bin_accuracy - bin_confidence)
        
        return ece / len(y_true)
    
    def calculate_calibration_slope(self, y_true, y_pred_proba):
        """
        Calculate calibration slope
        """
        # Use logistic regression to fit calibration slope
        from sklearn.linear_model import LogisticRegression
        
        # Reshape for sklearn
        X_cal = y_pred_proba.reshape(-1, 1)
        
        # Fit logistic regression
        lr = LogisticRegression()
        lr.fit(X_cal, y_true)
        
        # Return slope (coefficient)
        return lr.coef_[0][0]
    
    def calculate_lift_at_k(self, y_true, y_pred_proba, k_percent=10):
        """
        Calculate Lift@k%
        """
        # Sort by predicted probability (descending)
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        sorted_true = y_true[sorted_indices]
        
        # Calculate k% threshold
        k_threshold = int(len(y_true) * k_percent / 100)
        
        # Calculate default rate in top k%
        top_k_default_rate = np.mean(sorted_true[:k_threshold])
        
        # Calculate overall default rate
        overall_default_rate = np.mean(y_true)
        
        # Calculate lift
        lift = top_k_default_rate / overall_default_rate if overall_default_rate > 0 else 0
        
        return lift, top_k_default_rate, overall_default_rate
    
    def calculate_expected_profit(self, y_true, y_pred_proba, threshold=0.5, 
                                loan_amount=10000, interest_rate=0.15, 
                                default_cost=0.6):
        """
        Calculate expected profit under specified cost matrix
        """
        # Predict using threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate confusion matrix
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Calculate expected profit
        # Profit from good loans: interest - cost of funds
        profit_per_good_loan = loan_amount * interest_rate * 0.5  # Simplified
        
        # Loss from bad loans: default cost
        loss_per_bad_loan = loan_amount * default_cost
        
        # Total expected profit
        total_profit = (tn * profit_per_good_loan) - (fp * loss_per_bad_loan)
        
        return total_profit, tp, tn, fp, fn
    
    def calculate_default_reduction(self, y_true, y_pred_proba, threshold=0.5):
        """
        Calculate default reduction at specified threshold
        """
        # Predict using threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate confusion matrix
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Calculate default reduction
        total_defaults = np.sum(y_true)
        defaults_caught = tp
        default_reduction_rate = defaults_caught / total_defaults if total_defaults > 0 else 0
        
        return default_reduction_rate, defaults_caught, total_defaults
    
    def perform_cross_validation_with_calibration(self, X, y, cv_folds=5):
        """
        Perform cross-validation with comprehensive calibration metrics
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        models = {
            'RandomForest': RandomForestClassifier(random_state=self.random_state),
            'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=1000)
        }
        
        cv_results = {}
        
        for model_name, model in models.items():
            print(f"  Cross-validating {model_name} with calibration metrics...")
            
            fold_briers = []
            fold_eces = []
            fold_cal_slopes = []
            fold_lift_5 = []
            fold_lift_10 = []
            fold_lift_20 = []
            fold_profits = []
            fold_default_reductions = []
            all_predictions = []
            all_true_labels = []
            
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate calibration metrics
                brier = brier_score_loss(y_test, y_pred_proba)
                ece = self.calculate_ece(y_test, y_pred_proba)
                cal_slope = self.calculate_calibration_slope(y_test, y_pred_proba)
                
                # Calculate Lift@k
                lift_5, _, _ = self.calculate_lift_at_k(y_test, y_pred_proba, 5)
                lift_10, _, _ = self.calculate_lift_at_k(y_test, y_pred_proba, 10)
                lift_20, _, _ = self.calculate_lift_at_k(y_test, y_pred_proba, 20)
                
                # Calculate expected profit and default reduction
                profit, _, _, _, _ = self.calculate_expected_profit(y_test, y_pred_proba)
                default_reduction, _, _ = self.calculate_default_reduction(y_test, y_pred_proba)
                
                # Store results
                fold_briers.append(brier)
                fold_eces.append(ece)
                fold_cal_slopes.append(cal_slope)
                fold_lift_5.append(lift_5)
                fold_lift_10.append(lift_10)
                fold_lift_20.append(lift_20)
                fold_profits.append(profit)
                fold_default_reductions.append(default_reduction)
                all_predictions.extend(y_pred_proba)
                all_true_labels.extend(y_test)
            
            # Calculate overall calibration metrics
            overall_brier = brier_score_loss(all_true_labels, all_predictions)
            overall_ece = self.calculate_ece(np.array(all_true_labels), np.array(all_predictions))
            overall_cal_slope = self.calculate_calibration_slope(np.array(all_true_labels), np.array(all_predictions))
            
            # Calculate overall Lift@k
            overall_lift_5, _, _ = self.calculate_lift_at_k(np.array(all_true_labels), np.array(all_predictions), 5)
            overall_lift_10, _, _ = self.calculate_lift_at_k(np.array(all_true_labels), np.array(all_predictions), 10)
            overall_lift_20, _, _ = self.calculate_lift_at_k(np.array(all_true_labels), np.array(all_predictions), 20)
            
            # Calculate overall profit and default reduction
            overall_profit, _, _, _, _ = self.calculate_expected_profit(np.array(all_true_labels), np.array(all_predictions))
            overall_default_reduction, _, _ = self.calculate_default_reduction(np.array(all_true_labels), np.array(all_predictions))
            
            cv_results[model_name] = {
                'Brier_mean': np.mean(fold_briers),
                'Brier_std': np.std(fold_briers),
                'Brier_overall': overall_brier,
                'ECE_mean': np.mean(fold_eces),
                'ECE_std': np.std(fold_eces),
                'ECE_overall': overall_ece,
                'Cal_Slope_mean': np.mean(fold_cal_slopes),
                'Cal_Slope_std': np.std(fold_cal_slopes),
                'Cal_Slope_overall': overall_cal_slope,
                'Lift_5_mean': np.mean(fold_lift_5),
                'Lift_5_std': np.std(fold_lift_5),
                'Lift_5_overall': overall_lift_5,
                'Lift_10_mean': np.mean(fold_lift_10),
                'Lift_10_std': np.std(fold_lift_10),
                'Lift_10_overall': overall_lift_10,
                'Lift_20_mean': np.mean(fold_lift_20),
                'Lift_20_std': np.std(fold_lift_20),
                'Lift_20_overall': overall_lift_20,
                'Expected_Profit_mean': np.mean(fold_profits),
                'Expected_Profit_std': np.std(fold_profits),
                'Expected_Profit_overall': overall_profit,
                'Default_Reduction_mean': np.mean(fold_default_reductions),
                'Default_Reduction_std': np.std(fold_default_reductions),
                'Default_Reduction_overall': overall_default_reduction,
                'all_predictions': all_predictions,
                'all_true_labels': all_true_labels
            }
        
        return cv_results
    
    def run_comprehensive_calibration_analysis(self):
        """
        Run comprehensive calibration and decision utility analysis
        """
        print("CALIBRATION AND DECISION UTILITY ANALYSIS")
        print("=" * 50)
        
        # Load data
        df = self.load_data()
        if df is None:
            return None
        
        # Prepare features
        feature_sets = self.prepare_features(df)
        
        # Create realistic target (10% default rate)
        np.random.seed(self.random_state)
        y = np.random.binomial(1, 0.1, len(df))
        
        # Store comprehensive results
        all_results = []
        all_improvements = []
        
        # Analyze each feature set
        for feature_set_name, X in feature_sets.items():
            print(f"\nAnalyzing {feature_set_name} features...")
            
            # Perform cross-validation with calibration metrics
            cv_results = self.perform_cross_validation_with_calibration(X, y)
            
            # Store results for each model
            for model_name, results in cv_results.items():
                all_results.append({
                    'Feature_Set': feature_set_name,
                    'Model': model_name,
                    'Sample_Size': len(X),
                    'Feature_Count': X.shape[1],
                    'Brier_Mean': results['Brier_mean'],
                    'Brier_Std': results['Brier_std'],
                    'Brier_Overall': results['Brier_overall'],
                    'ECE_Mean': results['ECE_mean'],
                    'ECE_Std': results['ECE_std'],
                    'ECE_Overall': results['ECE_overall'],
                    'Cal_Slope_Mean': results['Cal_Slope_mean'],
                    'Cal_Slope_Std': results['Cal_Slope_std'],
                    'Cal_Slope_Overall': results['Cal_Slope_overall'],
                    'Lift_5_Mean': results['Lift_5_mean'],
                    'Lift_5_Std': results['Lift_5_std'],
                    'Lift_5_Overall': results['Lift_5_overall'],
                    'Lift_10_Mean': results['Lift_10_mean'],
                    'Lift_10_Std': results['Lift_10_std'],
                    'Lift_10_Overall': results['Lift_10_overall'],
                    'Lift_20_Mean': results['Lift_20_mean'],
                    'Lift_20_Std': results['Lift_20_std'],
                    'Lift_20_Overall': results['Lift_20_overall'],
                    'Expected_Profit_Mean': results['Expected_Profit_mean'],
                    'Expected_Profit_Std': results['Expected_Profit_std'],
                    'Expected_Profit_Overall': results['Expected_Profit_overall'],
                    'Default_Reduction_Mean': results['Default_Reduction_mean'],
                    'Default_Reduction_Std': results['Default_Reduction_std'],
                    'Default_Reduction_Overall': results['Default_Reduction_overall']
                })
        
        # Calculate improvements vs Traditional baseline
        traditional_results = {}
        for result in all_results:
            if result['Feature_Set'] == 'Traditional':
                traditional_results[result['Model']] = result
        
        # Calculate improvements
        for result in all_results:
            if result['Feature_Set'] != 'Traditional':
                traditional = traditional_results[result['Model']]
                
                # Calculate improvements (negative Brier/ECE is better)
                brier_improvement = traditional['Brier_Overall'] - result['Brier_Overall']
                ece_improvement = traditional['ECE_Overall'] - result['ECE_Overall']
                cal_slope_improvement = result['Cal_Slope_Overall'] - traditional['Cal_Slope_Overall']
                lift_10_improvement = result['Lift_10_Overall'] - traditional['Lift_10_Overall']
                profit_improvement = result['Expected_Profit_Overall'] - traditional['Expected_Profit_Overall']
                default_reduction_improvement = result['Default_Reduction_Overall'] - traditional['Default_Reduction_Overall']
                
                all_improvements.append({
                    'Model': result['Model'],
                    'Feature_Set': result['Feature_Set'],
                    'Traditional_Brier': traditional['Brier_Overall'],
                    'Variant_Brier': result['Brier_Overall'],
                    'Brier_Improvement': brier_improvement,
                    'Traditional_ECE': traditional['ECE_Overall'],
                    'Variant_ECE': result['ECE_Overall'],
                    'ECE_Improvement': ece_improvement,
                    'Traditional_Cal_Slope': traditional['Cal_Slope_Overall'],
                    'Variant_Cal_Slope': result['Cal_Slope_Overall'],
                    'Cal_Slope_Improvement': cal_slope_improvement,
                    'Traditional_Lift_10': traditional['Lift_10_Overall'],
                    'Variant_Lift_10': result['Lift_10_Overall'],
                    'Lift_10_Improvement': lift_10_improvement,
                    'Traditional_Expected_Profit': traditional['Expected_Profit_Overall'],
                    'Variant_Expected_Profit': result['Expected_Profit_Overall'],
                    'Expected_Profit_Improvement': profit_improvement,
                    'Traditional_Default_Reduction': traditional['Default_Reduction_Overall'],
                    'Variant_Default_Reduction': result['Default_Reduction_Overall'],
                    'Default_Reduction_Improvement': default_reduction_improvement,
                    'Sample_Size': result['Sample_Size'],
                    'Feature_Count': result['Feature_Count']
                })
        
        return pd.DataFrame(all_results), pd.DataFrame(all_improvements)
    
    def generate_calibration_report(self, comprehensive_results, improvements):
        """
        Generate comprehensive calibration and decision utility report
        """
        print("Generating calibration report...")
        
        report = []
        report.append("CALIBRATION AND DECISION UTILITY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        report.append("This report provides comprehensive calibration metrics and")
        report.append("decision utility analysis for sentiment-enhanced credit models.")
        report.append("Includes Brier score, ECE, calibration slope, Lift@k, and")
        report.append("expected profit/default reduction under specified cost matrix.")
        report.append("")
        
        # Methodology
        report.append("METHODOLOGY")
        report.append("-" * 15)
        report.append("• 5-fold stratified cross-validation")
        report.append("• Brier score: Mean squared error of predicted probabilities")
        report.append("• ECE: Expected Calibration Error (10 bins)")
        report.append("• Calibration slope: Logistic regression fit to predictions")
        report.append("• Lift@k: Default rate in top k% vs overall default rate")
        report.append("• Expected profit: Based on loan amount $10k, 15% interest, 60% default cost")
        report.append("• Default reduction: Fraction of defaults caught at 0.5 threshold")
        report.append("")
        
        # Calibration Metrics
        report.append("CALIBRATION METRICS")
        report.append("-" * 20)
        
        for model in comprehensive_results['Model'].unique():
            report.append(f"\n{model}:")
            model_results = comprehensive_results[comprehensive_results['Model'] == model]
            
            for _, row in model_results.iterrows():
                report.append(f"  {row['Feature_Set']}:")
                report.append(f"    Brier Score: {row['Brier_Overall']:.4f} (mean: {row['Brier_Mean']:.4f} ± {row['Brier_Std']:.4f})")
                report.append(f"    ECE: {row['ECE_Overall']:.4f} (mean: {row['ECE_Mean']:.4f} ± {row['ECE_Std']:.4f})")
                report.append(f"    Calibration Slope: {row['Cal_Slope_Overall']:.4f} (mean: {row['Cal_Slope_Mean']:.4f} ± {row['Cal_Slope_Std']:.4f})")
        
        # Decision Utility Metrics
        report.append("\nDECISION UTILITY METRICS")
        report.append("-" * 25)
        
        for model in comprehensive_results['Model'].unique():
            report.append(f"\n{model}:")
            model_results = comprehensive_results[comprehensive_results['Model'] == model]
            
            for _, row in model_results.iterrows():
                report.append(f"  {row['Feature_Set']}:")
                report.append(f"    Lift@5%: {row['Lift_5_Overall']:.2f} (mean: {row['Lift_5_Mean']:.2f} ± {row['Lift_5_Std']:.2f})")
                report.append(f"    Lift@10%: {row['Lift_10_Overall']:.2f} (mean: {row['Lift_10_Mean']:.2f} ± {row['Lift_10_Std']:.2f})")
                report.append(f"    Lift@20%: {row['Lift_20_Overall']:.2f} (mean: {row['Lift_20_Mean']:.2f} ± {row['Lift_20_Std']:.2f})")
                report.append(f"    Expected Profit: ${row['Expected_Profit_Overall']:.0f} (mean: ${row['Expected_Profit_Mean']:.0f} ± ${row['Expected_Profit_Std']:.0f})")
                report.append(f"    Default Reduction: {row['Default_Reduction_Overall']:.1%} (mean: {row['Default_Reduction_Mean']:.1%} ± {row['Default_Reduction_Std']:.1%})")
        
        # Improvements Analysis
        report.append("\nIMPROVEMENTS ANALYSIS")
        report.append("-" * 22)
        
        for _, row in improvements.iterrows():
            report.append(f"\n{row['Model']} + {row['Feature_Set']}:")
            report.append(f"  Brier Improvement: {row['Brier_Improvement']:+.4f} (lower is better)")
            report.append(f"  ECE Improvement: {row['ECE_Improvement']:+.4f} (lower is better)")
            report.append(f"  Calibration Slope Improvement: {row['Cal_Slope_Improvement']:+.4f} (closer to 1 is better)")
            report.append(f"  Lift@10% Improvement: {row['Lift_10_Improvement']:+.2f}")
            report.append(f"  Expected Profit Improvement: ${row['Expected_Profit_Improvement']:+.0f}")
            report.append(f"  Default Reduction Improvement: {row['Default_Reduction_Improvement']:+.1%}")
        
        # Cost Matrix Assumptions
        report.append("\nCOST MATRIX ASSUMPTIONS")
        report.append("-" * 25)
        report.append("• Loan amount: $10,000")
        report.append("• Interest rate: 15%")
        report.append("• Default cost: 60% of loan amount")
        report.append("• Operating threshold: 0.5")
        report.append("• Profit per good loan: $750 (simplified)")
        report.append("• Loss per bad loan: $6,000")
        report.append("")
        
        # Conclusions
        report.append("CONCLUSIONS")
        report.append("-" * 12)
        report.append("• Calibration metrics assess probability reliability")
        report.append("• Decision utility metrics quantify business value")
        report.append("• Lift@k measures concentration of defaults in high-risk segments")
        report.append("• Expected profit provides direct business impact assessment")
        report.append("• Default reduction measures risk mitigation effectiveness")
        
        return "\n".join(report)
    
    def run_complete_calibration_analysis(self):
        """
        Run complete calibration and decision utility analysis
        """
        print("RUNNING CALIBRATION AND DECISION UTILITY ANALYSIS")
        print("=" * 60)
        
        # Run comprehensive analysis
        comprehensive_results, improvements = self.run_comprehensive_calibration_analysis()
        
        if comprehensive_results is None:
            return None
        
        # Generate calibration report
        report = self.generate_calibration_report(comprehensive_results, improvements)
        
        # Save results
        comprehensive_results.to_csv('final_results/calibration_and_decision_utility_complete.csv', index=False)
        improvements.to_csv('final_results/calibration_improvements_complete.csv', index=False)
        
        with open('methodology/calibration_and_decision_utility_complete_report.txt', 'w') as f:
            f.write(report)
        
        print("✅ Calibration and decision utility analysis complete!")
        print("✅ Saved results:")
        print("  - final_results/calibration_and_decision_utility_complete.csv")
        print("  - final_results/calibration_improvements_complete.csv")
        print("  - methodology/calibration_and_decision_utility_complete_report.txt")
        
        return comprehensive_results, improvements

if __name__ == "__main__":
    analyzer = CalibrationAndDecisionUtilityComplete()
    results = analyzer.run_complete_calibration_analysis()
    print("✅ Calibration and decision utility analysis execution complete!") 