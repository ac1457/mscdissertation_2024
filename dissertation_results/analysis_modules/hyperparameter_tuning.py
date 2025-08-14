#!/usr/bin/env python3
"""
Hyperparameter Tuning and Cross-Validation - Lending Club Sentiment Analysis
============================================================================
Implements comprehensive hyperparameter tuning and cross-validation to optimize
model performance and ensure robustness.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuning:
    """
    Hyperparameter tuning and cross-validation for credit risk modeling
    """
    
    def __init__(self, random_state=42, cv_folds=5):
        self.random_state = random_state
        self.cv_folds = cv_folds
        np.random.seed(random_state)
        
    def load_data(self):
        """
        Load and prepare data for hyperparameter tuning
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
        # Traditional features (using available features)
        traditional_features = [
            'purpose', 'sentiment_score', 'sentiment_confidence', 
            'text_length', 'word_count', 'sentence_count',
            'has_positive_words', 'has_negative_words', 'has_financial_terms'
        ]
        traditional_features = [f for f in traditional_features if f in df.columns]
        
        # Sentiment features (additional sentiment-related features)
        sentiment_features = [
            'sentiment', 'sentiment_score', 'sentiment_confidence', 
            'text_length', 'word_count', 'sentence_count'
        ]
        sentiment_features = [f for f in sentiment_features if f in df.columns]
        
        # Hybrid features (interactions - create from existing features)
        df['sentiment_text_interaction'] = df['sentiment_score'] * df['text_length']
        df['sentiment_word_interaction'] = df['sentiment_score'] * df['word_count']
        df['sentiment_purpose_interaction'] = df['sentiment_score'] * df['purpose'].astype('category').cat.codes
        
        hybrid_features = [
            'sentiment_text_interaction', 'sentiment_word_interaction', 
            'sentiment_purpose_interaction'
        ]
        
        # Prepare feature sets
        X_traditional = df[traditional_features].copy()
        X_sentiment = df[traditional_features + sentiment_features].copy()
        X_hybrid = df[traditional_features + sentiment_features + hybrid_features].copy()
        
        # Handle missing values and categorical variables
        for X in [X_traditional, X_sentiment, X_hybrid]:
            # Convert categorical columns to numeric
            for col in X.columns:
                if col == 'purpose' or col == 'sentiment':
                    # Convert categorical to numeric
                    X[col] = X[col].astype('category').cat.codes
                # Fill any remaining missing values with median
                X[col] = X[col].fillna(X[col].median())
        
        # Target variable - create synthetic target
        np.random.seed(self.random_state)
        y = np.random.binomial(1, 0.1, len(df))  # 10% default rate
        
        return {
            'Traditional': X_traditional,
            'Sentiment': X_sentiment,
            'Hybrid': X_hybrid
        }, y
    
    def define_hyperparameter_grids(self):
        """
        Define hyperparameter grids for each model
        """
        # RandomForest hyperparameter grid
        rf_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # XGBoost hyperparameter grid
        xgb_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        
        # LogisticRegression hyperparameter grid
        lr_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        }
        
        return {
            'RandomForest': rf_grid,
            'XGBoost': xgb_grid,
            'LogisticRegression': lr_grid
        }
    
    def perform_hyperparameter_tuning(self, feature_sets, y):
        """
        Perform hyperparameter tuning for each model and feature set
        """
        print("Performing hyperparameter tuning...")
        
        # Define models and grids
        models = {
            'RandomForest': RandomForestClassifier(random_state=self.random_state),
            'XGBoost': XGBClassifier(random_state=self.random_state),
            'LogisticRegression': LogisticRegression(random_state=self.random_state)
        }
        
        grids = self.define_hyperparameter_grids()
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scorer = make_scorer(roc_auc_score, needs_proba=True)
        
        results = []
        
        for feature_set_name, X in feature_sets.items():
            print(f"\nTuning {feature_set_name} features...")
            
            for model_name, model in models.items():
                print(f"  Tuning {model_name}...")
                
                # Get hyperparameter grid
                grid = grids[model_name]
                
                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=grid,
                    cv=cv,
                    scoring=scorer,
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X, y)
                
                # Get best parameters and score
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                
                # Cross-validation scores
                cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring=scorer)
                
                results.append({
                    'Feature_Set': feature_set_name,
                    'Model': model_name,
                    'Best_Params': str(best_params),
                    'Best_CV_Score': best_score,
                    'CV_Score_Mean': cv_scores.mean(),
                    'CV_Score_Std': cv_scores.std(),
                    'CV_Score_Min': cv_scores.min(),
                    'CV_Score_Max': cv_scores.max(),
                    'Feature_Count': X.shape[1]
                })
                
                print(f"    Best CV Score: {best_score:.4f} ± {cv_scores.std():.4f}")
        
        return pd.DataFrame(results)
    
    def compare_tuned_vs_untuned(self, feature_sets, y):
        """
        Compare tuned vs untuned model performance
        """
        print("Comparing tuned vs untuned performance...")
        
        # Untuned models
        untuned_models = {
            'RandomForest': RandomForestClassifier(random_state=self.random_state),
            'XGBoost': XGBClassifier(random_state=self.random_state),
            'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=1000)
        }
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scorer = make_scorer(roc_auc_score, needs_proba=True)
        
        comparison_results = []
        
        for feature_set_name, X in feature_sets.items():
            for model_name, model in untuned_models.items():
                # Cross-validation scores for untuned model
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
                
                comparison_results.append({
                    'Feature_Set': feature_set_name,
                    'Model': model_name,
                    'Tuning_Status': 'Untuned',
                    'CV_Score_Mean': cv_scores.mean(),
                    'CV_Score_Std': cv_scores.std(),
                    'Feature_Count': X.shape[1]
                })
        
        return pd.DataFrame(comparison_results)
    
    def analyze_feature_importance(self, feature_sets, y):
        """
        Analyze feature importance for tuned models
        """
        print("Analyzing feature importance...")
        
        # Focus on RandomForest for feature importance
        rf_model = RandomForestClassifier(random_state=self.random_state)
        
        importance_results = []
        
        for feature_set_name, X in feature_sets.items():
            # Train model
            rf_model.fit(X, y)
            
            # Get feature importance
            importances = rf_model.feature_importances_
            feature_names = X.columns
            
            # Sort by importance
            importance_pairs = list(zip(feature_names, importances))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Top features
            top_features = importance_pairs[:10]
            
            for feature_name, importance in top_features:
                importance_results.append({
                    'Feature_Set': feature_set_name,
                    'Feature': feature_name,
                    'Importance': importance,
                    'Rank': len([x for x in importance_pairs if x[1] > importance]) + 1
                })
        
        return pd.DataFrame(importance_results)
    
    def generate_tuning_report(self, tuning_results, comparison_results, importance_results):
        """
        Generate comprehensive hyperparameter tuning report
        """
        report = []
        report.append("HYPERPARAMETER TUNING AND CROSS-VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Cross-validation setup
        report.append("CROSS-VALIDATION SETUP")
        report.append("-" * 25)
        report.append(f"Folds: {self.cv_folds}")
        report.append(f"Scoring: ROC AUC")
        report.append(f"Random State: {self.random_state}")
        report.append("")
        
        # Best performing combinations
        report.append("BEST PERFORMING COMBINATIONS")
        report.append("-" * 30)
        
        # Sort by best CV score
        best_results = tuning_results.sort_values('Best_CV_Score', ascending=False)
        
        for i, (_, row) in enumerate(best_results.head(5).iterrows()):
            report.append(f"{i+1}. {row['Model']} + {row['Feature_Set']}")
            report.append(f"   CV Score: {row['Best_CV_Score']:.4f} ± {row['CV_Score_Std']:.4f}")
            report.append(f"   Features: {row['Feature_Count']}")
            report.append("")
        
        # Tuning improvements
        report.append("TUNING IMPROVEMENTS ANALYSIS")
        report.append("-" * 30)
        
        # Compare tuned vs untuned
        for feature_set in tuning_results['Feature_Set'].unique():
            report.append(f"\n{feature_set} Features:")
            
            for model in tuning_results['Model'].unique():
                tuned_result = tuning_results[
                    (tuning_results['Feature_Set'] == feature_set) & 
                    (tuning_results['Model'] == model)
                ]
                
                untuned_result = comparison_results[
                    (comparison_results['Feature_Set'] == feature_set) & 
                    (comparison_results['Model'] == model)
                ]
                
                if len(tuned_result) > 0 and len(untuned_result) > 0:
                    tuned_score = tuned_result.iloc[0]['Best_CV_Score']
                    untuned_score = untuned_result.iloc[0]['CV_Score_Mean']
                    improvement = tuned_score - untuned_score
                    improvement_percent = (improvement / untuned_score) * 100
                    
                    report.append(f"  {model}:")
                    report.append(f"    Untuned: {untuned_score:.4f}")
                    report.append(f"    Tuned: {tuned_score:.4f}")
                    report.append(f"    Improvement: +{improvement:.4f} (+{improvement_percent:.2f}%)")
        
        # Feature importance analysis
        report.append("\nTOP FEATURES BY IMPORTANCE")
        report.append("-" * 30)
        
        for feature_set in importance_results['Feature_Set'].unique():
            report.append(f"\n{feature_set} Features:")
            feature_set_importance = importance_results[
                importance_results['Feature_Set'] == feature_set
            ].head(5)
            
            for _, row in feature_set_importance.iterrows():
                report.append(f"  {row['Rank']}. {row['Feature']}: {row['Importance']:.4f}")
        
        # Key insights
        report.append("\nKEY INSIGHTS")
        report.append("-" * 15)
        
        # Best overall performance
        best_overall = tuning_results.loc[tuning_results['Best_CV_Score'].idxmax()]
        report.append(f"• Best Overall: {best_overall['Model']} + {best_overall['Feature_Set']}")
        report.append(f"  CV Score: {best_overall['Best_CV_Score']:.4f} ± {best_overall['CV_Score_Std']:.4f}")
        
        # Average improvement from tuning
        avg_improvement = 0
        improvement_count = 0
        
        for feature_set in tuning_results['Feature_Set'].unique():
            for model in tuning_results['Model'].unique():
                tuned_result = tuning_results[
                    (tuning_results['Feature_Set'] == feature_set) & 
                    (tuning_results['Model'] == model)
                ]
                
                untuned_result = comparison_results[
                    (comparison_results['Feature_Set'] == feature_set) & 
                    (comparison_results['Model'] == model)
                ]
                
                if len(tuned_result) > 0 and len(untuned_result) > 0:
                    tuned_score = tuned_result.iloc[0]['Best_CV_Score']
                    untuned_score = untuned_result.iloc[0]['CV_Score_Mean']
                    improvement = (tuned_score - untuned_score) / untuned_score * 100
                    avg_improvement += improvement
                    improvement_count += 1
        
        if improvement_count > 0:
            avg_improvement /= improvement_count
            report.append(f"• Average Tuning Improvement: {avg_improvement:.2f}%")
        
        # Cross-validation stability
        cv_stability = tuning_results['CV_Score_Std'].mean()
        report.append(f"• Average CV Stability (Std): {cv_stability:.4f}")
        
        return "\n".join(report)
    
    def run_complete_tuning_analysis(self):
        """
        Run complete hyperparameter tuning and cross-validation analysis
        """
        print("HYPERPARAMETER TUNING AND CROSS-VALIDATION ANALYSIS")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        if df is None:
            return None
        
        # Prepare features
        feature_sets, y = self.prepare_features(df)
        
        # Perform hyperparameter tuning
        tuning_results = self.perform_hyperparameter_tuning(feature_sets, y)
        
        # Compare tuned vs untuned
        comparison_results = self.compare_tuned_vs_untuned(feature_sets, y)
        
        # Analyze feature importance
        importance_results = self.analyze_feature_importance(feature_sets, y)
        
        # Generate report
        report = self.generate_tuning_report(tuning_results, comparison_results, importance_results)
        
        # Save results
        tuning_results.to_csv('final_results/hyperparameter_tuning_results.csv', index=False)
        comparison_results.to_csv('final_results/tuned_vs_untuned_comparison.csv', index=False)
        importance_results.to_csv('final_results/feature_importance_analysis.csv', index=False)
        
        with open('methodology/hyperparameter_tuning_report.txt', 'w') as f:
            f.write(report)
        
        print("✅ Saved hyperparameter tuning results:")
        print("  - final_results/hyperparameter_tuning_results.csv")
        print("  - final_results/tuned_vs_untuned_comparison.csv")
        print("  - final_results/feature_importance_analysis.csv")
        print("  - methodology/hyperparameter_tuning_report.txt")
        
        return tuning_results, comparison_results, importance_results

if __name__ == "__main__":
    tuner = HyperparameterTuning()
    tuning_results, comparison_results, importance_results = tuner.run_complete_tuning_analysis()
    print("✅ Hyperparameter tuning and cross-validation complete!") 