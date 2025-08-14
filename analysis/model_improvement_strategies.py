#!/usr/bin/env python3
"""
Model Improvement Strategies
===========================
Comprehensive strategies to improve model performance from current AUC: 0.6226
Target: AUC > 0.70
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime

class ModelImprovementStrategies:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load the comprehensive dataset and prepare for improvement"""
        print("Loading comprehensive dataset...")
        
        # Load the existing comprehensive dataset
        df = pd.read_csv('comprehensive_results/data/comprehensive_dataset.csv')
        
        # Create enhanced features
        X, y = self.create_enhanced_features(df)
        
        print(f"Dataset loaded: {len(df):,} samples, {X.shape[1]} features")
        return X, y, df
    
    def create_enhanced_features(self, df):
        """Create enhanced feature set with advanced engineering"""
        # Base features
        base_features = [
            'loan_amnt', 'annual_inc', 'dti', 'emp_length', 'fico_score',
            'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
            'revol_bal', 'revol_util', 'total_acc', 'home_ownership', 'purpose',
            'sentiment_score', 'sentiment_confidence', 'age', 'gender', 'race',
            'education', 'text_length', 'word_count'
        ]
        
        X = df[base_features].copy()
        
        # ENHANCED FEATURE ENGINEERING
        
        # 1. Advanced Financial Ratios
        X['debt_to_income_ratio'] = X['loan_amnt'] / X['annual_inc']
        X['credit_utilization_risk'] = X['revol_util'] / 100
        X['fico_relative'] = (X['fico_score'] - 600) / 250
        X['income_per_account'] = X['annual_inc'] / (X['total_acc'] + 1)
        X['loan_to_income_log'] = np.log1p(X['loan_amnt'] / X['annual_inc'])
        
        # 2. Advanced Sentiment Features
        X['sentiment_strength'] = np.abs(X['sentiment_score'] - 0.5) * 2
        X['sentiment_risk_multiplier'] = (1 - X['sentiment_score']) * X['sentiment_confidence']
        X['high_confidence_negative'] = ((X['sentiment_score'] < 0.3) & (X['sentiment_confidence'] > 0.8)).astype(int)
        X['high_confidence_positive'] = ((X['sentiment_score'] > 0.7) & (X['sentiment_confidence'] > 0.8)).astype(int)
        X['sentiment_volatility'] = X['sentiment_confidence'] * (1 - X['sentiment_confidence'])
        
        # 3. Polynomial Features
        X['dti_squared'] = X['dti'] ** 2
        X['fico_squared'] = X['fico_score'] ** 2
        X['income_squared'] = (X['annual_inc'] / 100000) ** 2
        
        # 4. Interaction Features
        X['dti_fico_interaction'] = X['dti'] * X['fico_score'] / 1000
        X['sentiment_income_interaction'] = X['sentiment_score'] * X['annual_inc'] / 100000
        X['age_income_interaction'] = X['age'] * X['annual_inc'] / 1000000
        X['sentiment_dti_amplifier'] = X['sentiment_risk_multiplier'] * X['dti'] / 45
        X['sentiment_fico_penalty'] = (1 - X['sentiment_score']) * (850 - X['fico_score']) / 230
        
        # 5. Risk Scoring Features
        X['risk_score'] = (
            X['dti'] * 0.3 +
            (850 - X['fico_score']) * 0.4 +
            (1 - X['sentiment_score']) * 0.3
        )
        X['risk_percentile'] = X['risk_score'].rank(pct=True)
        
        # 6. Text Complexity Features
        X['text_complexity'] = X['text_length'] / (X['word_count'] + 1)
        X['text_sentiment_density'] = X['sentiment_strength'] / (X['word_count'] + 1)
        
        # 7. Demographic Risk Features
        X['age_risk'] = np.where(X['age'] < 25, 1.2, np.where(X['age'] > 65, 1.1, 1.0))
        X['education_risk'] = np.where(X['education'] <= 1, 1.1, 1.0)
        
        # 8. Economic Context Features
        X['state_unemployment_rate'] = df['state_unemployment_rate']
        X['inflation_rate'] = df['inflation_rate']
        X['gdp_growth_rate'] = df['gdp_growth_rate']
        
        # 9. Employment Features
        X['job_tenure_years'] = df['job_tenure_years']
        X['credit_history_length'] = df['credit_history_length']
        
        # 10. Advanced Risk Features
        X['bankruptcy_count'] = df['bankruptcy_count']
        X['collections_count'] = df['collections_count']
        X['charge_offs'] = df['charge_offs']
        
        # Handle missing values
        X = X.fillna(0)
        
        y = df['loan_status']
        
        return X, y
    
    def strategy_1_hyperparameter_optimization(self, X, y):
        """Strategy 1: Advanced Hyperparameter Optimization"""
        print("\n" + "="*60)
        print("STRATEGY 1: HYPERPARAMETER OPTIMIZATION")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        # Apply SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
        results = {}
        
        # 1. XGBoost Optimization
        print("1.1 Optimizing XGBoost...")
        xgb_param_grid = {
            'n_estimators': [300, 500, 700],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=self.random_state)
        xgb_search = RandomizedSearchCV(
            xgb_model, xgb_param_grid, n_iter=50, cv=5, 
            scoring='roc_auc', random_state=self.random_state, n_jobs=-1
        )
        xgb_search.fit(X_balanced, y_balanced)
        
        y_pred_xgb = xgb_search.predict_proba(X_test)[:, 1]
        auc_xgb = roc_auc_score(y_test, y_pred_xgb)
        results['XGBoost_Optimized'] = auc_xgb
        
        print(f"   XGBoost Optimized AUC: {auc_xgb:.4f}")
        print(f"   Best params: {xgb_search.best_params_}")
        
        # 2. LightGBM Optimization
        print("1.2 Optimizing LightGBM...")
        lgb_param_grid = {
            'n_estimators': [300, 500, 700],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        
        lgb_model = lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
        lgb_search = RandomizedSearchCV(
            lgb_model, lgb_param_grid, n_iter=50, cv=5,
            scoring='roc_auc', random_state=self.random_state, n_jobs=-1
        )
        lgb_search.fit(X_balanced, y_balanced)
        
        y_pred_lgb = lgb_search.predict_proba(X_test)[:, 1]
        auc_lgb = roc_auc_score(y_test, y_pred_lgb)
        results['LightGBM_Optimized'] = auc_lgb
        
        print(f"   LightGBM Optimized AUC: {auc_lgb:.4f}")
        print(f"   Best params: {lgb_search.best_params_}")
        
        # 3. Random Forest Optimization
        print("1.3 Optimizing Random Forest...")
        rf_param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        rf_model = RandomForestClassifier(random_state=self.random_state)
        rf_search = GridSearchCV(
            rf_model, rf_param_grid, cv=5,
            scoring='roc_auc', n_jobs=-1
        )
        rf_search.fit(X_balanced, y_balanced)
        
        y_pred_rf = rf_search.predict_proba(X_test)[:, 1]
        auc_rf = roc_auc_score(y_test, y_pred_rf)
        results['RandomForest_Optimized'] = auc_rf
        
        print(f"   Random Forest Optimized AUC: {auc_rf:.4f}")
        print(f"   Best params: {rf_search.best_params_}")
        
        return results
    
    def strategy_2_advanced_ensemble(self, X, y):
        """Strategy 2: Advanced Ensemble Methods"""
        print("\n" + "="*60)
        print("STRATEGY 2: ADVANCED ENSEMBLE METHODS")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        # Apply SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
        results = {}
        
        # 1. Voting Classifier
        print("2.1 Creating Voting Classifier...")
        estimators = [
            ('xgb', xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=self.random_state)),
            ('lgb', lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=self.random_state, verbose=-1)),
            ('rf', RandomForestClassifier(n_estimators=300, max_depth=15, random_state=self.random_state)),
            ('gbm', GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=self.random_state))
        ]
        
        voting_clf = VotingClassifier(estimators=estimators, voting='soft')
        voting_clf.fit(X_balanced, y_balanced)
        
        y_pred_voting = voting_clf.predict_proba(X_test)[:, 1]
        auc_voting = roc_auc_score(y_test, y_pred_voting)
        results['Voting_Classifier'] = auc_voting
        
        print(f"   Voting Classifier AUC: {auc_voting:.4f}")
        
        # 2. Stacking Classifier
        print("2.2 Creating Stacking Classifier...")
        base_estimators = [
            ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=self.random_state)),
            ('lgb', lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=self.random_state, verbose=-1)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=self.random_state))
        ]
        
        meta_estimator = LogisticRegression(random_state=self.random_state)
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_estimator,
            cv=5,
            stack_method='predict_proba'
        )
        stacking_clf.fit(X_balanced, y_balanced)
        
        y_pred_stacking = stacking_clf.predict_proba(X_test)[:, 1]
        auc_stacking = roc_auc_score(y_test, y_pred_stacking)
        results['Stacking_Classifier'] = auc_stacking
        
        print(f"   Stacking Classifier AUC: {auc_stacking:.4f}")
        
        # 3. Neural Network
        print("2.3 Creating Neural Network...")
        nn_clf = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=self.random_state
        )
        nn_clf.fit(X_balanced, y_balanced)
        
        y_pred_nn = nn_clf.predict_proba(X_test)[:, 1]
        auc_nn = roc_auc_score(y_test, y_pred_nn)
        results['Neural_Network'] = auc_nn
        
        print(f"   Neural Network AUC: {auc_nn:.4f}")
        
        return results
    
    def strategy_3_feature_selection_and_engineering(self, X, y):
        """Strategy 3: Advanced Feature Selection and Engineering"""
        print("\n" + "="*60)
        print("STRATEGY 3: ADVANCED FEATURE SELECTION AND ENGINEERING")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        # Apply SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
        results = {}
        
        # 1. Feature Importance Selection
        print("3.1 Performing Feature Importance Selection...")
        rf_importance = RandomForestClassifier(n_estimators=200, random_state=self.random_state)
        rf_importance.fit(X_balanced, y_balanced)
        
        # Get top features
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_importance.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top 30 features
        top_features = feature_importance.head(30)['feature'].tolist()
        X_selected = X[top_features]
        
        X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(
            X_selected, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        X_balanced_selected, y_balanced_selected = smote.fit_resample(X_train_selected, y_train_selected)
        
        # Test with selected features
        xgb_selected = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=self.random_state)
        xgb_selected.fit(X_balanced_selected, y_balanced_selected)
        
        y_pred_selected = xgb_selected.predict_proba(X_test_selected)[:, 1]
        auc_selected = roc_auc_score(y_test_selected, y_pred_selected)
        results['Feature_Selection'] = auc_selected
        
        print(f"   Feature Selection AUC: {auc_selected:.4f}")
        print(f"   Selected {len(top_features)} features out of {X.shape[1]}")
        
        # 2. Feature Scaling
        print("3.2 Applying Feature Scaling...")
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        X_balanced_scaled, y_balanced_scaled = smote.fit_resample(X_train_scaled, y_train_scaled)
        
        # Test with scaled features
        xgb_scaled = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=self.random_state)
        xgb_scaled.fit(X_balanced_scaled, y_balanced_scaled)
        
        y_pred_scaled = xgb_scaled.predict_proba(X_test_scaled)[:, 1]
        auc_scaled = roc_auc_score(y_test_scaled, y_pred_scaled)
        results['Feature_Scaling'] = auc_scaled
        
        print(f"   Feature Scaling AUC: {auc_scaled:.4f}")
        
        return results
    
    def strategy_4_advanced_sampling(self, X, y):
        """Strategy 4: Advanced Sampling Techniques"""
        print("\n" + "="*60)
        print("STRATEGY 4: ADVANCED SAMPLING TECHNIQUES")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        results = {}
        
        # 1. ADASYN
        print("4.1 Testing ADASYN...")
        adasyn = ADASYN(random_state=self.random_state)
        X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
        
        xgb_adasyn = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=self.random_state)
        xgb_adasyn.fit(X_adasyn, y_adasyn)
        
        y_pred_adasyn = xgb_adasyn.predict_proba(X_test)[:, 1]
        auc_adasyn = roc_auc_score(y_test, y_pred_adasyn)
        results['ADASYN'] = auc_adasyn
        
        print(f"   ADASYN AUC: {auc_adasyn:.4f}")
        
        # 2. BorderlineSMOTE
        print("4.2 Testing BorderlineSMOTE...")
        borderline_smote = BorderlineSMOTE(random_state=self.random_state)
        X_borderline, y_borderline = borderline_smote.fit_resample(X_train, y_train)
        
        xgb_borderline = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=self.random_state)
        xgb_borderline.fit(X_borderline, y_borderline)
        
        y_pred_borderline = xgb_borderline.predict_proba(X_test)[:, 1]
        auc_borderline = roc_auc_score(y_test, y_pred_borderline)
        results['BorderlineSMOTE'] = auc_borderline
        
        print(f"   BorderlineSMOTE AUC: {auc_borderline:.4f}")
        
        # 3. SMOTETomek
        print("4.3 Testing SMOTETomek...")
        smote_tomek = SMOTETomek(random_state=self.random_state)
        X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(X_train, y_train)
        
        xgb_smote_tomek = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=self.random_state)
        xgb_smote_tomek.fit(X_smote_tomek, y_smote_tomek)
        
        y_pred_smote_tomek = xgb_smote_tomek.predict_proba(X_test)[:, 1]
        auc_smote_tomek = roc_auc_score(y_test, y_pred_smote_tomek)
        results['SMOTETomek'] = auc_smote_tomek
        
        print(f"   SMOTETomek AUC: {auc_smote_tomek:.4f}")
        
        return results
    
    def run_all_improvement_strategies(self):
        """Run all improvement strategies"""
        print("MODEL IMPROVEMENT STRATEGIES")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Target: Improve AUC from 0.6226 to > 0.70")
        print("="*60)
        
        # Load data
        X, y, df = self.load_and_prepare_data()
        
        # Run all strategies
        strategy1_results = self.strategy_1_hyperparameter_optimization(X, y)
        strategy2_results = self.strategy_2_advanced_ensemble(X, y)
        strategy3_results = self.strategy_3_feature_selection_and_engineering(X, y)
        strategy4_results = self.strategy_4_advanced_sampling(X, y)
        
        # Combine all results
        all_results = {**strategy1_results, **strategy2_results, **strategy3_results, **strategy4_results}
        
        # Generate improvement report
        self.generate_improvement_report(all_results, df)
        
        return all_results
    
    def generate_improvement_report(self, results, df):
        """Generate comprehensive improvement report"""
        print("\n" + "="*60)
        print("IMPROVEMENT RESULTS SUMMARY")
        print("="*60)
        
        # Sort results by AUC
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        print("Top 10 Best Performing Models:")
        print("-" * 40)
        for i, (model_name, auc) in enumerate(sorted_results[:10], 1):
            improvement = (auc - 0.6226) * 100
            print(f"{i:2d}. {model_name:<25} AUC: {auc:.4f} (+{improvement:+.2f}%)")
        
        print(f"\nBaseline AUC: 0.6226")
        print(f"Best AUC: {sorted_results[0][1]:.4f}")
        print(f"Improvement: +{(sorted_results[0][1] - 0.6226) * 100:.2f}%")
        
        # Save detailed report
        os.makedirs('comprehensive_evaluation/improvements', exist_ok=True)
        
        with open('comprehensive_evaluation/improvements/model_improvement_report.txt', 'w') as f:
            f.write("MODEL IMPROVEMENT STRATEGIES REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Baseline AUC: 0.6226\n")
            f.write(f"Best Improved AUC: {sorted_results[0][1]:.4f}\n")
            f.write(f"Improvement: +{(sorted_results[0][1] - 0.6226) * 100:.2f}%\n\n")
            
            f.write("DETAILED RESULTS BY STRATEGY\n")
            f.write("-" * 30 + "\n")
            
            f.write("Strategy 1: Hyperparameter Optimization\n")
            f.write("-" * 30 + "\n")
            for model_name, auc in sorted_results:
                if 'Optimized' in model_name:
                    f.write(f"{model_name}: {auc:.4f}\n")
            f.write("\n")
            
            f.write("Strategy 2: Advanced Ensemble Methods\n")
            f.write("-" * 30 + "\n")
            for model_name, auc in sorted_results:
                if any(keyword in model_name for keyword in ['Voting', 'Stacking', 'Neural']):
                    f.write(f"{model_name}: {auc:.4f}\n")
            f.write("\n")
            
            f.write("Strategy 3: Feature Selection and Engineering\n")
            f.write("-" * 30 + "\n")
            for model_name, auc in sorted_results:
                if any(keyword in model_name for keyword in ['Feature', 'Scaling']):
                    f.write(f"{model_name}: {auc:.4f}\n")
            f.write("\n")
            
            f.write("Strategy 4: Advanced Sampling Techniques\n")
            f.write("-" * 30 + "\n")
            for model_name, auc in sorted_results:
                if any(keyword in model_name for keyword in ['ADASYN', 'Borderline', 'SMOTETomek']):
                    f.write(f"{model_name}: {auc:.4f}\n")
            f.write("\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            f.write("1. Implement the best performing model: " + sorted_results[0][0] + "\n")
            f.write("2. Consider ensemble methods for production deployment\n")
            f.write("3. Apply feature selection to reduce complexity\n")
            f.write("4. Use advanced sampling techniques for better balance\n")
            f.write("5. Continue hyperparameter optimization with larger search space\n")
        
        print(f"\nDetailed report saved to: comprehensive_evaluation/improvements/model_improvement_report.txt")

def run_model_improvement():
    """Run the model improvement strategies"""
    print("Starting Model Improvement Strategies...")
    
    # Initialize improvement strategies
    improvement = ModelImprovementStrategies()
    
    # Run all strategies
    results = improvement.run_all_improvement_strategies()
    
    print("\n" + "="*60)
    print("MODEL IMPROVEMENT COMPLETE!")
    print("="*60)
    
    return results

if __name__ == "__main__":
    run_model_improvement() 