#!/usr/bin/env python3
"""
STREAMLINED LENDING CLUB WORKFLOW - Ultra Optimized
====================================================
Reuses all existing files, minimal processing, fast execution
Expected runtime: 2-3 minutes for 1000 samples
"""

import os
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamlinedWorkflow:
    def __init__(self, sample_size=1000):
        self.sample_size = sample_size
        self.results = {}
        
    def load_data_smart(self):
        """Smart data loading - use existing files first"""
        logger.info(f"Loading data with smart caching for {self.sample_size} samples...")
        
        # Check if we have a processed file already
        processed_file = "processed_data_sample.csv"
        if os.path.exists(processed_file):
            logger.info(f"Found existing processed file: {processed_file}")
            df = pd.read_csv(processed_file)
            if len(df) >= self.sample_size:
                logger.info(f"Using existing processed data, taking {self.sample_size} samples")
                return df.sample(n=self.sample_size, random_state=42).reset_index(drop=True)
        
        # Load fresh data
        logger.info("Loading fresh data...")
        local_file = "accepted_2007_to_2018q4.csv"
        
        if os.path.exists(local_file):
            logger.info(f"Loading from local file: {local_file}")
            df = pd.read_csv(local_file, nrows=self.sample_size*2)  # Load extra for filtering
        else:
            from data_loader import load_lending_club_data
            df = load_lending_club_data(use_local_copy=True, sample_size=self.sample_size*2)
        
        logger.info(f"Loaded {len(df)} records, processing...")
        return df
    
    def prepare_data_fast(self, df):
        """Ultra-fast data preparation"""
        logger.info("Fast data preparation...")
        
        # Keep only essential features
        essential_cols = [
            'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'annual_inc',
            'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high', 'open_acc',
            'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'loan_status',
            'grade', 'emp_length', 'home_ownership', 'verification_status', 'purpose'
        ]
        
        # Keep only available columns
        available_cols = [col for col in essential_cols if col in df.columns]
        df_clean = df[available_cols].copy()
        
        # Remove rows with missing critical data
        critical_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'loan_status']
        for col in critical_cols:
            if col in df_clean.columns:
                df_clean = df_clean.dropna(subset=[col])
        
        # Create binary target
        if 'loan_status' in df_clean.columns:
            default_statuses = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off']
            df_clean['target'] = df_clean['loan_status'].isin(default_statuses).astype(int)
            df_clean = df_clean.drop(columns=['loan_status'])
        
        # Take sample now
        if len(df_clean) > self.sample_size:
            df_clean = df_clean.sample(n=self.sample_size, random_state=42).reset_index(drop=True)
        
        # Quick encoding of categorical variables
        categorical_cols = ['grade', 'emp_length', 'home_ownership', 'verification_status', 'purpose']
        le_dict = {}
        
        for col in categorical_cols:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                le_dict[col] = le
        
        # Fill missing values quickly
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        
        # Add sentiment if available
        self.add_sentiment_fast(df_clean)
        
        # Separate features and target
        X = df_clean.drop(columns=['target'])
        y = df_clean['target']
        
        logger.info(f"Prepared data: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def add_sentiment_fast(self, df):
        """Add sentiment features from existing CSV if available"""
        sentiment_file = 'loan_sentiment_results.csv'
        
        if os.path.exists(sentiment_file):
            logger.info(f"Loading sentiment from {sentiment_file}...")
            try:
                sentiment_df = pd.read_csv(sentiment_file)
                if 'loan_amnt' in sentiment_df.columns and 'loan_amnt' in df.columns:
                    # Quick merge on loan amount
                    sentiment_map = {'POSITIVE': 0.7, 'NEGATIVE': 0.3, 'NEUTRAL': 0.5}
                    
                    # Create a mapping dict for fast lookup
                    sentiment_lookup = dict(zip(
                        sentiment_df['loan_amnt'], 
                        sentiment_df['sentiment'].map(sentiment_map)
                    ))
                    
                    confidence_lookup = dict(zip(
                        sentiment_df['loan_amnt'], 
                        sentiment_df['confidence']
                    ))
                    
                    # Fast mapping
                    df['sentiment_score'] = df['loan_amnt'].map(sentiment_lookup).fillna(0.5)
                    df['sentiment_confidence'] = df['loan_amnt'].map(confidence_lookup).fillna(0.5)
                    
                    logger.info("Added sentiment features from existing data")
                    return
            except Exception as e:
                logger.warning(f"Could not load sentiment: {e}")
        
        # Default neutral sentiment
        df['sentiment_score'] = 0.5
        df['sentiment_confidence'] = 0.5
        logger.info("Added default neutral sentiment")
    
    def train_models_fast(self, X, y):
        """Train essential models only - no ensemble complexity"""
        logger.info("Training essential models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models = {}
        
        # 1. Baseline XGBoost
        logger.info("Training baseline XGBoost...")
        xgb_base = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=50,
            max_depth=4,
            random_state=42,
            verbosity=0,
            enable_categorical=False
        )
        xgb_base.fit(X_train, y_train)
        models['XGBoost_Baseline'] = xgb_base
        
        # 2. Class-weighted XGBoost
        logger.info("Training class-weighted XGBoost...")
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        xgb_weighted = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=50,
            max_depth=4,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=0,
            enable_categorical=False
        )
        xgb_weighted.fit(X_train, y_train)
        models['XGBoost_Weighted'] = xgb_weighted
        
        logger.info("Model training completed")
        return models, X_test, y_test
    
    def evaluate_fast(self, models, X_test, y_test):
        """Fast evaluation - essential metrics only"""
        logger.info("Running fast evaluation...")
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Essential metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            try:
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.5
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'test_samples': len(y_test)
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
        
        return results
    
    def run_streamlined_workflow(self):
        """Run the complete streamlined workflow"""
        logger.info("=== STREAMLINED LENDING CLUB WORKFLOW ===")
        logger.info(f"Sample size: {self.sample_size}")
        logger.info("Expected runtime: 2-3 minutes")
        
        try:
            # 1. Load data smartly
            df = self.load_data_smart()
            
            # 2. Prepare data fast
            X, y = self.prepare_data_fast(df)
            
            # 3. Train models fast
            models, X_test, y_test = self.train_models_fast(X, y)
            
            # 4. Evaluate fast
            self.results = self.evaluate_fast(models, X_test, y_test)
            
            # 5. Print final results
            self.print_final_results()
            
            logger.info("STREAMLINED WORKFLOW COMPLETED SUCCESSFULLY!")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in streamlined workflow: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def print_final_results(self):
        """Print clean final results"""
        logger.info("\n" + "="*50)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("="*50)
        
        for model_name, metrics in self.results.items():
            logger.info(f"{model_name:20} | Accuracy: {metrics['accuracy']:.3f} | AUC: {metrics['auc']:.3f}")
        
        logger.info("="*50)
        
        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['auc'])
        logger.info(f"BEST MODEL: {best_model[0]} (AUC: {best_model[1]['auc']:.3f})")

def main():
    """Main execution"""
    import sys
    
    # Get sample size from command line or use default
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    
    workflow = StreamlinedWorkflow(sample_size=sample_size)
    results = workflow.run_streamlined_workflow()
    
    if results:
        print(f"\nSUCCESS! Workflow completed with {sample_size} samples")
        print("Results saved in workflow.results")
    else:
        print("Workflow failed - check logs above")

if __name__ == "__main__":
    main() 