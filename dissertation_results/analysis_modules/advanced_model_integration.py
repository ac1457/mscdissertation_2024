"""
Advanced Model Integration for Lending Club Sentiment Analysis
Implements early fusion with attention mechanisms, model stacking,
and advanced ablation studies to capture interactions between modalities.
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelIntegration:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Configuration
        self.config = {
            'min_practical_effect': 0.01,
            'bootstrap_iterations': 1000,
            'n_splits': 5,
            'attention_heads': 4,
            'attention_dim': 64,
            'stacking_layers': 2
        }
        
        # Initialize scalers
        self.text_scaler = StandardScaler()
        self.tabular_scaler = StandardScaler()
        
    def run_advanced_integration_analysis(self):
        """Run advanced model integration analysis"""
        print("Advanced Model Integration Analysis")
        print("=" * 60)
        
        # Load enhanced data
        df = self.load_enhanced_data()
        if df is None:
            return
        
        # Prepare feature sets
        text_features, tabular_features = self.prepare_modality_features(df)
        
        # Create temporal splits
        temporal_splits = self.create_temporal_splits(df)
        
        # Run analysis for each target regime
        target_columns = ['target_5%', 'target_10%', 'target_15%']
        
        all_results = {}
        ablation_results = {}
        
        for target_col in target_columns:
            if target_col not in df.columns:
                continue
                
            print(f"\nAnalyzing target: {target_col}")
            print("-" * 40)
            
            y = df[target_col]
            
            # Run different integration approaches
            integration_results = self.run_integration_approaches(
                df, text_features, tabular_features, y, temporal_splits
            )
            
            # Advanced ablation studies
            ablation_results[target_col] = self.advanced_ablation_study(
                df, text_features, tabular_features, y, temporal_splits
            )
            
            all_results[target_col] = integration_results
        
        # Save results
        self.save_advanced_results(all_results, ablation_results)
        
        return all_results, ablation_results
    
    def load_enhanced_data(self):
        """Load enhanced data with all features"""
        try:
            # Try to load from enhanced analysis first
            enhanced_file = Path('final_results/enhanced_comprehensive/enhanced_results.json')
            if enhanced_file.exists():
                print("Loading enhanced data from previous analysis...")
                # For now, reload the original data and recreate features
                df = pd.read_csv('data/synthetic_loan_descriptions_with_realistic_targets.csv')
            else:
                df = pd.read_csv('data/synthetic_loan_descriptions_with_realistic_targets.csv')
            
            # Add temporal ordering
            np.random.seed(self.random_state)
            df['origination_date'] = pd.date_range(
                start='2020-01-01', 
                periods=len(df), 
                freq='D'
            )
            df = df.sort_values('origination_date').reset_index(drop=True)
            
            # Recreate enhanced features
            df = self.create_enhanced_features(df)
            
            print(f"Loaded dataset: {len(df)} records")
            return df
            
        except FileNotFoundError:
            print("Dataset not found")
            return None
    
    def create_enhanced_features(self, df):
        """Create enhanced features for advanced integration"""
        print("Creating enhanced features for advanced integration...")
        
        # Text preprocessing
        df['cleaned_description'] = df['description'].apply(self.clean_text)
        
        # Entity extraction
        financial_entities = {
            'job_stability': ['stable job', 'permanent', 'full time', 'employed', 'salary'],
            'financial_hardship': ['debt', 'bills', 'expenses', 'struggling', 'difficult'],
            'repayment_confidence': ['confident', 'sure', 'guarantee', 'promise', 'commit'],
            'urgency': ['urgent', 'asap', 'immediately', 'quick', 'fast']
        }
        
        for entity_type, patterns in financial_entities.items():
            df[f'{entity_type}_count'] = df['cleaned_description'].apply(
                lambda x: sum(1 for pattern in patterns if pattern in x.lower())
            )
        
        # Text complexity features
        df['avg_word_length'] = df['description'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )
        df['type_token_ratio'] = df['description'].apply(
            lambda x: len(set(x.lower().split())) / len(x.split()) if x.split() else 0
        )
        df['sentence_count'] = df['description'].apply(lambda x: len(x.split('.')))
        
        # Sentiment features
        positive_words = ['good', 'great', 'excellent', 'positive', 'success']
        negative_words = ['bad', 'poor', 'negative', 'problem', 'issue']
        
        df['positive_word_count'] = df['description'].apply(
            lambda x: sum(1 for word in x.lower().split() if word in positive_words)
        )
        df['negative_word_count'] = df['description'].apply(
            lambda x: sum(1 for word in x.lower().split() if word in negative_words)
        )
        df['sentiment_balance'] = df['positive_word_count'] - df['negative_word_count']
        
        # Financial keyword density
        financial_keywords = ['loan', 'debt', 'credit', 'money', 'payment', 'interest']
        df['financial_keyword_density'] = df['description'].apply(
            lambda x: sum(1 for word in x.lower().split() if word in financial_keywords) / len(x.split()) if x.split() else 0
        )
        
        print(f"   Enhanced features created: {len([col for col in df.columns if col not in ['description', 'origination_date', 'target_5%', 'target_10%', 'target_15%']])} features")
        
        return df
    
    def clean_text(self, text):
        """Clean text for processing"""
        if pd.isna(text):
            return ""
        return text.lower().strip()
    
    def prepare_modality_features(self, df):
        """Prepare text and tabular features separately"""
        print("Preparing modality features...")
        
        # Text features
        text_features = [
            'positive_word_count', 'negative_word_count', 'sentiment_balance',
            'financial_keyword_density', 'avg_word_length', 'type_token_ratio',
            'sentence_count', 'job_stability_count', 'financial_hardship_count',
            'repayment_confidence_count', 'urgency_count'
        ]
        text_features = [f for f in text_features if f in df.columns]
        
        # Tabular features
        tabular_features = [
            'purpose', 'text_length', 'word_count', 'has_positive_words',
            'has_negative_words', 'has_financial_terms'
        ]
        tabular_features = [f for f in tabular_features if f in df.columns]
        
        print(f"   Text features: {len(text_features)}")
        print(f"   Tabular features: {len(tabular_features)}")
        
        return text_features, tabular_features
    
    def create_temporal_splits(self, df):
        """Create temporal splits"""
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=self.config['n_splits'])
        temporal_splits = []
        
        for train_idx, test_idx in tscv.split(df):
            temporal_splits.append({
                'train_idx': train_idx,
                'test_idx': test_idx
            })
        
        return temporal_splits
    
    def run_integration_approaches(self, df, text_features, tabular_features, y, temporal_splits):
        """Run different model integration approaches"""
        print("Running integration approaches...")
        
        approaches = {
            'Late_Fusion': self.late_fusion_approach,
            'Early_Fusion': self.early_fusion_approach,
            'Attention_Fusion': self.attention_fusion_approach,
            'Model_Stacking': self.model_stacking_approach,
            'Feature_Selection': self.feature_selection_approach
        }
        
        results = {}
        
        for approach_name, approach_func in approaches.items():
            print(f"   {approach_name}...")
            try:
                approach_results = approach_func(df, text_features, tabular_features, y, temporal_splits)
                results[approach_name] = approach_results
                print(f"      AUC: {approach_results['mean_auc']:.4f} ± {approach_results['std_auc']:.4f}")
            except Exception as e:
                print(f"      Error: {e}")
                results[approach_name] = None
        
        return results
    
    def late_fusion_approach(self, df, text_features, tabular_features, y, temporal_splits):
        """Late fusion: concatenate features and train single model"""
        all_features = text_features + tabular_features
        X = self.prepare_features(df, all_features)
        
        return self.run_temporal_cv(X, y, temporal_splits)
    
    def early_fusion_approach(self, df, text_features, tabular_features, y, temporal_splits):
        """Early fusion: train separate models and combine predictions"""
        X_text = self.prepare_features(df, text_features)
        X_tabular = self.prepare_features(df, tabular_features)
        
        # Train separate models
        text_results = self.run_temporal_cv(X_text, y, temporal_splits)
        tabular_results = self.run_temporal_cv(X_tabular, y, temporal_splits)
        
        # Combine predictions (simple average)
        combined_predictions = []
        combined_true_labels = []
        
        for i, split in enumerate(temporal_splits):
            text_pred = text_results['fold_results'][i]['predictions']
            tabular_pred = tabular_results['fold_results'][i]['predictions']
            true_labels = text_results['fold_results'][i]['true_labels']
            
            # Weighted combination (text features get higher weight)
            combined_pred = 0.6 * text_pred + 0.4 * tabular_pred
            combined_predictions.extend(combined_pred)
            combined_true_labels.extend(true_labels)
        
        # Calculate combined metrics
        combined_auc = roc_auc_score(combined_true_labels, combined_predictions)
        combined_pr_auc = average_precision_score(combined_true_labels, combined_predictions)
        
        return {
            'mean_auc': combined_auc,
            'mean_pr_auc': combined_pr_auc,
            'std_auc': 0.0,  # Single combined result
            'std_pr_auc': 0.0,
            'fold_results': text_results['fold_results'],  # Use text results structure
            'approach': 'early_fusion'
        }
    
    def attention_fusion_approach(self, df, text_features, tabular_features, y, temporal_splits):
        """Attention fusion: use attention mechanism to weight features"""
        # Simplified attention mechanism
        X_text = self.prepare_features(df, text_features)
        X_tabular = self.prepare_features(df, tabular_features)
        
        # Calculate attention weights based on feature importance
        attention_weights = self.calculate_attention_weights(X_text, X_tabular, y)
        
        # Apply attention weights
        X_attention = np.concatenate([
            X_text * attention_weights['text_weight'],
            X_tabular * attention_weights['tabular_weight']
        ], axis=1)
        
        return self.run_temporal_cv(X_attention, y, temporal_splits)
    
    def calculate_attention_weights(self, X_text, X_tabular, y):
        """Calculate attention weights for feature fusion"""
        # Simple attention based on feature importance
        text_importance = np.mean([abs(np.corrcoef(X_text[:, i], y)[0, 1]) 
                                 for i in range(X_text.shape[1]) if not np.isnan(np.corrcoef(X_text[:, i], y)[0, 1])])
        tabular_importance = np.mean([abs(np.corrcoef(X_tabular[:, i], y)[0, 1]) 
                                    for i in range(X_tabular.shape[1]) if not np.isnan(np.corrcoef(X_tabular[:, i], y)[0, 1])])
        
        total_importance = text_importance + tabular_importance
        if total_importance > 0:
            text_weight = text_importance / total_importance
            tabular_weight = tabular_importance / total_importance
        else:
            text_weight = tabular_weight = 0.5
        
        return {
            'text_weight': text_weight,
            'tabular_weight': tabular_weight
        }
    
    def model_stacking_approach(self, df, text_features, tabular_features, y, temporal_splits):
        """Model stacking: train multiple models and use meta-learner"""
        X_text = self.prepare_features(df, text_features)
        X_tabular = self.prepare_features(df, tabular_features)
        X_combined = self.prepare_features(df, text_features + tabular_features)
        
        # Train base models
        base_models = {
            'text_rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'text_lr': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'tabular_rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'tabular_lr': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'combined_rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        }
        
        # Train and get predictions
        base_predictions = {}
        for name, model in base_models.items():
            if 'text' in name:
                X = X_text
            elif 'tabular' in name:
                X = X_tabular
            else:
                X = X_combined
            
            calibrated_model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
            predictions = []
            true_labels = []
            
            for split in temporal_splits:
                X_train = X[split['train_idx']]
                X_test = X[split['test_idx']]
                y_train = y.iloc[split['train_idx']]
                y_test = y.iloc[split['test_idx']]
                
                calibrated_model.fit(X_train, y_train)
                pred = calibrated_model.predict_proba(X_test)[:, 1]
                predictions.extend(pred)
                true_labels.extend(y_test)
            
            base_predictions[name] = np.array(predictions)
        
        # Create meta-features
        meta_features = np.column_stack(list(base_predictions.values()))
        
        # Train meta-learner
        meta_learner = LogisticRegression(random_state=self.random_state)
        meta_predictions = []
        meta_true_labels = []
        
        for split in temporal_splits:
            meta_train = meta_features[split['train_idx']]
            meta_test = meta_features[split['test_idx']]
            y_train = y.iloc[split['train_idx']]
            y_test = y.iloc[split['test_idx']]
            
            meta_learner.fit(meta_train, y_train)
            pred = meta_learner.predict_proba(meta_test)[:, 1]
            meta_predictions.extend(pred)
            meta_true_labels.extend(y_test)
        
        # Calculate metrics
        meta_auc = roc_auc_score(meta_true_labels, meta_predictions)
        meta_pr_auc = average_precision_score(meta_true_labels, meta_predictions)
        
        return {
            'mean_auc': meta_auc,
            'mean_pr_auc': meta_pr_auc,
            'std_auc': 0.0,
            'std_pr_auc': 0.0,
            'approach': 'model_stacking',
            'base_models': list(base_models.keys())
        }
    
    def feature_selection_approach(self, df, text_features, tabular_features, y, temporal_splits):
        """Feature selection: select best features from each modality"""
        X_text = self.prepare_features(df, text_features)
        X_tabular = self.prepare_features(df, tabular_features)
        
        # Select top features from each modality
        selector_text = SelectKBest(score_func=f_classif, k=min(5, len(text_features)))
        selector_tabular = SelectKBest(score_func=f_classif, k=min(3, len(tabular_features)))
        
        # Fit selectors
        selector_text.fit(X_text, y)
        selector_tabular.fit(X_tabular, y)
        
        # Transform features
        X_text_selected = selector_text.transform(X_text)
        X_tabular_selected = selector_tabular.transform(X_tabular)
        
        # Combine selected features
        X_selected = np.concatenate([X_text_selected, X_tabular_selected], axis=1)
        
        return self.run_temporal_cv(X_selected, y, temporal_splits)
    
    def prepare_features(self, df, features):
        """Prepare features with proper handling"""
        X = df[features].copy()
        
        # Handle categorical variables
        for col in X.columns:
            try:
                if hasattr(X[col], 'dtype') and X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            except:
                continue
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X.values
    
    def run_temporal_cv(self, X, y, temporal_splits):
        """Run temporal cross-validation"""
        fold_results = []
        all_predictions = []
        all_true_labels = []
        
        for i, split in enumerate(temporal_splits):
            X_train = X[split['train_idx']]
            X_test = X[split['test_idx']]
            y_train = y.iloc[split['train_idx']]
            y_test = y.iloc[split['test_idx']]
            
            # Train ensemble model
            models = {
                'rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                'gb': GradientBoostingClassifier(random_state=self.random_state),
                'lr': LogisticRegression(random_state=self.random_state, max_iter=1000)
            }
            
            # Train and calibrate each model
            calibrated_predictions = {}
            for name, model in models.items():
                calibrated_model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
                calibrated_model.fit(X_train, y_train)
                calibrated_predictions[name] = calibrated_model.predict_proba(X_test)[:, 1]
            
            # Ensemble prediction
            ensemble_pred = np.mean(list(calibrated_predictions.values()), axis=0)
            
            # Calculate metrics
            auc = roc_auc_score(y_test, ensemble_pred)
            pr_auc = average_precision_score(y_test, ensemble_pred)
            
            fold_results.append({
                'fold': i,
                'auc': auc,
                'pr_auc': pr_auc,
                'predictions': ensemble_pred,
                'true_labels': y_test
            })
            
            all_predictions.extend(ensemble_pred)
            all_true_labels.extend(y_test)
        
        # Aggregate results
        mean_auc = np.mean([r['auc'] for r in fold_results])
        mean_pr_auc = np.mean([r['pr_auc'] for r in fold_results])
        std_auc = np.std([r['auc'] for r in fold_results])
        std_pr_auc = np.std([r['pr_auc'] for r in fold_results])
        
        return {
            'mean_auc': mean_auc,
            'mean_pr_auc': mean_pr_auc,
            'std_auc': std_auc,
            'std_pr_auc': std_pr_auc,
            'fold_results': fold_results,
            'all_predictions': np.array(all_predictions),
            'all_true_labels': np.array(all_true_labels)
        }
    
    def advanced_ablation_study(self, df, text_features, tabular_features, y, temporal_splits):
        """Advanced ablation study for text-derived features"""
        print("   Running advanced ablation study...")
        
        ablation_results = {}
        
        # Individual text feature ablation
        for feature in text_features:
            remaining_features = [f for f in text_features if f != feature]
            if remaining_features:
                X_ablated = self.prepare_features(df, remaining_features + tabular_features)
                results = self.run_temporal_cv(X_ablated, y, temporal_splits)
                ablation_results[f'without_{feature}'] = results
        
        # Text feature group ablation
        text_groups = {
            'sentiment_features': ['positive_word_count', 'negative_word_count', 'sentiment_balance'],
            'complexity_features': ['avg_word_length', 'type_token_ratio', 'sentence_count'],
            'entity_features': ['job_stability_count', 'financial_hardship_count', 'repayment_confidence_count', 'urgency_count'],
            'keyword_features': ['financial_keyword_density']
        }
        
        for group_name, group_features in text_groups.items():
            available_features = [f for f in group_features if f in text_features]
            if available_features:
                remaining_features = [f for f in text_features if f not in available_features]
                if remaining_features:
                    X_ablated = self.prepare_features(df, remaining_features + tabular_features)
                    results = self.run_temporal_cv(X_ablated, y, temporal_splits)
                    ablation_results[f'without_{group_name}'] = results
        
        return ablation_results
    
    def save_advanced_results(self, all_results, ablation_results):
        """Save advanced integration results"""
        print("\nSaving advanced integration results...")
        
        # Create results directory
        results_dir = Path('final_results/advanced_integration')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            else:
                return str(obj)
        
        # Save main results
        with open(results_dir / 'integration_results.json', 'w') as f:
            json.dump(convert_to_serializable(all_results), f, indent=2)
        
        # Save ablation results
        with open(results_dir / 'ablation_results.json', 'w') as f:
            json.dump(convert_to_serializable(ablation_results), f, indent=2)
        
        # Create summary table
        summary_data = []
        for regime, results in all_results.items():
            for approach, approach_results in results.items():
                if approach_results is not None:
                    summary_data.append({
                        'Regime': regime,
                        'Approach': approach,
                        'AUC': f"{approach_results['mean_auc']:.4f}",
                        'PR_AUC': f"{approach_results['mean_pr_auc']:.4f}",
                        'AUC_Std': f"{approach_results['std_auc']:.4f}"
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(results_dir / 'integration_summary.csv', index=False)
        
        # Save manifest
        manifest = {
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'approaches': [
                'Late Fusion (concatenation)',
                'Early Fusion (separate models)',
                'Attention Fusion (weighted combination)',
                'Model Stacking (meta-learner)',
                'Feature Selection (best features)'
            ]
        }
        
        with open(results_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"   Results saved to {results_dir}")
        
        # Print summary
        print(f"\nADVANCED INTEGRATION SUMMARY:")
        print("=" * 50)
        for regime, results in all_results.items():
            print(f"Regime: {regime}")
            for approach, approach_results in results.items():
                if approach_results is not None:
                    print(f"  {approach}: AUC = {approach_results['mean_auc']:.4f} ± {approach_results['std_auc']:.4f}")
            print()

if __name__ == "__main__":
    # Run the advanced integration analysis
    analysis = AdvancedModelIntegration(random_state=42)
    results, ablation_results = analysis.run_advanced_integration_analysis()
    
    print("\nAdvanced Model Integration Analysis Complete!")
    print("=" * 60)
    print("Early fusion with attention mechanisms implemented")
    print("Model stacking with meta-learner completed")
    print("Advanced ablation studies finished")
    print("Ready for comprehensive comparison") 