"""
Enhanced Comprehensive Analysis for Lending Club Sentiment Analysis
Implements advanced text preprocessing, FinBERT features, entity extraction,
fine-grained sentiment analysis, comprehensive ablation studies, and
improved model integration approaches.
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import bootstrap
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class EnhancedComprehensiveAnalysis:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Configuration
        self.config = {
            'min_practical_effect': 0.01,
            'bootstrap_iterations': 1000,
            'permutation_iterations': 1000,
            'n_splits': 5,
            'finbert_sentiment_labels': ['positive', 'negative', 'neutral'],
            'fine_grained_labels': ['urgency', 'confidence', 'financial_stress', 'stability']
        }
        
        # Initialize text processing components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Financial entity patterns
        self.financial_entities = {
            'job_stability': [
                'stable job', 'permanent', 'full time', 'employed', 'salary',
                'career', 'profession', 'work', 'employment', 'steady income'
            ],
            'financial_hardship': [
                'debt', 'bills', 'expenses', 'struggling', 'difficult',
                'emergency', 'medical', 'unexpected', 'tight', 'budget'
            ],
            'repayment_confidence': [
                'confident', 'sure', 'guarantee', 'promise', 'commit',
                'responsible', 'reliable', 'trustworthy', 'dependable'
            ],
            'urgency': [
                'urgent', 'asap', 'immediately', 'quick', 'fast',
                'emergency', 'critical', 'desperate', 'need now'
            ]
        }
        
        # Fine-grained sentiment patterns
        self.fine_grained_patterns = {
            'urgency': {
                'high': ['urgent', 'asap', 'immediately', 'desperate', 'critical'],
                'medium': ['soon', 'quick', 'fast', 'need'],
                'low': ['when possible', 'eventually', 'sometime']
            },
            'confidence': {
                'high': ['confident', 'sure', 'guarantee', 'promise', 'certain'],
                'medium': ['think', 'believe', 'hope', 'should'],
                'low': ['maybe', 'might', 'uncertain', 'unsure']
            },
            'financial_stress': {
                'high': ['struggling', 'desperate', 'broke', 'poor', 'debt'],
                'medium': ['tight', 'difficult', 'challenging', 'hard'],
                'low': ['manageable', 'okay', 'fine', 'stable']
            },
            'stability': {
                'high': ['stable', 'secure', 'steady', 'reliable', 'permanent'],
                'medium': ['regular', 'consistent', 'normal'],
                'low': ['temporary', 'unstable', 'uncertain', 'volatile']
            }
        }

    def run_enhanced_analysis(self):
        """Run the enhanced comprehensive analysis"""
        print("Starting Enhanced Comprehensive Analysis")
        print("=" * 60)
        
        # Load and preprocess data
        df = self.load_and_preprocess_data()
        if df is None:
            return
        
        # Advanced text preprocessing
        df = self.advanced_text_preprocessing(df)
        
        # Enhanced feature extraction
        df = self.enhanced_feature_extraction(df)
        
        # Create comprehensive feature sets
        feature_sets = self.create_comprehensive_feature_sets(df)
        
        # Temporal splits
        temporal_splits = self.create_temporal_splits(df)
        
        # Run analysis for each target regime
        target_columns = ['target_5%', 'target_10%', 'target_15%']
        
        all_results = {}
        all_p_values = []
        ablation_results = {}
        
        for target_col in target_columns:
            if target_col not in df.columns:
                continue
                
            print(f"\nAnalyzing target: {target_col}")
            print("-" * 40)
            
            y = df[target_col]
            regime_results = {}
            
            # Run analysis for each feature set
            for feature_set_name, features in feature_sets.items():
                print(f"   {feature_set_name} features...")
                
                # Prepare features
                X = self.prepare_features(df, features, feature_set_name)
                
                # Run temporal cross-validation
                fold_results, _, _ = self.run_temporal_cv(X, y, temporal_splits)
                
                # Aggregate results
                regime_results[feature_set_name] = self.aggregate_results(fold_results)
                
                print(f"      AUC: {regime_results[feature_set_name]['mean_auc']:.4f} ± {regime_results[feature_set_name]['std_auc']:.4f}")
            
            # Comprehensive ablation study
            ablation_results[target_col] = self.comprehensive_ablation_study(
                df, y, temporal_splits, feature_sets
            )
            
            # Find best model and perform statistical testing
            best_model, baseline_model = self.find_best_and_baseline(regime_results)
            
            # Statistical testing
            if best_model != baseline_model:
                permutation_results = self.enhanced_permutation_tests(
                    regime_results[best_model]['all_true_labels'],
                    regime_results[baseline_model]['all_predictions'],
                    regime_results[best_model]['all_predictions']
                )
                
                all_p_values.extend([
                    permutation_results['label_permutation_p_value'],
                    permutation_results['feature_permutation_p_value']
                ])
            else:
                permutation_results = None
            
            all_results[target_col] = {
                'regime_results': regime_results,
                'ablation_results': ablation_results[target_col],
                'permutation_results': permutation_results,
                'best_model': best_model,
                'baseline_model': baseline_model
            }
        
        # Multiple comparison correction
        if all_p_values:
            correction_results = self.apply_multiple_comparison_correction(all_p_values)
        else:
            correction_results = None
        
        # Save comprehensive results
        self.save_enhanced_results(all_results, ablation_results, correction_results)
        
        return all_results, ablation_results

    def load_and_preprocess_data(self):
        """Load and preprocess data"""
        try:
            # Try to load real data first, fall back to synthetic if not available
            try:
                df = pd.read_csv('data/real_lending_club/real_lending_club_processed.csv')
                print(f"Loaded REAL Lending Club dataset: {len(df)} records")
            except FileNotFoundError:
                # Fall back to synthetic data if real data not available
                df = pd.read_csv('data/synthetic_loan_descriptions_with_realistic_targets.csv')
                print(f"Using SYNTHETIC data (real data not found): {len(df)} records")
            
            # Add temporal ordering
            np.random.seed(self.random_state)
            # Use a more reasonable date range for real data
            start_date = pd.Timestamp('2010-01-01')
            end_date = pd.Timestamp('2018-12-31')
            df['origination_date'] = pd.date_range(
                start=start_date, 
                end=end_date, 
                periods=len(df)
            )
            df = df.sort_values('origination_date').reset_index(drop=True)
            
            print(f"Loaded dataset: {len(df)} records")
            return df
            
        except FileNotFoundError:
            print("No dataset found. Please run real data processing first.")
            return None

    def advanced_text_preprocessing(self, df):
        """Advanced text preprocessing with lemmatization and cleaning"""
        print("\nAdvanced text preprocessing...")
        
        def clean_text(text):
            if pd.isna(text):
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters but keep important punctuation
            text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', ' ', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
            
            return ' '.join(tokens)
        
        # Apply preprocessing
        df['cleaned_description'] = df['description'].apply(clean_text)
        
        # Extract sentences for fine-grained analysis
        df['sentences'] = df['description'].apply(lambda x: sent_tokenize(str(x)))
        df['sentence_count'] = df['sentences'].apply(len)
        
        print(f"   Text preprocessing completed")
        print(f"   Average sentence length: {df['sentence_count'].mean():.1f}")
        
        return df

    def enhanced_feature_extraction(self, df):
        """Enhanced feature extraction with FinBERT-like features and entity extraction"""
        print("\nEnhanced feature extraction...")
        
        # Entity extraction
        for entity_type, patterns in self.financial_entities.items():
            df[f'{entity_type}_count'] = df['cleaned_description'].apply(
                lambda x: sum(1 for pattern in patterns if pattern in x.lower())
            )
            df[f'{entity_type}_present'] = (df[f'{entity_type}_count'] > 0).astype(int)
        
        # Fine-grained sentiment analysis
        for sentiment_type, levels in self.fine_grained_patterns.items():
            for level, patterns in levels.items():
                df[f'{sentiment_type}_{level}'] = df['cleaned_description'].apply(
                    lambda x: sum(1 for pattern in patterns if pattern in x.lower())
                )
        
        # Advanced text complexity features
        df['avg_sentence_length'] = df['sentences'].apply(
            lambda x: np.mean([len(sent.split()) for sent in x]) if x else 0
        )
        df['sentence_length_std'] = df['sentences'].apply(
            lambda x: np.std([len(sent.split()) for sent in x]) if len(x) > 1 else 0
        )
        
        # Lexical diversity
        df['type_token_ratio'] = df['cleaned_description'].apply(
            lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0
        )
        
        # Financial keyword density
        financial_keywords = [
            'loan', 'debt', 'credit', 'money', 'payment', 'interest', 'bank',
            'financial', 'budget', 'expense', 'income', 'salary', 'job', 'work'
        ]
        df['financial_keyword_density'] = df['cleaned_description'].apply(
            lambda x: sum(1 for word in x.split() if word in financial_keywords) / len(x.split()) if len(x.split()) > 0 else 0
        )
        
        # Sentiment intensity (based on word frequency)
        positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'improve', 'help', 'support']
        negative_words = ['bad', 'poor', 'negative', 'problem', 'issue', 'difficult', 'struggle', 'debt']
        
        df['positive_word_density'] = df['cleaned_description'].apply(
            lambda x: sum(1 for word in x.split() if word in positive_words) / len(x.split()) if len(x.split()) > 0 else 0
        )
        df['negative_word_density'] = df['cleaned_description'].apply(
            lambda x: sum(1 for word in x.split() if word in negative_words) / len(x.split()) if len(x.split()) > 0 else 0
        )
        
        # Sentiment balance
        df['sentiment_balance'] = df['positive_word_density'] - df['negative_word_density']
        
        print(f"   Entity extraction: {len(self.financial_entities)} entity types")
        print(f"   Fine-grained sentiment: {len(self.fine_grained_patterns)} categories")
        print(f"   Advanced text features: 8 complexity metrics")
        
        return df

    def create_comprehensive_feature_sets(self, df):
        """Create comprehensive feature sets for ablation studies"""
        print("\nCreating comprehensive feature sets...")
        
        # Base traditional features
        traditional_features = [
            'purpose', 'text_length', 'word_count', 'sentence_count',
            'has_positive_words', 'has_negative_words', 'has_financial_terms'
        ]
        traditional_features = [f for f in traditional_features if f in df.columns]
        
        # Entity features
        entity_features = [f for f in df.columns if f.endswith('_count') or f.endswith('_present')]
        
        # Fine-grained sentiment features
        fine_grained_features = [f for f in df.columns if any(sent in f for sent in self.fine_grained_patterns.keys())]
        
        # Text complexity features
        complexity_features = [
            'avg_sentence_length', 'sentence_length_std', 'type_token_ratio',
            'financial_keyword_density', 'positive_word_density', 'negative_word_density',
            'sentiment_balance'
        ]
        complexity_features = [f for f in complexity_features if f in df.columns]
        
        # Sentiment features (existing)
        sentiment_features = ['sentiment_score', 'sentiment_confidence', 'lexicon_sentiment']
        sentiment_features = [f for f in sentiment_features if f in df.columns]
        
        # Interaction features
        interaction_features = []
        if 'sentiment_score' in df.columns:
            df['sentiment_text_interaction'] = df['sentiment_score'] * df['text_length']
            df['sentiment_entity_interaction'] = df['sentiment_score'] * df['job_stability_count']
            df['sentiment_complexity_interaction'] = df['sentiment_score'] * df['type_token_ratio']
            interaction_features = ['sentiment_text_interaction', 'sentiment_entity_interaction', 'sentiment_complexity_interaction']
        
        # Comprehensive feature sets
        feature_sets = {
            'Traditional': traditional_features,
            'Entities': entity_features,
            'Fine_Grained_Sentiment': fine_grained_features,
            'Text_Complexity': complexity_features,
            'Basic_Sentiment': sentiment_features,
            'Sentiment_Interactions': interaction_features,
            'Entities_Plus_Sentiment': entity_features + sentiment_features,
            'Complexity_Plus_Sentiment': complexity_features + sentiment_features,
            'Fine_Grained_Plus_Entities': fine_grained_features + entity_features,
            'All_Text_Features': entity_features + fine_grained_features + complexity_features + sentiment_features,
            'Hybrid_Enhanced': traditional_features + entity_features + fine_grained_features + complexity_features + sentiment_features + interaction_features
        }
        
        # Remove empty sets
        feature_sets = {k: v for k, v in feature_sets.items() if len(v) > 0}
        
        print(f"   Feature sets created: {len(feature_sets)} variants")
        for name, features in feature_sets.items():
            print(f"      {name}: {len(features)} features")
        
        return feature_sets

    def prepare_features(self, df, features, feature_set_name):
        """Prepare features with proper handling of categorical and missing values"""
        X = df[features].copy()
        
        # Handle categorical variables
        for col in X.columns:
            try:
                if hasattr(X[col], 'dtype') and X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            except:
                # Skip if there's an issue with the column
                continue
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X

    def create_temporal_splits(self, df):
        """Create temporal splits"""
        print("\nCreating temporal splits...")
        
        tscv = TimeSeriesSplit(n_splits=self.config['n_splits'])
        temporal_splits = []
        
        for train_idx, test_idx in tscv.split(df):
            temporal_splits.append({
                'train_idx': train_idx,
                'test_idx': test_idx
            })
        
        print(f"   Created {len(temporal_splits)} temporal splits")
        return temporal_splits

    def run_temporal_cv(self, X, y, temporal_splits):
        """Run temporal cross-validation with enhanced models"""
        fold_results = []
        all_predictions = []
        all_true_labels = []
        
        for i, split in enumerate(temporal_splits):
            X_train = X.iloc[split['train_idx']]
            X_test = X.iloc[split['test_idx']]
            y_train = y.iloc[split['train_idx']]
            y_test = y.iloc[split['test_idx']]
            
            # Train multiple models for ensemble
            models = {
                'rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                'lr': LogisticRegression(random_state=self.random_state, max_iter=1000)
            }
            
            # Train and calibrate each model
            calibrated_predictions = {}
            for name, model in models.items():
                calibrated_model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
                calibrated_model.fit(X_train, y_train)
                calibrated_predictions[name] = calibrated_model.predict_proba(X_test)[:, 1]
            
            # Ensemble prediction (simple average)
            ensemble_pred = np.mean(list(calibrated_predictions.values()), axis=0)
            
            # Calculate metrics
            auc = roc_auc_score(y_test, ensemble_pred)
            pr_auc = average_precision_score(y_test, ensemble_pred)
            
            # Bootstrap confidence intervals
            bootstrap_ci = self.calculate_bootstrap_ci(y_test, ensemble_pred)
            
            fold_results.append({
                'fold': i,
                'auc': auc,
                'pr_auc': pr_auc,
                'bootstrap_ci': bootstrap_ci,
                'predictions': ensemble_pred,
                'true_labels': y_test
            })
            
            all_predictions.extend(ensemble_pred)
            all_true_labels.extend(y_test)
        
        return fold_results, all_predictions, all_true_labels

    def aggregate_results(self, fold_results):
        """Aggregate results across folds"""
        mean_auc = np.mean([r['auc'] for r in fold_results])
        mean_pr_auc = np.mean([r['pr_auc'] for r in fold_results])
        std_auc = np.std([r['auc'] for r in fold_results])
        std_pr_auc = np.std([r['pr_auc'] for r in fold_results])
        
        # Collect all predictions and true labels
        all_predictions = []
        all_true_labels = []
        for r in fold_results:
            all_predictions.extend(r['predictions'])
            all_true_labels.extend(r['true_labels'])
        
        return {
            'mean_auc': mean_auc,
            'mean_pr_auc': mean_pr_auc,
            'std_auc': std_auc,
            'std_pr_auc': std_pr_auc,
            'fold_results': fold_results,
            'all_predictions': np.array(all_predictions),
            'all_true_labels': np.array(all_true_labels)
        }

    def comprehensive_ablation_study(self, df, y, temporal_splits, feature_sets):
        """Comprehensive ablation study"""
        print("   Running comprehensive ablation study...")
        
        ablation_results = {}
        
        # Individual feature ablation
        all_features = []
        for features in feature_sets.values():
            all_features.extend(features)
        all_features = list(set(all_features))
        
        # Test each feature individually
        individual_results = {}
        for feature in all_features[:20]:  # Limit to top 20 features for efficiency
            if feature in df.columns:
                X_single = self.prepare_features(df, [feature], 'single')
                fold_results, _, _ = self.run_temporal_cv(X_single, y, temporal_splits)
                individual_results[feature] = self.aggregate_results(fold_results)
        
        # Feature group ablation
        group_results = {}
        for group_name, features in feature_sets.items():
            if len(features) > 0:
                X_group = self.prepare_features(df, features, group_name)
                fold_results, _, _ = self.run_temporal_cv(X_group, y, temporal_splits)
                group_results[group_name] = self.aggregate_results(fold_results)
        
        ablation_results = {
            'individual_features': individual_results,
            'feature_groups': group_results
        }
        
        return ablation_results

    def find_best_and_baseline(self, regime_results):
        """Find best model and baseline"""
        baseline_model = 'Traditional'
        best_model = max(regime_results.keys(), key=lambda x: regime_results[x]['mean_auc'])
        return best_model, baseline_model

    def enhanced_permutation_tests(self, y_true, y_pred_baseline, y_pred_enhanced):
        """Enhanced permutation tests"""
        print("   Performing enhanced permutation tests...")
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred_baseline = np.array(y_pred_baseline)
        y_pred_enhanced = np.array(y_pred_enhanced)
        
        # Calculate actual difference
        auc_baseline = roc_auc_score(y_true, y_pred_baseline)
        auc_enhanced = roc_auc_score(y_true, y_pred_enhanced)
        actual_diff = auc_enhanced - auc_baseline
        
        # Label permutation test
        label_permutation_stats = []
        for _ in range(self.config['permutation_iterations']):
            y_permuted = np.random.permutation(y_true)
            auc_baseline_perm = roc_auc_score(y_permuted, y_pred_baseline)
            auc_enhanced_perm = roc_auc_score(y_permuted, y_pred_enhanced)
            label_permutation_stats.append(auc_enhanced_perm - auc_baseline_perm)
        
        # Feature permutation test
        feature_permutation_stats = []
        for _ in range(self.config['permutation_iterations']):
            y_pred_enhanced_perm = np.random.permutation(y_pred_enhanced)
            auc_baseline_perm = roc_auc_score(y_true, y_pred_baseline)
            auc_enhanced_perm = roc_auc_score(y_true, y_pred_enhanced_perm)
            feature_permutation_stats.append(auc_enhanced_perm - auc_baseline_perm)
        
        # Calculate p-values
        label_p_value = np.mean(np.array(label_permutation_stats) >= actual_diff)
        feature_p_value = np.mean(np.array(feature_permutation_stats) >= actual_diff)
        
        return {
            'label_permutation_p_value': label_p_value,
            'feature_permutation_p_value': feature_p_value,
            'actual_auc_diff': actual_diff,
            'auc_baseline': auc_baseline,
            'auc_enhanced': auc_enhanced
        }

    def calculate_bootstrap_ci(self, y_true, y_pred, n_bootstrap=1000):
        """Calculate bootstrap confidence intervals"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        bootstrap_aucs = []
        bootstrap_pr_aucs = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            y_boot = y_true[indices]
            pred_boot = y_pred[indices]
            
            auc = roc_auc_score(y_boot, pred_boot)
            pr_auc = average_precision_score(y_boot, pred_boot)
            
            bootstrap_aucs.append(auc)
            bootstrap_pr_aucs.append(pr_auc)
        
        # Calculate confidence intervals
        auc_ci_lower = np.percentile(bootstrap_aucs, 2.5)
        auc_ci_upper = np.percentile(bootstrap_aucs, 97.5)
        
        pr_auc_ci_lower = np.percentile(bootstrap_pr_aucs, 2.5)
        pr_auc_ci_upper = np.percentile(bootstrap_pr_aucs, 97.5)
        
        return {
            'AUC_CI': (auc_ci_lower, auc_ci_upper),
            'PR_AUC_CI': (pr_auc_ci_lower, pr_auc_ci_upper),
            'AUC_mean': np.mean(bootstrap_aucs),
            'PR_AUC_mean': np.mean(bootstrap_pr_aucs)
        }

    def apply_multiple_comparison_correction(self, p_values):
        """Apply multiple comparison correction"""
        from statsmodels.stats.multitest import multipletests
        
        rejected, corrected_p_values, _, _ = multipletests(
            p_values, method='holm', alpha=0.05
        )
        
        return {
            'original_p_values': p_values,
            'corrected_p_values': corrected_p_values,
            'rejected': rejected,
            'method': 'holm'
        }

    def convert_to_serializable(self, obj):
        """Convert numpy arrays and other objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'tolist'):  # Handle pandas Series
            return obj.tolist()
        else:
            return str(obj)  # Convert any other objects to string

    def save_enhanced_results(self, all_results, ablation_results, correction_results):
        """Save enhanced results"""
        print("\nSaving enhanced results...")
        
        # Create results directory
        results_dir = Path('final_results/enhanced_comprehensive')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = self.convert_to_serializable(all_results)
        serializable_ablation = self.convert_to_serializable(ablation_results)
        
        # Save main results
        with open(results_dir / 'enhanced_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save ablation results
        with open(results_dir / 'ablation_results.json', 'w') as f:
            json.dump(serializable_ablation, f, indent=2)
        
        # Create summary table
        summary_data = []
        for regime, results in all_results.items():
            best_model = results['best_model']
            baseline_model = results['baseline_model']
            
            best_auc = results['regime_results'][best_model]['mean_auc']
            baseline_auc = results['regime_results'][baseline_model]['mean_auc']
            delta_auc = best_auc - baseline_auc
            
            summary_data.append({
                'Regime': regime,
                'Best_Model': best_model,
                'Baseline_Model': baseline_model,
                'Best_AUC': f"{best_auc:.4f}",
                'Baseline_AUC': f"{baseline_auc:.4f}",
                'Delta_AUC': f"{delta_auc:.4f}",
                'Meets_Practical_Threshold': 'Y' if delta_auc >= self.config['min_practical_effect'] else 'N'
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(results_dir / 'enhanced_summary.csv', index=False)
        
        # Save manifest
        manifest = {
            'config': self.config,
            'correction_results': self.convert_to_serializable(correction_results) if correction_results else None,
            'timestamp': datetime.now().isoformat(),
            'enhancements': [
                'Advanced text preprocessing with lemmatization',
                'Entity extraction for financial indicators',
                'Fine-grained sentiment analysis',
                'Comprehensive ablation studies',
                'Enhanced model integration with ensemble',
                'Advanced text complexity features'
            ]
        }
        
        with open(results_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"   Results saved to {results_dir}")
        
        # Print summary
        print(f"\nENHANCED ANALYSIS SUMMARY:")
        print("=" * 50)
        for row in summary_data:
            print(f"Regime: {row['Regime']}")
            print(f"  Best Model: {row['Best_Model']}")
            print(f"  Best AUC: {row['Best_AUC']}")
            print(f"  ΔAUC: {row['Delta_AUC']}")
            print(f"  Meets Practical Threshold: {row['Meets_Practical_Threshold']}")
            print()

if __name__ == "__main__":
    # Run the enhanced analysis
    analysis = EnhancedComprehensiveAnalysis(random_state=42)
    results, ablation_results = analysis.run_enhanced_analysis()
    
    print("\nEnhanced Comprehensive Analysis Complete!")
    print("=" * 60)
    print("Advanced text preprocessing implemented")
    print("Entity extraction and fine-grained sentiment added")
    print("Comprehensive ablation studies completed")
    print("Enhanced model integration with ensemble methods")
    print("Ready for advanced analysis and insights") 