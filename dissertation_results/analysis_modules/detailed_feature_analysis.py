"""
Detailed Feature Analysis for Lending Club Sentiment Analysis
Implements comprehensive SHAP analysis, text feature breakdown, error analysis,
case studies, TF-IDF baseline comparison, and misclassification examples.
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import SHAP for feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

class DetailedFeatureAnalysis:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Configuration
        self.config = {
            'n_splits': 5,
            'shap_samples': 1000,  # For SHAP analysis
            'tfidf_max_features': 1000,
            'case_study_samples': 20
        }
        
        # Initialize SHAP explainer
        self.shap_explainer = None
        
    def run_comprehensive_analysis(self):
        """Run comprehensive detailed feature analysis"""
        print("Detailed Feature Analysis")
        print("=" * 60)
        
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
            return df
            
        except FileNotFoundError:
            print("No dataset found. Please run real data processing first.")
            return None
    
    def create_enhanced_features(self, df):
        """Create enhanced features for detailed analysis"""
        print("Creating enhanced features for detailed analysis...")
        
        # Text preprocessing
        df['cleaned_description'] = df['description'].apply(self.clean_text)
        
        # Sentiment features (detailed breakdown)
        df = self.create_detailed_sentiment_features(df)
        
        # Text structure features
        df = self.create_text_structure_features(df)
        
        # Financial entity features
        df = self.create_financial_entity_features(df)
        
        # Language style features
        df = self.create_language_style_features(df)
        
        print(f"   Enhanced features created: {len([col for col in df.columns if col not in ['description', 'origination_date', 'target_5%', 'target_10%', 'target_15%']])} features")
        
        return df
    
    def clean_text(self, text):
        """Clean text for processing"""
        if pd.isna(text):
            return ""
        return text.lower().strip()
    
    def create_detailed_sentiment_features(self, df):
        """Create detailed sentiment features"""
        # Basic sentiment
        positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'improve', 'help', 'support']
        negative_words = ['bad', 'poor', 'negative', 'problem', 'issue', 'difficult', 'struggle', 'debt']
        
        # Count positive and negative words
        df['positive_word_count'] = df['description'].apply(
            lambda x: sum(1 for word in x.lower().split() if word in positive_words)
        )
        df['negative_word_count'] = df['description'].apply(
            lambda x: sum(1 for word in x.lower().split() if word in negative_words)
        )
        
        # Sentiment balance
        df['sentiment_balance'] = df['positive_word_count'] - df['negative_word_count']
        
        # Sentiment intensity (max vs mean)
        df['sentiment_intensity_max'] = df['description'].apply(
            lambda x: max([sum(1 for word in sent.lower().split() if word in positive_words) - 
                          sum(1 for word in sent.lower().split() if word in negative_words)
                          for sent in x.split('.') if sent.strip()])
        )
        df['sentiment_intensity_mean'] = df['description'].apply(
            lambda x: np.mean([sum(1 for word in sent.lower().split() if word in positive_words) - 
                              sum(1 for word in sent.lower().split() if word in negative_words)
                              for sent in x.split('.') if sent.strip()])
        )
        
        # Sentiment variance
        df['sentiment_variance'] = df['description'].apply(
            lambda x: np.var([sum(1 for word in sent.lower().split() if word in positive_words) - 
                             sum(1 for word in sent.lower().split() if word in negative_words)
                             for sent in x.split('.') if sent.strip()])
        )
        
        return df
    
    def create_text_structure_features(self, df):
        """Create text structure features"""
        # Sentence-level features
        df['sentence_count'] = df['description'].apply(lambda x: len([s for s in x.split('.') if s.strip()]))
        df['avg_sentence_length'] = df['description'].apply(
            lambda x: np.mean([len(s.split()) for s in x.split('.') if s.strip()]) if len([s for s in x.split('.') if s.strip()]) > 0 else 0
        )
        df['sentence_length_std'] = df['description'].apply(
            lambda x: np.std([len(s.split()) for s in x.split('.') if s.strip()]) if len([s for s in x.split('.') if s.strip()]) > 1 else 0
        )
        
        # Word-level features
        df['word_count'] = df['description'].apply(lambda x: len(x.split()))
        df['avg_word_length'] = df['description'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )
        df['unique_word_ratio'] = df['description'].apply(
            lambda x: len(set(x.lower().split())) / len(x.split()) if x.split() else 0
        )
        
        return df
    
    def create_financial_entity_features(self, df):
        """Create financial entity features"""
        # Financial keywords by category
        financial_categories = {
            'job_stability': ['stable job', 'permanent', 'full time', 'employed', 'salary', 'career'],
            'financial_hardship': ['debt', 'bills', 'expenses', 'struggling', 'difficult', 'emergency'],
            'repayment_confidence': ['confident', 'sure', 'guarantee', 'promise', 'commit', 'responsible'],
            'loan_purpose': ['home', 'car', 'education', 'business', 'medical', 'consolidation'],
            'financial_terms': ['loan', 'credit', 'payment', 'interest', 'budget', 'income']
        }
        
        for category, keywords in financial_categories.items():
            df[f'{category}_count'] = df['description'].apply(
                lambda x: sum(1 for keyword in keywords if keyword in x.lower())
            )
        
        return df
    
    def create_language_style_features(self, df):
        """Create language style features"""
        # Formality indicators
        formal_words = ['therefore', 'furthermore', 'moreover', 'consequently', 'subsequently']
        informal_words = ['gonna', 'wanna', 'gotta', 'yeah', 'okay']
        
        df['formal_language_count'] = df['description'].apply(
            lambda x: sum(1 for word in x.lower().split() if word in formal_words)
        )
        df['informal_language_count'] = df['description'].apply(
            lambda x: sum(1 for word in x.lower().split() if word in informal_words)
        )
        
        # Language complexity
        df['avg_word_length'] = df['description'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )
        df['type_token_ratio'] = df['description'].apply(
            lambda x: len(set(x.lower().split())) / len(x.split()) if x.split() else 0
        )
        
        return df
    
    def run_tfidf_baseline_comparison(self, df):
        """Run TF-IDF baseline comparison"""
        print("Running TF-IDF baseline comparison...")
        
        # Create TF-IDF features
        tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config['tfidf_max_features'],
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        tfidf_features = tfidf_vectorizer.fit_transform(df['cleaned_description'])
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # Run comparison for each target
        results = {}
        target_columns = ['target_5%', 'target_10%', 'target_15%']
        
        for target_col in target_columns:
            if target_col not in df.columns:
                continue
            
            print(f"   Comparing {target_col}...")
            y = df[target_col]
            
            # Sentiment features
            sentiment_features = [
                'positive_word_count', 'negative_word_count', 'sentiment_balance',
                'sentiment_intensity_max', 'sentiment_intensity_mean', 'sentiment_variance'
            ]
            sentiment_features = [f for f in sentiment_features if f in df.columns]
            
            # Compare performance
            sentiment_auc = self.evaluate_features(df[sentiment_features], y)
            tfidf_auc = self.evaluate_features(tfidf_df, y)
            
            results[target_col] = {
                'sentiment_auc': sentiment_auc,
                'tfidf_auc': tfidf_auc,
                'improvement': sentiment_auc - tfidf_auc,
                'sentiment_features': sentiment_features,
                'tfidf_features': list(tfidf_df.columns)
            }
            
            print(f"      Sentiment AUC: {sentiment_auc:.4f}")
            print(f"      TF-IDF AUC: {tfidf_auc:.4f}")
            print(f"      Improvement: {sentiment_auc - tfidf_auc:.4f}")
        
        return results
    
    def evaluate_features(self, X, y):
        """Evaluate features using temporal cross-validation"""
        tscv = TimeSeriesSplit(n_splits=self.config['n_splits'])
        aucs = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            # Handle missing values
            X_train = X_train.fillna(X_train.mean())
            X_test = X_test.fillna(X_train.mean())
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred)
            aucs.append(auc)
        
        return np.mean(aucs)
    
    def run_detailed_shap_analysis(self, df):
        """Run detailed SHAP analysis"""
        if not SHAP_AVAILABLE:
            print("SHAP not available. Skipping SHAP analysis.")
            return None
        
        print("Running detailed SHAP analysis...")
        
        # Prepare features for SHAP - use only numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_features if col not in 
                          ['target_5%', 'target_10%', 'target_15%']]
        
        X = df[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Ensure all features are numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(X.mean())
        
        # Train model for SHAP
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        model.fit(X, df['target_10%'])  # Use 10% target for analysis
        
        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        sample_size = min(self.config['shap_samples'], len(X))
        X_sample = X.iloc[:sample_size]
        shap_values = self.shap_explainer.shap_values(X_sample)
        
        # Feature importance analysis
        feature_importance = self.analyze_feature_importance(X_sample, shap_values, X_sample.columns.tolist())
        
        # Text feature breakdown
        text_feature_breakdown = self.analyze_text_feature_breakdown(X_sample, shap_values, X_sample.columns.tolist())
        
        # Create SHAP visualizations
        self.create_shap_visualizations(X_sample, shap_values, X_sample.columns.tolist())
        
        return {
            'feature_importance': feature_importance,
            'text_feature_breakdown': text_feature_breakdown,
            'shap_values': shap_values,
            'feature_columns': X_sample.columns.tolist()
        }
    
    def analyze_feature_importance(self, X, shap_values, feature_columns):
        """Analyze feature importance from SHAP values"""
        # Calculate mean absolute SHAP values
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': mean_shap_values
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def analyze_text_feature_breakdown(self, X, shap_values, feature_columns):
        """Analyze breakdown of text features"""
        # Categorize features
        sentiment_features = [f for f in feature_columns if 'sentiment' in f.lower()]
        structure_features = [f for f in feature_columns if any(x in f.lower() for x in ['sentence', 'word', 'length'])]
        entity_features = [f for f in feature_columns if any(x in f.lower() for x in ['job', 'financial', 'repayment', 'loan'])]
        style_features = [f for f in feature_columns if any(x in f.lower() for x in ['formal', 'informal', 'complexity'])]
        
        # Calculate importance by category
        categories = {
            'sentiment': sentiment_features,
            'structure': structure_features,
            'entity': entity_features,
            'style': style_features
        }
        
        breakdown = {}
        for category, features in categories.items():
            if features:
                feature_indices = [feature_columns.index(f) for f in features]
                category_importance = np.mean([np.mean(np.abs(shap_values[:, i])) for i in feature_indices])
                breakdown[category] = {
                    'features': features,
                    'importance': category_importance,
                    'top_feature': max(features, key=lambda x: np.mean(np.abs(shap_values[:, feature_columns.index(x)])))
                }
        
        return breakdown
    
    def create_shap_visualizations(self, X, shap_values, feature_columns):
        """Create SHAP visualizations"""
        # Create output directory
        output_dir = Path('final_results/detailed_analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature importance plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_columns, show=False)
        plt.title('SHAP Feature Importance Summary')
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance bar plot
        plt.figure(figsize=(12, 8))
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        top_features_idx = np.argsort(mean_shap_values)[-15:]  # Top 15 features
        
        plt.barh(range(len(top_features_idx)), mean_shap_values[top_features_idx])
        plt.yticks(range(len(top_features_idx)), [feature_columns[i] for i in top_features_idx])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Top 15 Features by SHAP Importance')
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_top_features.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_error_analysis_with_case_studies(self, df):
        """Run error analysis with case studies"""
        print("Running error analysis with case studies...")
        
        # Prepare features
        feature_columns = [col for col in df.columns if col not in 
                          ['description', 'cleaned_description', 'origination_date', 'target_5%', 'target_10%', 'target_15%']]
        
        X = df[feature_columns].copy()
        
        # Handle categorical variables and missing values
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        X = X.fillna(X.mean())
        
        # Train models
        tabular_model = self.train_tabular_model(df)
        hybrid_model = self.train_hybrid_model(X, df['target_10%'])
        
        # Analyze errors
        error_analysis = self.analyze_model_errors(df, tabular_model, hybrid_model)
        
        # Generate case studies
        case_studies = self.generate_case_studies(df, tabular_model, hybrid_model)
        
        return {
            'error_analysis': error_analysis,
            'case_studies': case_studies
        }
    
    def train_tabular_model(self, df):
        """Train tabular-only model"""
        tabular_features = ['purpose', 'text_length', 'word_count', 'has_positive_words', 'has_negative_words', 'has_financial_terms']
        tabular_features = [f for f in tabular_features if f in df.columns]
        
        X_tabular = df[tabular_features].copy()
        
        # Handle categorical variables
        for col in X_tabular.columns:
            if X_tabular[col].dtype == 'object':
                le = LabelEncoder()
                X_tabular[col] = le.fit_transform(X_tabular[col].astype(str))
        
        X_tabular = X_tabular.fillna(X_tabular.mean())
        
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        model.fit(X_tabular, df['target_10%'])
        
        return model
    
    def train_hybrid_model(self, X, y):
        """Train hybrid model"""
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        model.fit(X, y)
        return model
    
    def analyze_model_errors(self, df, tabular_model, hybrid_model):
        """Analyze model errors"""
        # Prepare features
        tabular_features = ['purpose', 'text_length', 'word_count', 'has_positive_words', 'has_negative_words', 'has_financial_terms']
        tabular_features = [f for f in tabular_features if f in df.columns]
        
        X_tabular = df[tabular_features].copy()
        for col in X_tabular.columns:
            if X_tabular[col].dtype == 'object':
                le = LabelEncoder()
                X_tabular[col] = le.fit_transform(X_tabular[col].astype(str))
        X_tabular = X_tabular.fillna(X_tabular.mean())
        
        feature_columns = [col for col in df.columns if col not in 
                          ['description', 'cleaned_description', 'origination_date', 'target_5%', 'target_10%', 'target_15%']]
        X_hybrid = df[feature_columns].copy()
        for col in X_hybrid.columns:
            if X_hybrid[col].dtype == 'object':
                le = LabelEncoder()
                X_hybrid[col] = le.fit_transform(X_hybrid[col].astype(str))
        X_hybrid = X_hybrid.fillna(X_hybrid.mean())
        
        # Get predictions
        tabular_pred = tabular_model.predict_proba(X_tabular)[:, 1]
        hybrid_pred = hybrid_model.predict_proba(X_hybrid)[:, 1]
        
        # Analyze errors
        y_true = df['target_10%']
        
        # Cases where hybrid improves over tabular
        hybrid_improvements = (hybrid_pred > tabular_pred) & (y_true == 1)
        hybrid_worsens = (hybrid_pred < tabular_pred) & (y_true == 0)
        
        error_analysis = {
            'hybrid_improvements_count': sum(hybrid_improvements),
            'hybrid_worsens_count': sum(hybrid_worsens),
            'improvement_rate': sum(hybrid_improvements) / len(df),
            'worsen_rate': sum(hybrid_worsens) / len(df),
            'tabular_auc': roc_auc_score(y_true, tabular_pred),
            'hybrid_auc': roc_auc_score(y_true, hybrid_pred),
            'improvement_auc': roc_auc_score(y_true, hybrid_pred) - roc_auc_score(y_true, tabular_pred)
        }
        
        return error_analysis
    
    def generate_case_studies(self, df, tabular_model, hybrid_model):
        """Generate case studies of model performance"""
        print("   Generating case studies...")
        
        # Prepare features
        tabular_features = ['purpose', 'text_length', 'word_count', 'has_positive_words', 'has_negative_words', 'has_financial_terms']
        tabular_features = [f for f in tabular_features if f in df.columns]
        
        X_tabular = df[tabular_features].copy()
        for col in X_tabular.columns:
            if X_tabular[col].dtype == 'object':
                le = LabelEncoder()
                X_tabular[col] = le.fit_transform(X_tabular[col].astype(str))
        X_tabular = X_tabular.fillna(X_tabular.mean())
        
        feature_columns = [col for col in df.columns if col not in 
                          ['description', 'cleaned_description', 'origination_date', 'target_5%', 'target_10%', 'target_15%']]
        X_hybrid = df[feature_columns].copy()
        for col in X_hybrid.columns:
            if X_hybrid[col].dtype == 'object':
                le = LabelEncoder()
                X_hybrid[col] = le.fit_transform(X_hybrid[col].astype(str))
        X_hybrid = X_hybrid.fillna(X_hybrid.mean())
        
        # Get predictions
        tabular_pred = tabular_model.predict_proba(X_tabular)[:, 1]
        hybrid_pred = hybrid_model.predict_proba(X_hybrid)[:, 1]
        y_true = df['target_10%']
        
        # Find interesting cases
        case_studies = []
        
        # Case 1: Hybrid improves prediction for thin files with strong narratives
        thin_file_mask = (df['word_count'] < df['word_count'].median()) & (df['sentiment_balance'] > 0)
        improvement_cases = (hybrid_pred > tabular_pred) & (y_true == 1) & thin_file_mask
        
        if sum(improvement_cases) > 0:
            case_idx = df[improvement_cases].index[0]
            case_studies.append({
                'type': 'thin_file_improvement',
                'description': df.loc[case_idx, 'description'],
                'tabular_pred': tabular_pred[case_idx],
                'hybrid_pred': hybrid_pred[case_idx],
                'true_label': y_true[case_idx],
                'word_count': df.loc[case_idx, 'word_count'],
                'sentiment_balance': df.loc[case_idx, 'sentiment_balance']
            })
        
        # Case 2: Hybrid overfits to formal language
        formal_mask = df['formal_language_count'] > df['formal_language_count'].quantile(0.8)
        overfit_cases = (hybrid_pred > tabular_pred) & (y_true == 0) & formal_mask
        
        if sum(overfit_cases) > 0:
            case_idx = df[overfit_cases].index[0]
            case_studies.append({
                'type': 'formal_language_overfit',
                'description': df.loc[case_idx, 'description'],
                'tabular_pred': tabular_pred[case_idx],
                'hybrid_pred': hybrid_pred[case_idx],
                'true_label': y_true[case_idx],
                'formal_language_count': df.loc[case_idx, 'formal_language_count']
            })
        
        # Case 3: Hybrid corrects tabular errors for strong narratives
        strong_narrative_mask = (df['sentiment_balance'] > df['sentiment_balance'].quantile(0.8)) & (y_true == 1)
        correction_cases = (tabular_pred < 0.5) & (hybrid_pred > 0.5) & strong_narrative_mask
        
        if sum(correction_cases) > 0:
            case_idx = df[correction_cases].index[0]
            case_studies.append({
                'type': 'narrative_correction',
                'description': df.loc[case_idx, 'description'],
                'tabular_pred': tabular_pred[case_idx],
                'hybrid_pred': hybrid_pred[case_idx],
                'true_label': y_true[case_idx],
                'sentiment_balance': df.loc[case_idx, 'sentiment_balance']
            })
        
        return case_studies
    
    def generate_misclassification_examples(self, df):
        """Generate misclassification examples table"""
        print("Generating misclassification examples...")
        
        # Prepare features
        feature_columns = [col for col in df.columns if col not in 
                          ['description', 'cleaned_description', 'origination_date', 'target_5%', 'target_10%', 'target_15%']]
        
        X = df[feature_columns].copy()
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        X = X.fillna(X.mean())
        
        # Train hybrid model
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        model.fit(X, df['target_10%'])
        
        # Get predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        y_true = df['target_10%']
        
        # Find misclassifications
        misclassifications = y_pred != y_true
        
        # Sample misclassification examples
        misclassified_indices = df[misclassifications].index[:self.config['case_study_samples']]
        
        examples = []
        for idx in misclassified_indices:
            examples.append({
                'description': df.loc[idx, 'description'][:200] + '...' if len(df.loc[idx, 'description']) > 200 else df.loc[idx, 'description'],
                'true_label': int(y_true[idx]),
                'predicted_label': int(y_pred[idx]),
                'prediction_probability': float(y_pred_proba[idx]),
                'word_count': int(df.loc[idx, 'word_count']),
                'sentiment_balance': float(df.loc[idx, 'sentiment_balance']),
                'formal_language_count': int(df.loc[idx, 'formal_language_count']),
                'misclassification_type': 'False Positive' if y_pred[idx] == 1 and y_true[idx] == 0 else 'False Negative'
            })
        
        return examples
    
    def save_detailed_results(self, tfidf_results, shap_results, error_results, misclassification_examples):
        """Save detailed analysis results"""
        print("Saving detailed analysis results...")
        
        # Create output directory
        output_dir = Path('final_results/detailed_analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save TF-IDF comparison results
        with open(output_dir / 'tfidf_comparison.json', 'w') as f:
            json.dump(tfidf_results, f, indent=2, default=str)
        
        # Save SHAP results
        if shap_results:
            with open(output_dir / 'shap_analysis.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                shap_data = {
                    'feature_importance': shap_results['feature_importance'].to_dict('records'),
                    'text_feature_breakdown': shap_results['text_feature_breakdown'],
                    'feature_columns': shap_results['feature_columns']
                }
                json.dump(shap_data, f, indent=2)
        
        # Save error analysis
        with open(output_dir / 'error_analysis.json', 'w') as f:
            json.dump(error_results, f, indent=2, default=str)
        
        # Save misclassification examples
        with open(output_dir / 'misclassification_examples.json', 'w') as f:
            json.dump(misclassification_examples, f, indent=2)
        
        # Create summary report
        self.create_detailed_summary_report(tfidf_results, shap_results, error_results, misclassification_examples)
        
        print(f"   Results saved to {output_dir}")
    
    def create_detailed_summary_report(self, tfidf_results, shap_results, error_results, misclassification_examples):
        """Create detailed summary report"""
        report_content = f"""# Detailed Feature Analysis Report

## TF-IDF Baseline Comparison

### Performance Comparison
"""
        
        for regime, results in tfidf_results.items():
            report_content += f"""
**{regime}:**
- Sentiment AUC: {results['sentiment_auc']:.4f}
- TF-IDF AUC: {results['tfidf_auc']:.4f}
- Improvement: {results['improvement']:.4f}
"""
        
        if shap_results:
            report_content += f"""

## SHAP Feature Importance Analysis

### Top 10 Features by Importance
"""
            top_features = shap_results['feature_importance'].head(10)
            for _, row in top_features.iterrows():
                report_content += f"- {row['feature']}: {row['importance']:.4f}\n"
            
            report_content += f"""

### Text Feature Breakdown
"""
            for category, data in shap_results['text_feature_breakdown'].items():
                report_content += f"""
**{category.title()} Features:**
- Average Importance: {data['importance']:.4f}
- Top Feature: {data['top_feature']}
- Number of Features: {len(data['features'])}
"""
        
        report_content += f"""

## Error Analysis

### Model Performance Comparison
- Tabular Model AUC: {error_results['error_analysis']['tabular_auc']:.4f}
- Hybrid Model AUC: {error_results['error_analysis']['hybrid_auc']:.4f}
- Improvement: {error_results['error_analysis']['improvement_auc']:.4f}

### Error Patterns
- Hybrid Improvements: {error_results['error_analysis']['hybrid_improvements_count']} cases
- Hybrid Worsens: {error_results['error_analysis']['hybrid_worsens_count']} cases
- Improvement Rate: {error_results['error_analysis']['improvement_rate']:.3f}
- Worsen Rate: {error_results['error_analysis']['worsen_rate']:.3f}

## Case Studies

### Key Findings
"""
        
        for case in error_results['case_studies']:
            report_content += f"""
**{case['type'].replace('_', ' ').title()}:**
- Description: {case['description'][:100]}...
- Tabular Prediction: {case['tabular_pred']:.3f}
- Hybrid Prediction: {case['hybrid_pred']:.3f}
- True Label: {case['true_label']}
"""
        
        report_content += f"""

## Misclassification Examples

### Sample Misclassifications
"""
        
        for i, example in enumerate(misclassification_examples[:5]):
            report_content += f"""
**Example {i+1} ({example['misclassification_type']}):**
- Description: {example['description']}
- True Label: {example['true_label']}
- Predicted Label: {example['predicted_label']}
- Prediction Probability: {example['prediction_probability']:.3f}
- Word Count: {example['word_count']}
- Sentiment Balance: {example['sentiment_balance']:.2f}
"""
        
        with open('final_results/detailed_analysis/detailed_summary_report.md', 'w') as f:
            f.write(report_content)

if __name__ == "__main__":
    # Run the detailed feature analysis
    analysis = DetailedFeatureAnalysis(random_state=42)
    results = analysis.run_comprehensive_analysis()
    
    print("\nDetailed Feature Analysis Complete!")
    print("=" * 60)
    print("TF-IDF baseline comparison completed")
    print("SHAP feature importance analysis completed")
    print("Error analysis with case studies completed")
    print("Misclassification examples generated")
    print("Ready for comprehensive insights") 