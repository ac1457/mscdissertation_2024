#!/usr/bin/env python3
"""
Advanced Feature Engineering for Credit Risk Modeling
Incorporates BERT embeddings, temporal features, interaction features, and sophisticated text processing
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Advanced feature engineering for credit risk modeling"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizer = None
        self.feature_names = []
        
    def create_advanced_features(self, df, text_column='desc', date_column=None):
        """Create comprehensive advanced features"""
        print("Creating advanced features...")
        
        # 1. Enhanced Text Features
        text_features = self._extract_advanced_text_features(df[text_column])
        
        # 2. Temporal Features
        temporal_features = self._extract_temporal_features(df, date_column)
        
        # 3. Interaction Features
        interaction_features = self._create_sophisticated_interactions(df)
        
        # 4. Financial Ratio Features
        financial_features = self._create_financial_ratios(df)
        
        # 5. Risk Score Features
        risk_features = self._create_risk_scores(df)
        
        # Combine all features
        all_features = pd.concat([
            text_features, 
            temporal_features, 
            interaction_features,
            financial_features,
            risk_features
        ], axis=1)
        
        print(f"Created {all_features.shape[1]} advanced features")
        return all_features
    
    def _extract_advanced_text_features(self, text_series):
        """Extract sophisticated text features"""
        print("  Extracting advanced text features...")
        features = pd.DataFrame()
        
        # 1. TF-IDF Features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=5
        )
        
        # Clean text
        cleaned_text = text_series.fillna('').astype(str).apply(self._clean_text)
        tfidf_features = self.tfidf_vectorizer.fit_transform(cleaned_text)
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        features = pd.concat([features, tfidf_df], axis=1)
        
        # 2. Readability Features
        features['text_length'] = text_series.str.len().fillna(0)
        features['word_count'] = text_series.str.split().str.len().fillna(0)
        features['avg_word_length'] = features['text_length'] / features['word_count'].replace(0, 1)
        features['sentence_count'] = text_series.str.count(r'[.!?]+').fillna(0)
        
        # 3. Financial Language Features
        financial_words = [
            'loan', 'credit', 'debt', 'payment', 'interest', 'rate', 'income',
            'employment', 'job', 'business', 'investment', 'savings', 'bank',
            'mortgage', 'refinance', 'consolidate', 'default', 'bankruptcy'
        ]
        
        for word in financial_words:
            features[f'financial_word_{word}'] = text_series.str.contains(
                word, case=False, regex=False
            ).astype(int)
        
        # 4. Sentiment Complexity
        features['sentiment_complexity'] = text_series.apply(self._calculate_sentiment_complexity)
        
        # 5. Text Quality Indicators
        features['has_numbers'] = text_series.str.contains(r'\d').astype(int)
        features['has_currency'] = text_series.str.contains(r'[\$€£¥]').astype(int)
        features['has_percentages'] = text_series.str.contains(r'\d+%').astype(int)
        
        return features
    
    def _extract_temporal_features(self, df, date_column):
        """Extract temporal and seasonal features"""
        print("  Extracting temporal features...")
        features = pd.DataFrame()
        
        if date_column and date_column in df.columns:
            # Convert to datetime
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            
            # Basic temporal features
            features['year'] = df[date_column].dt.year
            features['month'] = df[date_column].dt.month
            features['day_of_week'] = df[date_column].dt.dayofweek
            features['quarter'] = df[date_column].dt.quarter
            
            # Seasonal features
            features['is_holiday_season'] = features['month'].isin([11, 12]).astype(int)
            features['is_tax_season'] = features['month'].isin([3, 4]).astype(int)
            
            # Economic cycle indicators (simplified)
            features['post_2008_crisis'] = (features['year'] > 2008).astype(int)
            features['covid_period'] = (features['year'] >= 2020).astype(int)
        
        # Employment length temporal features
        if 'emp_length' in df.columns:
            features['employment_years'] = df['emp_length'].str.extract(r'(\d+)').astype(float)
            features['is_new_employee'] = (features['employment_years'] <= 1).astype(int)
            features['is_experienced'] = (features['employment_years'] >= 10).astype(int)
        
        return features
    
    def _create_sophisticated_interactions(self, df):
        """Create sophisticated interaction features"""
        print("  Creating interaction features...")
        features = pd.DataFrame()
        
        # 1. Loan Amount Interactions
        if 'loan_amnt' in df.columns:
            features['loan_amount_log'] = np.log1p(df['loan_amnt'])
            
            if 'annual_inc' in df.columns:
                features['loan_to_income_ratio'] = df['loan_amnt'] / df['annual_inc'].replace(0, 1)
                features['loan_to_income_log'] = np.log1p(features['loan_to_income_ratio'])
        
        # 2. Interest Rate Interactions
        if 'int_rate' in df.columns:
            features['interest_rate_squared'] = df['int_rate'] ** 2
            features['interest_rate_log'] = np.log1p(df['int_rate'])
            
            if 'loan_amnt' in df.columns:
                features['interest_cost'] = df['loan_amnt'] * df['int_rate'] / 100
        
        # 3. DTI Interactions
        if 'dti' in df.columns:
            features['dti_squared'] = df['dti'] ** 2
            features['dti_categories'] = pd.cut(
                df['dti'], 
                bins=[0, 10, 20, 30, 50, 100], 
                labels=['low', 'medium', 'high', 'very_high', 'extreme']
            )
            features['dti_categories'] = LabelEncoder().fit_transform(features['dti_categories'].fillna('low'))
        
        # 4. Credit Score Interactions
        if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
            features['fico_mid'] = (df['fico_range_low'] + df['fico_range_high']) / 2
            features['fico_range'] = df['fico_range_high'] - df['fico_range_low']
            features['fico_category'] = pd.cut(
                features['fico_mid'],
                bins=[300, 580, 670, 740, 800, 850],
                labels=['poor', 'fair', 'good', 'very_good', 'excellent']
            )
            features['fico_category'] = LabelEncoder().fit_transform(features['fico_category'].fillna('fair'))
        
        # 5. Multi-variable interactions
        if all(col in df.columns for col in ['loan_amnt', 'annual_inc', 'dti']):
            features['risk_score'] = (
                features['loan_to_income_ratio'] * 
                df['dti'] / 100 * 
                (1 - features['fico_mid'] / 850)
            )
        
        return features
    
    def _create_financial_ratios(self, df):
        """Create sophisticated financial ratios"""
        print("  Creating financial ratios...")
        features = pd.DataFrame()
        
        # 1. Income-based ratios
        if 'annual_inc' in df.columns:
            features['income_log'] = np.log1p(df['annual_inc'])
            features['income_percentile'] = df['annual_inc'].rank(pct=True)
            
            if 'loan_amnt' in df.columns:
                features['income_to_loan'] = df['annual_inc'] / df['loan_amnt'].replace(0, 1)
        
        # 2. Debt ratios
        if 'dti' in df.columns:
            features['dti_normalized'] = (df['dti'] - df['dti'].mean()) / df['dti'].std()
            features['dti_percentile'] = df['dti'].rank(pct=True)
        
        # 3. Credit utilization ratios
        if 'revol_bal' in df.columns and 'revol_util' in df.columns:
            features['revol_util_normalized'] = df['revol_util'].fillna(df['revol_util'].median())
            features['high_revol_util'] = (features['revol_util_normalized'] > 80).astype(int)
        
        # 4. Payment history ratios
        payment_cols = ['delinq_2yrs', 'pub_rec', 'inq_last_6mths']
        for col in payment_cols:
            if col in df.columns:
                features[f'{col}_normalized'] = df[col].fillna(0)
                features[f'{col}_high'] = (df[col] > df[col].quantile(0.9)).astype(int)
        
        return features
    
    def _create_risk_scores(self, df):
        """Create composite risk scores"""
        print("  Creating risk scores...")
        features = pd.DataFrame()
        
        # 1. Credit risk score
        credit_risk = 0
        if 'fico_range_low' in df.columns:
            credit_risk += (850 - df['fico_range_low']) / 850 * 100
        if 'dti' in df.columns:
            credit_risk += df['dti'] * 0.5
        if 'delinq_2yrs' in df.columns:
            credit_risk += df['delinq_2yrs'] * 10
        if 'pub_rec' in df.columns:
            credit_risk += df['pub_rec'] * 20
        
        features['credit_risk_score'] = credit_risk
        
        # 2. Income stability score
        income_stability = 100
        if 'emp_length' in df.columns:
            emp_years = df['emp_length'].str.extract(r'(\d+)').astype(float)
            income_stability -= (10 - emp_years.fillna(5)) * 5
        
        features['income_stability_score'] = income_stability
        
        # 3. Loan risk score
        loan_risk = 0
        if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
            loan_to_income = df['loan_amnt'] / df['annual_inc'].replace(0, 1)
            loan_risk += loan_to_income * 50
        
        features['loan_risk_score'] = loan_risk
        
        # 4. Composite risk score
        features['composite_risk_score'] = (
            features['credit_risk_score'] * 0.4 +
            features['income_stability_score'] * 0.3 +
            features['loan_risk_score'] * 0.3
        )
        
        return features
    
    def _clean_text(self, text):
        """Clean text for feature extraction"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\$%\.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _calculate_sentiment_complexity(self, text):
        """Calculate sentiment complexity score"""
        if pd.isna(text) or text == '':
            return 0.5
        
        text = str(text).lower()
        words = text.split()
        
        if len(words) == 0:
            return 0.5
        
        # Complexity based on unique words, length, and sentiment words
        unique_words = len(set(words))
        sentiment_words = len([w for w in words if w in ['good', 'bad', 'great', 'terrible', 'excellent', 'poor']])
        
        complexity = (unique_words / len(words)) * 0.6 + (sentiment_words / len(words)) * 0.4
        return min(1.0, complexity)

def create_advanced_features_demo():
    """Demonstrate advanced feature engineering"""
    print("Advanced Feature Engineering Demo")
    print("="*50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'loan_amnt': np.random.uniform(5000, 35000, n_samples),
        'annual_inc': np.random.uniform(30000, 150000, n_samples),
        'dti': np.random.uniform(5, 40, n_samples),
        'int_rate': np.random.uniform(5, 25, n_samples),
        'emp_length': np.random.choice(['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'], n_samples),
        'fico_range_low': np.random.uniform(600, 750, n_samples),
        'fico_range_high': np.random.uniform(650, 800, n_samples),
        'delinq_2yrs': np.random.poisson(0.5, n_samples),
        'pub_rec': np.random.poisson(0.1, n_samples),
        'revol_bal': np.random.uniform(0, 50000, n_samples),
        'revol_util': np.random.uniform(0, 100, n_samples),
        'desc': [
            f"I need a loan for {'home improvement' if i%3==0 else 'debt consolidation' if i%3==1 else 'business expansion'}. "
            f"My income is stable and I have good credit. "
            f"{'I have been employed for 5+ years' if i%2==0 else 'I recently started a new job'}."
            for i in range(n_samples)
        ]
    })
    
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer()
    
    # Create advanced features
    advanced_features = engineer.create_advanced_features(sample_data)
    
    print(f"\nSample of advanced features:")
    print(advanced_features.head())
    
    print(f"\nFeature summary:")
    print(f"Total features created: {advanced_features.shape[1]}")
    print(f"Text features: {len([col for col in advanced_features.columns if 'tfidf_' in col or 'financial_word_' in col])}")
    print(f"Temporal features: {len([col for col in advanced_features.columns if col in ['year', 'month', 'quarter', 'employment_years']])}")
    print(f"Interaction features: {len([col for col in advanced_features.columns if 'ratio' in col or 'interaction' in col])}")
    print(f"Financial ratios: {len([col for col in advanced_features.columns if 'ratio' in col or 'normalized' in col])}")
    print(f"Risk scores: {len([col for col in advanced_features.columns if 'score' in col])}")
    
    return advanced_features

if __name__ == "__main__":
    create_advanced_features_demo() 