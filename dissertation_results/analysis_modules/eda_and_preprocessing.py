"""
Exploratory Data Analysis and Data Preprocessing
Comprehensive EDA and data cleaning/preprocessing for Lending Club sentiment analysis.
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class EDAAndPreprocessing:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Configuration
        self.config = {
            'eda_plots_dir': 'eda_plots',
            'fast_eda_plots_dir': 'fast_eda_plots',
            'data_dir': 'data',
            'output_dir': 'final_results/eda_and_preprocessing'
        }
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def run_comprehensive_eda_and_preprocessing(self):
        """Run comprehensive EDA and data preprocessing"""
        print("Exploratory Data Analysis and Data Preprocessing")
        print("=" * 60)
        
        # Load and explore data
        df = self.load_and_explore_data()
        if df is None:
            return
        
        # Perform comprehensive EDA
        eda_results = self.perform_comprehensive_eda(df)
        
        # Perform data cleaning and preprocessing
        cleaned_df = self.perform_data_cleaning_and_preprocessing(df)
        
        # Create enhanced features
        enhanced_df = self.create_enhanced_features(cleaned_df)
        
        # Generate comprehensive EDA plots
        self.generate_comprehensive_eda_plots(df, enhanced_df)
        
        # Save preprocessing results
        self.save_preprocessing_results(cleaned_df, enhanced_df, eda_results)
        
        return cleaned_df, enhanced_df, eda_results
    
    def load_and_explore_data(self):
        """Load and perform initial data exploration"""
        print("Loading and exploring data...")
        
        try:
            # Load synthetic loan descriptions
            df = pd.read_csv('data/synthetic_loan_descriptions.csv')
            
            print(f"Dataset loaded: {len(df)} records, {len(df.columns)} columns")
            print(f"Columns: {list(df.columns)}")
            
            # Basic data info
            print(f"\nData Types:")
            print(df.dtypes.value_counts())
            
            print(f"\nMissing Values:")
            missing_data = df.isnull().sum()
            print(missing_data[missing_data > 0])
            
            print(f"\nDataset Shape: {df.shape}")
            print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return df
            
        except FileNotFoundError:
            print("Dataset not found")
            return None
    
    def perform_comprehensive_eda(self, df):
        """Perform comprehensive exploratory data analysis"""
        print("Performing comprehensive EDA...")
        
        eda_results = {
            'dataset_info': {},
            'text_analysis': {},
            'sentiment_analysis': {},
            'feature_analysis': {},
            'correlation_analysis': {},
            'missing_data_analysis': {}
        }
        
        # Dataset information
        eda_results['dataset_info'] = {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Missing data analysis
        missing_data = df.isnull().sum()
        eda_results['missing_data_analysis'] = {
            'missing_counts': missing_data.to_dict(),
            'missing_percentages': (missing_data / len(df) * 100).to_dict(),
            'columns_with_missing': missing_data[missing_data > 0].index.tolist()
        }
        
        # Text analysis
        if 'description' in df.columns:
            eda_results['text_analysis'] = self.analyze_text_features(df)
        
        # Sentiment analysis
        if 'sentiment' in df.columns:
            eda_results['sentiment_analysis'] = self.analyze_sentiment_features(df)
        
        # Feature analysis
        eda_results['feature_analysis'] = self.analyze_features(df)
        
        # Correlation analysis
        eda_results['correlation_analysis'] = self.analyze_correlations(df)
        
        return eda_results
    
    def analyze_text_features(self, df):
        """Analyze text features"""
        text_analysis = {}
        
        # Text length analysis
        df['text_length'] = df['description'].str.len()
        df['word_count'] = df['description'].str.split().str.len()
        df['sentence_count'] = df['description'].str.count(r'[.!?]+')
        
        text_analysis['text_length_stats'] = {
            'mean': df['text_length'].mean(),
            'std': df['text_length'].std(),
            'min': df['text_length'].min(),
            'max': df['text_length'].max(),
            'median': df['text_length'].median()
        }
        
        text_analysis['word_count_stats'] = {
            'mean': df['word_count'].mean(),
            'std': df['word_count'].std(),
            'min': df['word_count'].min(),
            'max': df['word_count'].max(),
            'median': df['word_count'].median()
        }
        
        text_analysis['sentence_count_stats'] = {
            'mean': df['sentence_count'].mean(),
            'std': df['sentence_count'].std(),
            'min': df['sentence_count'].min(),
            'max': df['sentence_count'].max(),
            'median': df['sentence_count'].median()
        }
        
        return text_analysis
    
    def analyze_sentiment_features(self, df):
        """Analyze sentiment features"""
        sentiment_analysis = {}
        
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            sentiment_analysis['sentiment_distribution'] = sentiment_counts.to_dict()
            sentiment_analysis['sentiment_percentages'] = (sentiment_counts / len(df) * 100).to_dict()
        
        return sentiment_analysis
    
    def analyze_features(self, df):
        """Analyze all features"""
        feature_analysis = {}
        
        for column in df.columns:
            if df[column].dtype in ['object', 'category']:
                # Categorical features
                value_counts = df[column].value_counts()
                feature_analysis[column] = {
                    'type': 'categorical',
                    'unique_values': df[column].nunique(),
                    'top_values': value_counts.head(5).to_dict(),
                    'missing_count': df[column].isnull().sum()
                }
            elif df[column].dtype in ['int64', 'float64']:
                # Numerical features
                feature_analysis[column] = {
                    'type': 'numerical',
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'min': df[column].min(),
                    'max': df[column].max(),
                    'median': df[column].median(),
                    'missing_count': df[column].isnull().sum()
                }
        
        return feature_analysis
    
    def analyze_correlations(self, df):
        """Analyze correlations between features"""
        correlation_analysis = {}
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) > 1:
            correlation_matrix = df[numerical_cols].corr()
            
            # Find high correlations
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        high_correlations.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            correlation_analysis['correlation_matrix'] = correlation_matrix.to_dict()
            correlation_analysis['high_correlations'] = high_correlations
            correlation_analysis['numerical_features'] = numerical_cols
        
        return correlation_analysis
    
    def perform_data_cleaning_and_preprocessing(self, df):
        """Perform data cleaning and preprocessing"""
        print("Performing data cleaning and preprocessing...")
        
        # Create a copy for cleaning
        cleaned_df = df.copy()
        
        # Handle missing values
        cleaned_df = self.handle_missing_values(cleaned_df)
        
        # Clean text data
        if 'description' in cleaned_df.columns:
            cleaned_df = self.clean_text_data(cleaned_df)
        
        # Handle outliers
        cleaned_df = self.handle_outliers(cleaned_df)
        
        # Feature engineering
        cleaned_df = self.perform_feature_engineering(cleaned_df)
        
        print(f"Data cleaning completed. Shape: {cleaned_df.shape}")
        
        return cleaned_df
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        print("   Handling missing values...")
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                print(f"      {column}: {missing_count} missing values")
                
                if df[column].dtype in ['object', 'category']:
                    # Categorical: fill with mode
                    mode_value = df[column].mode()[0] if len(df[column].mode()) > 0 else 'Unknown'
                    df[column] = df[column].fillna(mode_value)
                else:
                    # Numerical: fill with median
                    df[column] = df[column].fillna(df[column].median())
        
        return df
    
    def clean_text_data(self, df):
        """Clean text data"""
        print("   Cleaning text data...")
        
        # Basic text cleaning
        df['description_cleaned'] = df['description'].astype(str).apply(self.clean_text)
        
        # Remove very short or very long texts
        df = df[df['description_cleaned'].str.len() > 10]
        df = df[df['description_cleaned'].str.len() < 5000]
        
        return df
    
    def clean_text(self, text):
        """Clean individual text"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def handle_outliers(self, df):
        """Handle outliers in numerical features"""
        print("   Handling outliers...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in numerical_cols:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def perform_feature_engineering(self, df):
        """Perform feature engineering"""
        print("   Performing feature engineering...")
        
        # Text-based features
        if 'description_cleaned' in df.columns:
            df['text_length'] = df['description_cleaned'].str.len()
            df['word_count'] = df['description_cleaned'].str.split().str.len()
            df['sentence_count'] = df['description_cleaned'].str.count(r'[.!?]+')
            df['avg_word_length'] = df['description_cleaned'].str.split().apply(
                lambda x: np.mean([len(word) for word in x]) if x else 0
            )
        
        # Sentiment features
        if 'sentiment' in df.columns:
            # Create sentiment encoding
            sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
            df['sentiment_encoded'] = df['sentiment'].map(sentiment_mapping)
        
        return df
    
    def create_enhanced_features(self, df):
        """Create enhanced features"""
        print("Creating enhanced features...")
        
        enhanced_df = df.copy()
        
        # Sentiment analysis features
        enhanced_df = self.create_sentiment_features(enhanced_df)
        
        # Text complexity features
        enhanced_df = self.create_text_complexity_features(enhanced_df)
        
        # Financial keyword features
        enhanced_df = self.create_financial_keyword_features(enhanced_df)
        
        print(f"Enhanced features created. Shape: {enhanced_df.shape}")
        
        return enhanced_df
    
    def create_sentiment_features(self, df):
        """Create sentiment analysis features"""
        # Positive and negative word lists
        positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'improve', 'help', 'support']
        negative_words = ['bad', 'poor', 'negative', 'problem', 'issue', 'difficult', 'struggle', 'debt']
        
        df['positive_word_count'] = df['description_cleaned'].apply(
            lambda x: sum(1 for word in x.split() if word in positive_words)
        )
        df['negative_word_count'] = df['description_cleaned'].apply(
            lambda x: sum(1 for word in x.split() if word in negative_words)
        )
        df['sentiment_balance'] = df['positive_word_count'] - df['negative_word_count']
        df['sentiment_score'] = df['sentiment_balance'] / (df['positive_word_count'] + df['negative_word_count'] + 1)
        
        return df
    
    def create_text_complexity_features(self, df):
        """Create text complexity features"""
        df['type_token_ratio'] = df['description_cleaned'].apply(
            lambda x: len(set(x.split())) / len(x.split()) if x.split() else 0
        )
        df['sentence_length_std'] = df['description_cleaned'].apply(
            lambda x: np.std([len(sent.split()) for sent in x.split('.') if sent.strip()]) if len([sent for sent in x.split('.') if sent.strip()]) > 1 else 0
        )
        
        return df
    
    def create_financial_keyword_features(self, df):
        """Create financial keyword features"""
        financial_keywords = ['loan', 'debt', 'credit', 'money', 'payment', 'interest', 'bank', 'financial']
        
        df['financial_keyword_count'] = df['description_cleaned'].apply(
            lambda x: sum(1 for word in x.split() if word in financial_keywords)
        )
        df['has_financial_terms'] = (df['financial_keyword_count'] > 0).astype(int)
        
        return df
    
    def generate_comprehensive_eda_plots(self, original_df, enhanced_df):
        """Generate comprehensive EDA plots"""
        print("Generating comprehensive EDA plots...")
        
        # Create output directories
        eda_dir = Path(self.config['eda_plots_dir'])
        fast_eda_dir = Path(self.config['fast_eda_plots_dir'])
        output_dir = Path(self.config['output_dir'])
        
        eda_dir.mkdir(exist_ok=True)
        fast_eda_dir.mkdir(exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate basic EDA plots
        self.generate_basic_eda_plots(original_df, eda_dir)
        
        # Generate fast EDA plots
        self.generate_fast_eda_plots(original_df, fast_eda_dir)
        
        # Generate enhanced EDA plots
        self.generate_enhanced_eda_plots(enhanced_df, output_dir)
    
    def generate_basic_eda_plots(self, df, output_dir):
        """Generate basic EDA plots"""
        print("   Generating basic EDA plots...")
        
        # 1. Text length distribution
        plt.figure(figsize=(10, 6))
        df['text_length'] = df['description'].str.len()
        plt.hist(df['text_length'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of Text Length')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'text_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Word count distribution
        plt.figure(figsize=(10, 6))
        df['word_count'] = df['description'].str.split().str.len()
        plt.hist(df['word_count'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.title('Distribution of Word Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'word_count_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Sentiment distribution (if available)
        if 'sentiment' in df.columns:
            plt.figure(figsize=(10, 6))
            sentiment_counts = df['sentiment'].value_counts()
            plt.bar(sentiment_counts.index, sentiment_counts.values, alpha=0.7)
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.title('Sentiment Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'sentiment_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Missing data analysis
        plt.figure(figsize=(12, 6))
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            plt.bar(missing_data.index, missing_data.values, alpha=0.7)
            plt.xlabel('Features')
            plt.ylabel('Missing Count')
            plt.title('Missing Data Analysis')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'missing_data_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Correlation heatmap
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = df[numerical_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_fast_eda_plots(self, df, output_dir):
        """Generate fast EDA plots for quick insights"""
        print("   Generating fast EDA plots...")
        
        # 1. Quick text length overview
        plt.figure(figsize=(8, 6))
        df['text_length'] = df['description'].str.len()
        plt.hist(df['text_length'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.title('Text Length Distribution (Fast EDA)')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'text_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Quick word count overview
        plt.figure(figsize=(8, 6))
        df['word_count'] = df['description'].str.split().str.len()
        plt.hist(df['word_count'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.title('Word Count Distribution (Fast EDA)')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'word_count_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Quick sentiment overview
        if 'sentiment' in df.columns:
            plt.figure(figsize=(8, 6))
            sentiment_counts = df['sentiment'].value_counts()
            plt.bar(sentiment_counts.index, sentiment_counts.values, alpha=0.7)
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.title('Sentiment Distribution (Fast EDA)')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'sentiment_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_enhanced_eda_plots(self, df, output_dir):
        """Generate enhanced EDA plots"""
        print("   Generating enhanced EDA plots...")
        
        # 1. Feature importance plot
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) > 1:
            plt.figure(figsize=(12, 8))
            # Calculate correlation with target if available
            if 'target_10%' in df.columns:
                correlations = df[numerical_cols].corrwith(df['target_10%']).abs().sort_values(ascending=False)
                plt.bar(range(len(correlations)), correlations.values, alpha=0.7)
                plt.xlabel('Features')
                plt.ylabel('Absolute Correlation with Target')
                plt.title('Feature Importance (Correlation with Target)')
                plt.xticks(range(len(correlations)), correlations.index, rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. Enhanced text analysis
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.hist(df['text_length'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.title('Text Length Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.hist(df['word_count'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.title('Word Count Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        plt.hist(df['sentiment_score'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        plt.title('Sentiment Score Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 4)
        plt.hist(df['financial_keyword_count'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Financial Keyword Count')
        plt.ylabel('Frequency')
        plt.title('Financial Keyword Count Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        plt.hist(df['type_token_ratio'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Type-Token Ratio')
        plt.ylabel('Frequency')
        plt.title('Type-Token Ratio Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 6)
        plt.hist(df['sentence_length_std'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Sentence Length Std')
        plt.ylabel('Frequency')
        plt.title('Sentence Length Std Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'enhanced_text_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_preprocessing_results(self, cleaned_df, enhanced_df, eda_results):
        """Save preprocessing results"""
        print("Saving preprocessing results...")
        
        # Create output directory
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned data
        cleaned_df.to_csv(output_dir / 'cleaned_data.csv', index=False)
        
        # Save enhanced data
        enhanced_df.to_csv(output_dir / 'enhanced_data.csv', index=False)
        
        # Save EDA results
        with open(output_dir / 'eda_results.json', 'w') as f:
            json.dump(eda_results, f, indent=2, default=str)
        
        # Create preprocessing summary
        self.create_preprocessing_summary(cleaned_df, enhanced_df, eda_results)
        
        print(f"   Results saved to {output_dir}")
    
    def create_preprocessing_summary(self, cleaned_df, enhanced_df, eda_results):
        """Create preprocessing summary report"""
        # Create output directory
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_content = f"""# EDA and Data Preprocessing Summary

## Dataset Overview
- **Original shape:** {eda_results['dataset_info']['shape']}
- **Cleaned shape:** {cleaned_df.shape}
- **Enhanced shape:** {enhanced_df.shape}
- **Memory usage:** {eda_results['dataset_info']['memory_usage_mb']:.2f} MB

## Data Cleaning Steps
1. **Missing value handling:** Filled categorical with mode, numerical with median
2. **Text cleaning:** Basic text preprocessing and length filtering
3. **Outlier handling:** Capped outliers using IQR method
4. **Feature engineering:** Created text-based and sentiment features

## Enhanced Features Created
- **Text features:** text_length, word_count, sentence_count, avg_word_length
- **Sentiment features:** positive_word_count, negative_word_count, sentiment_balance, sentiment_score
- **Complexity features:** type_token_ratio, sentence_length_std
- **Financial features:** financial_keyword_count, has_financial_terms

## Key Insights
- **Missing data:** {len(eda_results['missing_data_analysis']['columns_with_missing'])} columns had missing values
- **Text analysis:** Average text length: {eda_results['text_analysis']['text_length_stats']['mean']:.1f} characters
- **Feature engineering:** {len(enhanced_df.columns) - len(cleaned_df.columns)} new features created

---
**Analysis completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(output_dir / 'preprocessing_summary.md', 'w') as f:
            f.write(summary_content)

if __name__ == "__main__":
    # Run the EDA and preprocessing analysis
    analysis = EDAAndPreprocessing(random_state=42)
    results = analysis.run_comprehensive_eda_and_preprocessing()
    
    print("\nEDA and Data Preprocessing Complete!")
    print("=" * 60)
    print("Comprehensive EDA completed")
    print("Data cleaning and preprocessing completed")
    print("Enhanced features created")
    print("EDA plots generated")
    print("Ready for analysis") 