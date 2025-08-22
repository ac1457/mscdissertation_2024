#!/usr/bin/env python3
"""
Comprehensive Data Analysis for Dissertation - Fixed Version
Generates descriptive statistics, tables, and plots for the data section
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DissertationDataAnalysis:
    def __init__(self):
        self.data = None
        self.output_dir = Path("../dissertation_data_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load the processed dataset and create target variable"""
        try:
            # Try to load real data first
            self.data = pd.read_csv('../data/real_lending_club/real_lending_club_processed.csv')
            print(f"Loaded REAL Lending Club dataset: {len(self.data):,} records")
        except FileNotFoundError:
            # Fall back to synthetic data
            self.data = pd.read_csv('../data/synthetic_loan_descriptions_with_realistic_targets.csv')
            print(f"Using SYNTHETIC data: {len(self.data):,} records")
        
        # Create target variable from loan_status if it exists
        if 'loan_status' in self.data.columns:
            # Define default statuses
            default_statuses = ['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)']
            self.data['target'] = self.data['loan_status'].isin(default_statuses).astype(int)
            print(f"Created target variable from loan_status. Default rate: {self.data['target'].mean()*100:.1f}%")
        elif 'target' not in self.data.columns:
            print("Warning: No target variable found in dataset")
        
        print(f"Dataset shape: {self.data.shape}")
        return self.data
    
    def generate_descriptive_statistics(self):
        """Generate comprehensive descriptive statistics"""
        print("\n=== GENERATING DESCRIPTIVE STATISTICS ===")
        
        # Basic dataset info
        stats = {
            'Total Records': len(self.data),
            'Total Features': len(self.data.columns),
            'Missing Values (%)': (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100,
            'Duplicate Records': self.data.duplicated().sum()
        }
        
        # Target variable statistics
        if 'target' in self.data.columns:
            target_stats = self.data['target'].value_counts()
            stats['Default Rate (%)'] = (target_stats[1] / len(self.data)) * 100
            stats['Non-Default Rate (%)'] = (target_stats[0] / len(self.data)) * 100
        
        # Save basic statistics
        stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
        stats_df.to_csv(self.output_dir / 'basic_dataset_statistics.csv', index=False)
        
        # Numerical features summary
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        numerical_stats = self.data[numerical_cols].describe()
        numerical_stats.to_csv(self.output_dir / 'numerical_features_summary.csv')
        
        # Categorical features summary
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        categorical_stats = {}
        for col in categorical_cols:
            categorical_stats[col] = {
                'unique_values': self.data[col].nunique(),
                'missing_values': self.data[col].isnull().sum(),
                'missing_percentage': (self.data[col].isnull().sum() / len(self.data)) * 100,
                'most_common': self.data[col].mode().iloc[0] if not self.data[col].mode().empty else 'N/A'
            }
        
        categorical_df = pd.DataFrame(categorical_stats).T
        categorical_df.to_csv(self.output_dir / 'categorical_features_summary.csv')
        
        print("Descriptive statistics saved to dissertation_data_analysis/")
        return stats_df, numerical_stats, categorical_df
    
    def create_feature_distribution_plots(self):
        """Create distribution plots for key features"""
        print("\n=== CREATING FEATURE DISTRIBUTION PLOTS ===")
        
        # Set up subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
        
        # 1. Loan Amount Distribution
        if 'loan_amnt' in self.data.columns:
            axes[0, 0].hist(self.data['loan_amnt'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Loan Amount Distribution')
            axes[0, 0].set_xlabel('Loan Amount ($)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Annual Income Distribution
        if 'annual_inc' in self.data.columns:
            axes[0, 1].hist(self.data['annual_inc'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('Annual Income Distribution')
            axes[0, 1].set_xlabel('Annual Income ($)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Interest Rate Distribution
        if 'int_rate' in self.data.columns:
            axes[0, 2].hist(self.data['int_rate'], bins=30, alpha=0.7, color='salmon', edgecolor='black')
            axes[0, 2].set_title('Interest Rate Distribution')
            axes[0, 2].set_xlabel('Interest Rate (%)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. DTI Ratio Distribution
        if 'dti' in self.data.columns:
            axes[1, 0].hist(self.data['dti'], bins=40, alpha=0.7, color='gold', edgecolor='black')
            axes[1, 0].set_title('Debt-to-Income Ratio Distribution')
            axes[1, 0].set_xlabel('DTI Ratio')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Credit Grade Distribution
        if 'grade' in self.data.columns:
            grade_counts = self.data['grade'].value_counts().sort_index()
            axes[1, 1].bar(grade_counts.index, grade_counts.values, alpha=0.7, color='lightcoral')
            axes[1, 1].set_title('Credit Grade Distribution')
            axes[1, 1].set_xlabel('Credit Grade')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Loan Purpose Distribution
        if 'purpose' in self.data.columns:
            purpose_counts = self.data['purpose'].value_counts().head(10)
            axes[1, 2].barh(range(len(purpose_counts)), purpose_counts.values, alpha=0.7, color='lightblue')
            axes[1, 2].set_title('Top 10 Loan Purposes')
            axes[1, 2].set_xlabel('Count')
            axes[1, 2].set_yticks(range(len(purpose_counts)))
            axes[1, 2].set_yticklabels(purpose_counts.index, fontsize=8)
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Feature distribution plots saved")
    
    def create_text_analysis_plots(self):
        """Create plots for text feature analysis"""
        print("\n=== CREATING TEXT ANALYSIS PLOTS ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Text Feature Analysis', fontsize=16, fontweight='bold')
        
        # 1. Text Length Distribution
        if 'text_length' in self.data.columns:
            axes[0, 0].hist(self.data['text_length'], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
            axes[0, 0].set_title('Text Length Distribution')
            axes[0, 0].set_xlabel('Text Length (characters)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Word Count Distribution
        if 'word_count' in self.data.columns:
            axes[0, 1].hist(self.data['word_count'], bins=40, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('Word Count Distribution')
            axes[0, 1].set_xlabel('Word Count')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Sentiment Score Distribution
        if 'sentiment_score' in self.data.columns:
            axes[1, 0].hist(self.data['sentiment_score'], bins=50, alpha=0.7, color='salmon', edgecolor='black')
            axes[1, 0].set_title('Sentiment Score Distribution')
            axes[1, 0].set_xlabel('Sentiment Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Sentiment Categories
        if 'sentiment' in self.data.columns:
            sentiment_counts = self.data['sentiment'].value_counts()
            axes[1, 1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Sentiment Categories Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'text_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Text analysis plots saved")
    
    def create_missing_data_analysis(self):
        """Create missing data analysis and visualization"""
        print("\n=== CREATING MISSING DATA ANALYSIS ===")
        
        # Calculate missing data
        missing_data = self.data.isnull().sum()
        missing_percentage = (missing_data / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Feature': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percentage.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        # Save missing data summary
        missing_df.to_csv(self.output_dir / 'missing_data_summary.csv', index=False)
        
        # Create missing data visualization
        plt.figure(figsize=(12, 8))
        
        # Top 20 features with missing data
        top_missing = missing_df[missing_df['Missing_Percentage'] > 0].head(20)
        
        plt.barh(range(len(top_missing)), top_missing['Missing_Percentage'], 
                color='lightcoral', alpha=0.7, edgecolor='black')
        plt.yticks(range(len(top_missing)), top_missing['Feature'])
        plt.xlabel('Missing Data Percentage (%)')
        plt.title('Top 20 Features with Missing Data')
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'missing_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Missing data analysis saved")
        return missing_df
    
    def create_correlation_analysis(self):
        """Create correlation analysis for numerical features"""
        print("\n=== CREATING CORRELATION ANALYSIS ===")
        
        # Select numerical features
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrix
        correlation_matrix = self.data[numerical_cols].corr()
        
        # Save correlation matrix
        correlation_matrix.to_csv(self.output_dir / 'correlation_matrix.csv')
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        
        # Select top correlated features for visualization
        top_features = correlation_matrix.abs().sum().sort_values(ascending=False).head(15).index
        top_corr = correlation_matrix.loc[top_features, top_features]
        
        sns.heatmap(top_corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Correlation Heatmap - Top 15 Features')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Correlation analysis saved")
        return correlation_matrix
    
    def create_target_analysis(self):
        """Create target variable analysis"""
        print("\n=== CREATING TARGET VARIABLE ANALYSIS ===")
        
        if 'target' not in self.data.columns:
            print("Target variable not found in dataset")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Target Variable Analysis', fontsize=16, fontweight='bold')
        
        # 1. Target Distribution
        target_counts = self.data['target'].value_counts()
        axes[0, 0].pie(target_counts.values, labels=['Non-Default', 'Default'], autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Target Variable Distribution')
        
        # 2. Default Rate by Credit Grade
        if 'grade' in self.data.columns:
            default_by_grade = self.data.groupby('grade')['target'].mean().sort_index()
            axes[0, 1].bar(default_by_grade.index, default_by_grade.values * 100, alpha=0.7, color='lightcoral')
            axes[0, 1].set_title('Default Rate by Credit Grade')
            axes[0, 1].set_xlabel('Credit Grade')
            axes[0, 1].set_ylabel('Default Rate (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Default Rate by Loan Purpose
        if 'purpose' in self.data.columns:
            default_by_purpose = self.data.groupby('purpose')['target'].mean().sort_values(ascending=False).head(10)
            axes[1, 0].barh(range(len(default_by_purpose)), default_by_purpose.values * 100, alpha=0.7, color='lightblue')
            axes[1, 0].set_title('Default Rate by Loan Purpose (Top 10)')
            axes[1, 0].set_xlabel('Default Rate (%)')
            axes[1, 0].set_yticks(range(len(default_by_purpose)))
            axes[1, 0].set_yticklabels(default_by_purpose.index, fontsize=8)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Default Rate by Loan Amount Bins
        if 'loan_amnt' in self.data.columns:
            self.data['loan_amount_bin'] = pd.cut(self.data['loan_amnt'], bins=10)
            default_by_amount = self.data.groupby('loan_amount_bin')['target'].mean()
            axes[1, 1].bar(range(len(default_by_amount)), default_by_amount.values * 100, alpha=0.7, color='lightgreen')
            axes[1, 1].set_title('Default Rate by Loan Amount')
            axes[1, 1].set_xlabel('Loan Amount Bin')
            axes[1, 1].set_ylabel('Default Rate (%)')
            axes[1, 1].set_xticks(range(len(default_by_amount)))
            axes[1, 1].set_xticklabels([f'Bin {i+1}' for i in range(len(default_by_amount))], rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'target_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Target variable analysis saved")
    
    def create_summary_tables(self):
        """Create summary tables for the dissertation"""
        print("\n=== CREATING SUMMARY TABLES ===")
        
        # 1. Dataset Overview Table
        overview_data = {
            'Metric': [
                'Total Records',
                'Total Features',
                'Numerical Features',
                'Categorical Features',
                'Text Features',
                'Missing Values (%)',
                'Default Rate (%)',
                'Average Loan Amount ($)',
                'Average Annual Income ($)',
                'Average Interest Rate (%)'
            ],
            'Value': [
                f"{len(self.data):,}",
                len(self.data.columns),
                len(self.data.select_dtypes(include=[np.number]).columns),
                len(self.data.select_dtypes(include=['object', 'category']).columns),
                len([col for col in self.data.columns if 'sentiment' in col or 'text' in col]),
                f"{(self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100:.2f}",
                f"{(self.data['target'].mean() * 100):.1f}" if 'target' in self.data.columns else "N/A",
                f"{self.data['loan_amnt'].mean():,.0f}" if 'loan_amnt' in self.data.columns else "N/A",
                f"{self.data['annual_inc'].mean():,.0f}" if 'annual_inc' in self.data.columns else "N/A",
                f"{self.data['int_rate'].mean():.2f}" if 'int_rate' in self.data.columns else "N/A"
            ]
        }
        
        overview_df = pd.DataFrame(overview_data)
        overview_df.to_csv(self.output_dir / 'dataset_overview_table.csv', index=False)
        
        # 2. Feature Categories Table
        feature_categories = {
            'Category': [
                'Demographic',
                'Financial',
                'Credit History',
                'Loan Characteristics',
                'Text Features',
                'Sentiment Features',
                'Interaction Features'
            ],
            'Features': [
                'age, annual_inc, emp_length, home_ownership, addr_state',
                'loan_amnt, funded_amnt, int_rate, installment, dti',
                'fico_range_low, fico_range_high, open_acc, pub_rec, inq_last_6mths',
                'purpose, term, grade, sub_grade, verification_status',
                'text_length, word_count, sentence_count, avg_word_length',
                'sentiment_score, sentiment_confidence, sentiment, has_financial_terms',
                'sentiment_text_interaction, sentiment_word_interaction'
            ],
            'Count': [
                len([col for col in self.data.columns if any(x in col for x in ['age', 'inc', 'emp', 'home', 'addr'])]),
                len([col for col in self.data.columns if any(x in col for x in ['loan', 'funded', 'int_rate', 'installment', 'dti'])]),
                len([col for col in self.data.columns if any(x in col for x in ['fico', 'open_acc', 'pub_rec', 'inq'])]),
                len([col for col in self.data.columns if any(x in col for x in ['purpose', 'term', 'grade', 'verification'])]),
                len([col for col in self.data.columns if any(x in col for x in ['text', 'word', 'sentence', 'avg_word'])]),
                len([col for col in self.data.columns if 'sentiment' in col]),
                len([col for col in self.data.columns if 'interaction' in col])
            ]
        }
        
        feature_cat_df = pd.DataFrame(feature_categories)
        feature_cat_df.to_csv(self.output_dir / 'feature_categories_table.csv', index=False)
        
        print("Summary tables saved")
        return overview_df, feature_cat_df
    
    def create_comprehensive_report(self, stats_df, numerical_stats, categorical_df, 
                                  missing_df, correlation_matrix, overview_df, feature_cat_df):
        """Create a comprehensive markdown report"""
        
        # Get default rate safely
        default_rate = "N/A"
        if 'target' in self.data.columns:
            default_rate = f"{self.data['target'].mean() * 100:.1f}%"
        
        report_content = f"""# Dissertation Data Analysis Report

## Dataset Overview

{overview_df.to_markdown(index=False)}

## Feature Categories

{feature_cat_df.to_markdown(index=False)}

## Key Findings

### Data Quality
- **Missing Data**: {stats_df[stats_df['Metric'] == 'Missing Values (%)']['Value'].iloc[0]}% of all values are missing
- **Data Completeness**: {stats_df[stats_df['Metric'] == 'Total Records']['Value'].iloc[0]} complete records
- **Feature Coverage**: {stats_df[stats_df['Metric'] == 'Total Features']['Value'].iloc[0]} total features

### Target Variable
- **Default Rate**: {default_rate} of loans default
- **Class Imbalance**: Significant imbalance requiring careful handling

### Text Features
- **Synthetic Text**: Generated for missing descriptions
- **Text Quality**: Comprehensive sentiment and complexity features extracted

## Generated Visualizations

1. **feature_distributions.png**: Distribution of key numerical features
2. **text_analysis_plots.png**: Text feature analysis and sentiment distribution
3. **missing_data_analysis.png**: Missing data patterns across features
4. **correlation_heatmap.png**: Feature correlation analysis
5. **target_analysis.png**: Target variable analysis by various factors

## Data Files

1. **basic_dataset_statistics.csv**: Basic dataset statistics
2. **numerical_features_summary.csv**: Numerical feature statistics
3. **categorical_features_summary.csv**: Categorical feature statistics
4. **missing_data_summary.csv**: Missing data analysis
5. **correlation_matrix.csv**: Full correlation matrix
6. **dataset_overview_table.csv**: Dataset overview table
7. **feature_categories_table.csv**: Feature categorization

## Recommendations for Dissertation

1. **Use the overview table** in your data section for dataset characteristics
2. **Include the feature distributions** to show data patterns
3. **Reference the missing data analysis** to justify synthetic text generation
4. **Use correlation analysis** to show feature relationships
5. **Include target analysis** to demonstrate class imbalance challenges

This analysis provides comprehensive insights into your dataset for the dissertation data section.
"""
        
        with open(self.output_dir / 'dissertation_data_analysis_report.md', 'w') as f:
            f.write(report_content)
        
        print("Comprehensive report saved")

    def generate_dissertation_data_section(self):
        """Generate the complete dissertation data section with all analysis"""
        print("=== GENERATING COMPLETE DISSERTATION DATA ANALYSIS ===")
        
        # Load data
        self.load_data()
        
        # Generate all analyses
        stats_df, numerical_stats, categorical_df = self.generate_descriptive_statistics()
        self.create_feature_distribution_plots()
        self.create_text_analysis_plots()
        missing_df = self.create_missing_data_analysis()
        correlation_matrix = self.create_correlation_analysis()
        self.create_target_analysis()
        overview_df, feature_cat_df = self.create_summary_tables()
        
        # Create comprehensive report
        self.create_comprehensive_report(stats_df, numerical_stats, categorical_df, 
                                       missing_df, correlation_matrix, overview_df, feature_cat_df)
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"All outputs saved to: {self.output_dir}")
        print(f"Generated files:")
        for file in self.output_dir.glob("*"):
            print(f"  - {file.name}")

if __name__ == "__main__":
    analyzer = DissertationDataAnalysis()
    analyzer.generate_dissertation_data_section() 