# Dissertation Data Analysis Report

## Dataset Overview

| Metric                    | Value   |
|:--------------------------|:--------|
| Total Records             | 97,020  |
| Total Features            | 167     |
| Numerical Features        | 128     |
| Categorical Features      | 39      |
| Text Features             | 4       |
| Missing Values (%)        | 0.60    |
| Default Rate (%)          | 13.1    |
| Average Loan Amount ($)   | 15,176  |
| Average Annual Income ($) | 78,343  |
| Average Interest Rate (%) | 13.14   |

## Feature Categories

| Category             | Features                                                              |   Count |
|:---------------------|:----------------------------------------------------------------------|--------:|
| Demographic          | age, annual_inc, emp_length, home_ownership, addr_state               |      16 |
| Financial            | loan_amnt, funded_amnt, int_rate, installment, dti                    |      10 |
| Credit History       | fico_range_low, fico_range_high, open_acc, pub_rec, inq_last_6mths    |      21 |
| Loan Characteristics | purpose, term, grade, sub_grade, verification_status                  |       9 |
| Text Features        | text_length, word_count, sentence_count, avg_word_length              |       9 |
| Sentiment Features   | sentiment_score, sentiment_confidence, sentiment, has_financial_terms |       3 |
| Interaction Features | sentiment_text_interaction, sentiment_word_interaction                |       0 |

## Key Findings

### Data Quality
- **Missing Data**: 0.6024096385542169% of all values are missing
- **Data Completeness**: 97020.0 complete records
- **Feature Coverage**: 166.0 total features

### Target Variable
- **Default Rate**: 13.1% of loans default
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
