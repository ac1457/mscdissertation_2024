# Executive Summary: Lending Club Sentiment Analysis for Credit Risk Modeling

## Project Overview
This dissertation investigates the integration of FinBERT-derived sentiment features with traditional credit risk predictors to enhance default prediction accuracy. The study employs real Lending Club data with synthetic text generation to address missing descriptions.

## Key Findings

### Model Performance
- **Best Approach**: Sentiment_Interactions (AUC: 0.5634963851136686 in 5% default regime)
- **Improvement**: Modest AUC gains (0.0054216084427507-0.0082035909491503) across regimes
- **Statistical Significance**: Not achieved after multiple comparison correction
- **Practical Impact**: Negative business impact (-$72,000 total value added)

### Feature Importance
- **Top Features**: Text structure metrics (word_count: 0.5597236675949344, text_length: 0.5577127992524351, sentence_length_std: 0.5564981601367105)
- **Sentiment Features**: Moderate predictive value
- **Financial Indicators**: Domain-specific features show promise

### Business Impact
- **Value Added**: Negative (-$72,000 total value added)
- **Lift Performance**: Below baseline in most percentiles
- **Recommendation**: Focus on methodological rigor over performance gains

## Methodology
- **Data**: 97,020 Lending Club records with synthetic descriptions
- **Features**: 165 total features (151 original + 14 enhanced)
- **Validation**: 5-fold temporal cross-validation
- **Testing**: Concept drift monitoring, counterfactual fairness analysis

## Academic Contribution
- **Methodological Framework**: Rigorous evaluation of text features in credit risk
- **Statistical Rigor**: Proper multiple comparison correction
- **Transparent Reporting**: Honest assessment of negative results
- **Reproducible Research**: Complete code and documentation

## Conclusion
While sentiment features provide modest improvements, the primary contribution is establishing methodological boundaries and providing a framework for future text-based credit risk research. The work demonstrates the importance of rigorous statistical validation and transparent reporting of negative results.

## Files Structure
- `performance_metrics_table.csv`: Complete model comparison results
- `feature_importance_ranking.csv`: Feature importance analysis
- `business_impact_metrics.csv`: Business impact quantification
- `figures/`: All generated plots and visualizations
- `models/`: Serialized model files (if applicable)
