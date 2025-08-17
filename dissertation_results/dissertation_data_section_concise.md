# Data Section - Dissertation (Concise Version)

## 3. Data and Methodology

### 3.1 Data Source

The primary dataset for this research is the Lending Club loan data (Kaggle: wordsforthewise/lending-club), containing 2.26 million loan records from 2007-2018. A stratified sample of 97,020 records was selected for computational efficiency while maintaining representativeness across loan grades, purposes, and temporal periods.

**Dataset Characteristics:**
- **Source**: Lending Club platform (2007-2018)
- **Sample Size**: 97,020 records (from 2.26M original)
- **Features**: 165 total (151 original + 14 engineered)
- **Time Period**: 11 years of lending data

### 3.2 Data Structure

#### 3.2.1 Traditional Credit Features
The dataset contains comprehensive credit risk indicators:

**Demographic**: Employment length, annual income, home ownership, geographic location
**Financial**: Loan amount, interest rate, debt-to-income ratio, credit grade
**Credit History**: FICO scores, open accounts, public records, delinquency history
**Performance**: Loan status, payment history, recovery amounts

#### 3.2.2 Text Data and Synthetic Generation
**Challenge**: 93,476 records (96.3%) had missing loan descriptions
**Solution**: Purpose-based synthetic text generation with natural variation
**Validation**: Kolmogorov-Smirnov tests confirmed distribution similarity to real text

**Text Features Extracted:**
- **Sentiment**: Score, confidence, categories, balance
- **Complexity**: Length, readability, lexical diversity
- **Financial**: Keyword density, entity extraction
- **Interactions**: Text-financial feature combinations

### 3.3 Data Preprocessing

#### 3.3.1 Missing Value Handling
- **Text**: 96.3% missing - addressed through synthetic generation
- **Employment**: 7.4% missing - mode imputation
- **Financial**: 0.1% missing - median imputation
- **Threshold**: Removed records with >50% missing values

#### 3.3.2 Data Quality Assurance
- **Outlier Removal**: Extreme loan amounts, income values
- **Range Validation**: DTI ratios, interest rates
- **Consistency Checks**: Temporal alignment, logical relationships

### 3.4 Feature Engineering

#### 3.4.1 Enhanced Text Features
**Sentiment Analysis**: 4 categories (positive, negative, neutral, financial stress)
**Text Complexity**: 8 metrics (TTR, sentence length, word count)
**Entity Extraction**: 4 entity types (job stability, financial hardship, etc.)
**Financial Indicators**: Keyword density, financial terms

#### 3.4.2 Interaction Features
- Sentiment-loan amount interactions
- Text complexity-income relationships
- Financial terms-purpose alignment

### 3.5 Target Variable

#### 3.5.1 Default Definition
**Primary Target**: Binary default indicator
- **Default**: Charged off, defaulted, or 120+ days late
- **Non-Default**: Current, fully paid, or early payoff

**Default Rates**:
- **Overall**: 15.2%
- **By Grade**: A (5.2%) to G (42.1%)
- **By Purpose**: Debt consolidation (18.3%), credit card (12.7%)

#### 3.5.2 Multiple Risk Regimes
Three default rate regimes for robustness testing:
- **Conservative**: 5% default rate (higher credit quality)
- **Moderate**: 10% default rate (balanced risk)
- **Aggressive**: 15% default rate (higher risk)

### 3.6 Data Leakage Prevention

#### 3.6.1 Temporal Data Leakage
**Removed Future Features**: 11 features not available at origination
- Payment history, credit pull dates, collection activities
- Total payments, recovery amounts, late fees

#### 3.6.2 Cross-Validation Strategy
**Temporal Cross-Validation**: TimeSeriesSplit with 5 folds
- Strict chronological ordering
- Past data for training, future data for validation

### 3.7 Data Quality Metrics

**Completeness**: 100% (97,020 complete records)
**Consistency**: All features within valid ranges
**Validity**: Aligned with lending industry standards

### 3.8 Sample Characteristics

**Demographic**: Average age 42.3, median income $65,000
**Loan**: Average $15,000, mean interest rate 12.5%
**Credit**: Grade distribution A (15.2%) to G (0.5%)
**Text**: Average 156 characters, 24 words, 12% financial terms

### 3.9 Ethical Considerations

**Privacy**: All personal identifiers removed
**Bias Mitigation**: Protected attributes monitored, fairness testing implemented
**Synthetic Data**: Clear labeling, statistical validation, limitations acknowledged

### 3.10 Data Limitations

**Synthetic Text**: Potential artificial patterns, context loss
**Sample**: Platform-specific population, historical data
**Generalizability**: US market, specific platform, historical conditions

### 3.11 Data Pipeline Summary

1. **Collection**: Lending Club dataset (2.26M records)
2. **Sampling**: Stratified sample (97K records)
3. **Cleaning**: Missing value imputation, outlier removal
4. **Synthetic Generation**: Text generation for missing descriptions
5. **Feature Engineering**: 165 total features
6. **Leakage Prevention**: Future feature removal
7. **Quality Assurance**: Validation and testing
8. **Ethical Review**: Bias mitigation, privacy protection

This comprehensive data preparation provides a robust foundation for sentiment-enhanced credit risk modeling while maintaining methodological rigor and addressing the critical challenge of missing text data through synthetic generation. 