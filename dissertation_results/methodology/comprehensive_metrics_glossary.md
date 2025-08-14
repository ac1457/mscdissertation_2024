# Comprehensive Metrics Glossary - Lending Club Sentiment Analysis

## **METRIC DEFINITIONS AND STANDARDS**

### **Core Discrimination Metrics**

#### **AUC (Area Under ROC Curve)**
- **Definition**: Probability that a randomly selected positive instance is ranked higher than a randomly selected negative instance
- **Range**: 0.0 to 1.0 (0.5 = random, 1.0 = perfect)
- **Interpretation**: Higher is better
- **Format**: 4 decimal places (e.g., 0.6234)
- **Example**: AUC = 0.6234 indicates 62.34% probability of correct ranking

#### **PR-AUC (Area Under Precision-Recall Curve)**
- **Definition**: Area under the precision-recall curve, measuring precision-recall trade-off
- **Range**: 0.0 to 1.0
- **Interpretation**: Higher is better, especially important for imbalanced datasets
- **Format**: 4 decimal places (e.g., 0.5678)
- **Example**: PR-AUC = 0.5678 indicates good precision-recall balance

#### **KS (Kolmogorov-Smirnov Statistic)**
- **Definition**: Maximum difference between cumulative distribution functions of positive and negative classes
- **Range**: 0.0 to 1.0
- **Interpretation**: Higher is better
- **Format**: 4 decimal places (e.g., 0.2345)
- **Example**: KS = 0.2345 indicates 23.45% maximum separation

### **Calibration Metrics**

#### **Brier Score**
- **Definition**: Mean squared error between predicted probabilities and actual outcomes
- **Range**: 0.0 to 1.0 (0.0 = perfect calibration)
- **Interpretation**: Lower is better
- **Format**: 4 decimal places (e.g., 0.2345)
- **Example**: Brier = 0.2345 indicates moderate calibration quality

#### **ECE (Expected Calibration Error)**
- **Definition**: Expected absolute difference between predicted probabilities and actual frequencies within bins
- **Range**: 0.0 to 1.0 (0.0 = perfect calibration)
- **Interpretation**: Lower is better
- **Format**: 4 decimal places (e.g., 0.0456)
- **Example**: ECE = 0.0456 indicates good calibration

#### **Calibration Slope**
- **Definition**: Slope of logistic regression fit to predicted probabilities vs actual outcomes
- **Range**: 0.0 to ∞ (1.0 = perfect calibration)
- **Interpretation**: Closer to 1.0 is better
- **Format**: 4 decimal places (e.g., 0.8765)
- **Example**: Calibration slope = 0.8765 indicates slight overconfidence

### **Decision Utility Metrics**

#### **Lift@k%**
- **Definition**: Ratio of default rate in top k% of predictions vs overall default rate
- **Range**: 0.0 to ∞ (1.0 = no lift, >1.0 = positive lift)
- **Interpretation**: Higher is better
- **Format**: 2 decimal places (e.g., 1.23)
- **Example**: Lift@10% = 1.23 means 23% higher default rate in top 10%

#### **Gain@k%**
- **Definition**: Percentage of all defaults captured in top k% of predictions
- **Range**: 0.0% to 100.0%
- **Interpretation**: Higher is better
- **Format**: 1 decimal place (e.g., 45.6%)
- **Example**: Gain@10% = 45.6% means 45.6% of defaults in top 10%

#### **Capture Rate@k%**
- **Definition**: Percentage of positive cases captured in top k% of predictions
- **Range**: 0.0% to 100.0%
- **Interpretation**: Higher is better
- **Format**: 1 decimal place (e.g., 67.8%)
- **Example**: Capture Rate@10% = 67.8% means 67.8% of positives in top 10%

### **Improvement Metrics**

#### **AUC_Improvement**
- **Definition**: AUC_Variant - AUC_Traditional
- **Sign Convention**: **POSITIVE values indicate improvement** (higher AUC is better)
- **Format**: +0.0234 (always show sign)
- **Example**: AUC_Improvement = +0.0234 means variant has higher AUC

#### **Improvement_Percent**
- **Definition**: (AUC_Improvement / AUC_Traditional) * 100
- **Sign Convention**: **POSITIVE values indicate improvement**
- **Format**: +3.45% (always show sign and %)
- **Example**: Improvement_Percent = +3.45% means 3.45% relative improvement

#### **Brier_Improvement**
- **Definition**: Brier_Traditional - Brier_Variant
- **Sign Convention**: **NEGATIVE values indicate improvement** (lower Brier is better)
- **Format**: -0.0023 (always show sign)
- **Example**: Brier_Improvement = -0.0023 means variant has better calibration

#### **ECE_Improvement**
- **Definition**: ECE_Traditional - ECE_Variant
- **Sign Convention**: **NEGATIVE values indicate improvement** (lower ECE is better)
- **Format**: -0.0012 (always show sign)
- **Example**: ECE_Improvement = -0.0012 means variant has better calibration

#### **PR_AUC_Improvement**
- **Definition**: PR_AUC_Variant - PR_AUC_Traditional
- **Sign Convention**: **POSITIVE values indicate improvement** (higher PR-AUC is better)
- **Format**: +0.0156 (always show sign)
- **Example**: PR_AUC_Improvement = +0.0156 means variant has higher PR-AUC

#### **Lift_Improvement**
- **Definition**: Lift_Variant - Lift_Traditional
- **Sign Convention**: **POSITIVE values indicate improvement** (higher lift is better)
- **Format**: +0.12 (always show sign)
- **Example**: Lift_Improvement = +0.12 means variant has higher lift

### **Statistical Testing**

#### **DeLong Test p-value**
- **Definition**: p-value from DeLong test comparing two ROC AUCs
- **Range**: 0.0 to 1.0
- **Interpretation**: <0.05 = statistically significant difference
- **Format**: Scientific notation for small values (e.g., 1.23e-05)
- **Example**: DeLong p-value = 1.23e-05 indicates highly significant difference

#### **FDR-Adjusted p-value**
- **Definition**: False Discovery Rate adjusted p-value for multiple comparisons
- **Range**: 0.0 to 1.0
- **Interpretation**: <0.05 = statistically significant after correction
- **Format**: Scientific notation for small values (e.g., 2.45e-04)
- **Example**: FDR p-value = 2.45e-04 indicates significant after correction

#### **Confidence Intervals**
- **Definition**: Bootstrap 95% confidence intervals for metric estimates
- **Format**: [lower, upper] with 4 decimal places
- **Example**: [0.6123, 0.6345] means 95% CI for AUC

### **Significance Levels**

#### **Statistical Significance**
- **p < 0.001**: Highly significant (***)
- **p < 0.01**: Very significant (**)
- **p < 0.05**: Significant (*)
- **p ≥ 0.05**: Not significant (ns)

#### **Practical Significance**
- **Small effect**: 0.01-0.03 AUC improvement
- **Medium effect**: 0.03-0.05 AUC improvement
- **Large effect**: >0.05 AUC improvement

### **Decimal Precision Standards**

#### **Primary Metrics**
- **AUC**: 4 decimal places (0.6234)
- **PR-AUC**: 4 decimal places (0.5678)
- **Brier Score**: 4 decimal places (0.2345)
- **ECE**: 4 decimal places (0.0456)
- **Calibration Slope**: 4 decimal places (0.8765)

#### **Improvement Metrics**
- **AUC_Improvement**: 4 decimal places (+0.0234)
- **Improvement_Percent**: 2 decimal places (+3.45%)
- **Brier_Improvement**: 4 decimal places (-0.0023)
- **PR_AUC_Improvement**: 4 decimal places (+0.0156)

#### **Decision Utility**
- **Lift@k**: 2 decimal places (1.23)
- **Gain@k**: 1 decimal place (45.6%)
- **Capture Rate@k**: 1 decimal place (67.8%)

#### **Statistical Tests**
- **p-values**: Scientific notation for <0.001 (1.23e-05)
- **Confidence Intervals**: 4 decimal places ([0.6123, 0.6345])

### **Industry Benchmarks**

#### **Typical Production AUC Ranges**
- **Personal Loans**: ~0.70
- **Credit Cards**: ~0.65
- **Mortgage**: ~0.75
- **Small Business**: ~0.68

#### **Calibration Quality**
- **Excellent**: Brier < 0.1, ECE < 0.01
- **Good**: Brier < 0.2, ECE < 0.05
- **Fair**: Brier < 0.3, ECE < 0.1
- **Poor**: Brier ≥ 0.3, ECE ≥ 0.1

#### **Lift Benchmarks**
- **Strong**: Lift@10% > 2.0
- **Moderate**: Lift@10% 1.5-2.0
- **Weak**: Lift@10% 1.0-1.5
- **Poor**: Lift@10% < 1.0

### **Data Regimes**

#### **Balanced Experimental Regime**
- **Default Rate**: 51.3% (NOT representative)
- **Use**: Internal exploratory benchmarking only
- **Caveat**: Absolute improvements inflated vs real-world prevalence

#### **Realistic Prevalence Regimes**
- **5% Default Rate**: Representative of low-risk portfolios
- **10% Default Rate**: Representative of moderate-risk portfolios
- **15% Default Rate**: Representative of high-risk portfolios
- **Use**: Primary for external interpretation

### **Reporting Standards**

#### **Required Elements**
- **Metric value** with appropriate precision
- **Confidence intervals** for all estimates
- **Statistical significance** with p-values
- **Effect size** interpretation
- **Practical significance** assessment

#### **Negative Delta Handling**
- **Always report** negative improvements explicitly
- **Provide context** for negative results
- **Include variance/noise** explanations
- **No suppression** of unfavorable results

#### **Consistency Requirements**
- **Same precision** across all tables
- **Consistent sign conventions** throughout
- **Standardized formatting** for all metrics
- **Clear definitions** in all reports

### **Quality Checks**

#### **Metric Validation**
- **AUC bounds**: 0.0 ≤ AUC ≤ 1.0
- **Brier bounds**: 0.0 ≤ Brier ≤ 1.0
- **Probability bounds**: 0.0 ≤ p ≤ 1.0
- **No NaN values** in final results

#### **Statistical Validation**
- **Sample size** sufficient for statistical power
- **Cross-validation** properly implemented
- **Bootstrap resampling** with adequate iterations
- **Multiple comparison correction** applied

#### **Reproducibility**
- **Random seeds** logged and fixed
- **Environment versions** documented
- **Data processing** pipeline reproducible
- **Results traceable** to source code

This glossary ensures consistent, accurate, and transparent reporting across all dissertation documentation. 