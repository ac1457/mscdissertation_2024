# Updated Metrics Glossary - Lending Club Sentiment Analysis

## Core Discrimination Metrics

### **AUC (Area Under ROC Curve)**
- **Definition**: Probability that a randomly selected positive instance is ranked higher than a randomly selected negative instance
- **Range**: 0.0 to 1.0 (0.5 = random, 1.0 = perfect)
- **Interpretation**: Higher is better
- **Precision**: 4 decimal places (e.g., 0.6234)
- **Formula**: `AUC = P(score_positive > score_negative)`

### **PR-AUC (Area Under Precision-Recall Curve)**
- **Definition**: Area under the precision-recall curve, measuring precision-recall trade-off
- **Range**: 0.0 to 1.0
- **Interpretation**: Higher is better, especially important for imbalanced datasets
- **Precision**: 4 decimal places
- **Formula**: `PR-AUC = ∫ Precision(Recall) dRecall`

### **KS (Kolmogorov-Smirnov Statistic)**
- **Definition**: Maximum difference between cumulative distribution functions of positive and negative classes
- **Range**: 0.0 to 1.0
- **Interpretation**: Higher is better
- **Precision**: 4 decimal places
- **Formula**: `KS = max|F_positive(x) - F_negative(x)|`

## Calibration Metrics

### **Brier Score**
- **Definition**: Mean squared error between predicted probabilities and actual outcomes
- **Range**: 0.0 to 1.0 (0.0 = perfect calibration)
- **Interpretation**: Lower is better
- **Precision**: 4 decimal places
- **Formula**: `Brier = (1/N) * Σ(pred_i - actual_i)²`

### **Brier Improvement**
- **Definition**: `Brier_Improvement = Brier_Traditional - Brier_Variant`
- **Sign Convention**: **NEGATIVE values indicate improvement** (lower Brier is better)
- **Range**: -1.0 to 1.0
- **Interpretation**: Negative = better calibration, Positive = worse calibration
- **Precision**: 4 decimal places
- **Example**: `Brier_Improvement = -0.0023` means variant has better calibration

### **ECE (Expected Calibration Error)**
- **Definition**: Expected absolute difference between predicted confidence and accuracy
- **Range**: 0.0 to 1.0 (0.0 = perfect calibration)
- **Interpretation**: Lower is better
- **Precision**: 4 decimal places
- **Formula**: `ECE = Σ(|confidence_bin - accuracy_bin| * bin_size) / total_size`

### **ECE Improvement**
- **Definition**: `ECE_Improvement = ECE_Traditional - ECE_Variant`
- **Sign Convention**: **NEGATIVE values indicate improvement** (lower ECE is better)
- **Precision**: 4 decimal places

## Decision Utility Metrics

### **Lift@k%**
- **Definition**: Ratio of default rate in top k% of predictions vs overall default rate
- **Range**: 0.0 to ∞ (1.0 = no lift, >1.0 = positive lift)
- **Interpretation**: Higher is better
- **Precision**: 2 decimal places
- **Formula**: `Lift@k% = (default_rate_in_top_k%) / (overall_default_rate)`

### **Gain@k%**
- **Definition**: Proportion of all defaults captured in top k% of predictions
- **Range**: 0.0 to 1.0
- **Interpretation**: Higher is better
- **Precision**: 3 decimal places
- **Formula**: `Gain@k% = (defaults_in_top_k%) / (total_defaults)`

### **Capture Rate@k%**
- **Definition**: Default rate in top k% of predictions
- **Range**: 0.0 to 1.0
- **Interpretation**: Higher is better
- **Precision**: 3 decimal places
- **Formula**: `Capture_Rate@k% = (defaults_in_top_k%) / (total_predictions_in_top_k%)`

## Improvement Metrics

### **AUC Improvement**
- **Definition**: `AUC_Improvement = AUC_Variant - AUC_Traditional`
- **Sign Convention**: **POSITIVE values indicate improvement** (higher AUC is better)
- **Range**: -1.0 to 1.0
- **Precision**: 4 decimal places
- **Example**: `AUC_Improvement = 0.0234` means variant has higher AUC

### **Improvement Percent**
- **Definition**: `Improvement_Percent = (AUC_Improvement / AUC_Traditional) * 100`
- **Sign Convention**: **POSITIVE values indicate improvement**
- **Range**: -100% to ∞
- **Precision**: 2 decimal places
- **Example**: `Improvement_Percent = 3.45%` means 3.45% relative improvement

### **PR-AUC Improvement**
- **Definition**: `PR_AUC_Improvement = PR_AUC_Variant - PR_AUC_Traditional`
- **Sign Convention**: **POSITIVE values indicate improvement**
- **Precision**: 4 decimal places

### **Lift Improvement**
- **Definition**: `Lift_Improvement = Lift_Variant - Lift_Traditional`
- **Sign Convention**: **POSITIVE values indicate improvement**
- **Precision**: 2 decimal places

## Statistical Testing

### **DeLong Test p-value**
- **Definition**: p-value from DeLong test comparing two ROC AUCs
- **Range**: 0.0 to 1.0
- **Interpretation**: <0.05 = statistically significant difference
- **Precision**: Scientific notation for very small values (<1e-15), 6 decimal places otherwise
- **Significance Levels**: *** (p<0.001), ** (p<0.01), * (p<0.05), ns (not significant)

### **FDR-Adjusted p-value**
- **Definition**: Benjamini-Hochberg FDR-corrected p-value for multiple comparisons
- **Range**: 0.0 to 1.0
- **Interpretation**: <0.05 = statistically significant after correction
- **Precision**: 6 decimal places

### **Confidence Intervals**
- **Definition**: Bootstrap 95% confidence intervals
- **Format**: `[lower, upper]`
- **Precision**: 4 decimal places
- **Example**: `AUC = 0.6234 (95% CI: 0.6123-0.6345)`

## Effect Size Interpretation

### **Small Effect**
- **AUC Improvement**: 0.01-0.03
- **Improvement Percent**: 1-3%
- **Practical Significance**: Minimal but measurable

### **Medium Effect**
- **AUC Improvement**: 0.03-0.05
- **Improvement Percent**: 3-5%
- **Practical Significance**: Meaningful improvement

### **Large Effect**
- **AUC Improvement**: >0.05
- **Improvement Percent**: >5%
- **Practical Significance**: Substantial improvement

## Industry Benchmarks

### **Typical Production AUC Ranges**
- **Credit Cards**: 0.65-0.70
- **Personal Loans**: 0.70-0.75
- **Mortgages**: 0.75-0.80
- **Small Business**: 0.60-0.65

### **Calibration Benchmarks**
- **Good Brier Score**: <0.20
- **Excellent Brier Score**: <0.10
- **Good ECE**: <0.05
- **Excellent ECE**: <0.02

## Data Regimes

### **Balanced Experimental Regime**
- **Default Rate**: 51.3%
- **Purpose**: Internal benchmarking only
- **Caveat**: Not representative of real-world prevalence
- **Use**: Controlled comparative analysis

### **Realistic Prevalence Regimes**
- **Default Rates**: 5%, 10%, 15%
- **Purpose**: External interpretation and practical assessment
- **Method**: Stratified downsampling of majority class
- **Use**: Primary results for external stakeholders

## Reporting Standards

### **Decimal Precision Standards**
- **AUC, PR-AUC, KS**: 4 decimal places
- **Brier Score, ECE**: 4 decimal places
- **Improvement Metrics**: 4 decimal places (absolute), 2 decimal places (percent)
- **Lift Metrics**: 2 decimal places
- **Capture Rates**: 3 decimal places
- **P-values**: Scientific notation (<1e-15), 6 decimal places otherwise
- **Confidence Intervals**: 4 decimal places

### **Significance Reporting**
- **Statistical**: Report p-values and confidence intervals
- **Practical**: Report absolute improvements before percentages
- **Effect Size**: Use standardized effect size interpretations
- **Multiple Comparisons**: Report FDR-adjusted p-values

### **Transparency Requirements**
- **Sample Sizes**: Report train/test sizes and class counts
- **Sampling Method**: Document exact sampling methodology
- **Random Seeds**: Log all random seeds for reproducibility
- **Data Leakage**: Explicitly state removed features
- **Limitations**: Acknowledge synthetic data and experimental nature

## Key Clarifications

### **Brier Improvement Sign Convention**
- **CRITICAL**: Brier_Improvement = Brier_Traditional - Brier_Variant
- **Negative values = Better calibration** (lower Brier is better)
- **Positive values = Worse calibration** (higher Brier is worse)
- **Example**: `Brier_Improvement = -0.0023` means the variant has better calibration

### **AUC vs PR-AUC**
- **AUC**: Good for balanced datasets, less sensitive to class imbalance
- **PR-AUC**: Better for imbalanced datasets, focuses on positive class
- **Recommendation**: Report both for comprehensive assessment

### **Statistical vs Practical Significance**
- **Statistical**: Based on p-values and sample size
- **Practical**: Based on effect size and business impact
- **Requirement**: Report both, especially for large sample sizes

### **Confidence Intervals**
- **Overlapping CIs**: Do not automatically imply non-significance
- **Primary Test**: DeLong test for AUC comparisons
- **Bootstrap Method**: Percentile method with 1000 resamples

This glossary ensures consistent interpretation and reporting across all analyses in the dissertation. 