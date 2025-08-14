# Metrics Glossary - Lending Club Sentiment Analysis

## Metric Definitions and Standards

### **Core Discrimination Metrics**

#### **AUC (Area Under ROC Curve)**
- **Definition**: Area under the Receiver Operating Characteristic curve
- **Range**: 0.5 (random) to 1.0 (perfect discrimination)
- **Interpretation**: Probability that a randomly selected positive instance is ranked higher than a randomly selected negative instance
- **Precision**: 4 decimal places (e.g., 0.5866)
- **Formula**: `AUC = P(score_positive > score_negative)`

#### **PR-AUC (Precision-Recall AUC)**
- **Definition**: Area under the Precision-Recall curve
- **Range**: 0.0 to 1.0
- **Interpretation**: Better metric for imbalanced datasets; focuses on positive class performance
- **Precision**: 4 decimal places
- **Formula**: `PR-AUC = ∫ Precision(Recall) dRecall`

#### **KS Statistic (Kolmogorov-Smirnov)**
- **Definition**: Maximum difference between cumulative distribution functions of positive and negative classes
- **Range**: 0.0 to 1.0
- **Interpretation**: Higher values indicate better separation between classes
- **Precision**: 4 decimal places

### **Calibration Metrics**

#### **Brier Score**
- **Definition**: Mean squared error between predicted probabilities and actual outcomes
- **Range**: 0.0 (perfect calibration) to 1.0 (worst calibration)
- **Interpretation**: Lower is better; measures probability calibration accuracy
- **Precision**: 4 decimal places
- **Formula**: `Brier = (1/N) * Σ(y_true - y_pred)²`

#### **Expected Calibration Error (ECE)**
- **Definition**: Expected absolute difference between predicted confidence and accuracy
- **Range**: 0.0 (perfect calibration) to 1.0
- **Interpretation**: Lower is better; measures reliability of probability estimates
- **Precision**: 4 decimal places
- **Formula**: `ECE = Σ|accuracy(bin_i) - confidence(bin_i)| * P(bin_i)`

### **Decision Utility Metrics**

#### **Lift@k%**
- **Definition**: Ratio of default rate in top k% of predictions vs overall default rate
- **Range**: 0.0 to ∞
- **Interpretation**: Higher is better; measures decision utility at specific threshold
- **Precision**: 2 decimal places
- **Formula**: `Lift@k% = (default_rate_top_k / overall_default_rate)`

#### **Gain@k%**
- **Definition**: Default rate captured in top k% of predictions
- **Range**: 0.0 to 1.0
- **Interpretation**: Higher is better; percentage of defaults identified in top k%
- **Precision**: 3 decimal places
- **Formula**: `Gain@k% = (defaults_in_top_k / total_defaults)`

#### **Capture Rate@k%**
- **Definition**: Default rate among top k% of predictions
- **Range**: 0.0 to 1.0
- **Interpretation**: Higher is better; probability of default in top k%
- **Precision**: 3 decimal places

### **Improvement Metrics**

#### **AUC_Improvement**
- **Definition**: `Variant_AUC - Traditional_AUC` (same model)
- **Sign Convention**: Positive = improvement, Negative = degradation
- **Precision**: 4 decimal places
- **Example**: +0.0226 means 0.0226 AUC points better than traditional

#### **Improvement_Percent**
- **Definition**: `(AUC_Improvement / Traditional_AUC) * 100`
- **Sign Convention**: Positive = improvement, Negative = degradation
- **Precision**: 2 decimal places
- **Example**: +3.86% means 3.86% relative improvement

#### **Brier_Improvement**
- **Definition**: `Traditional_Brier - Variant_Brier` (same model)
- **Sign Convention**: **Positive = improvement** (lower Brier is better)
- **Precision**: 4 decimal places
- **Example**: +0.0015 means 0.0015 Brier points better (lower) than traditional

#### **ECE_Improvement**
- **Definition**: `Traditional_ECE - Variant_ECE` (same model)
- **Sign Convention**: **Positive = improvement** (lower ECE is better)
- **Precision**: 4 decimal places

#### **PR_AUC_Improvement**
- **Definition**: `Variant_PR_AUC - Traditional_PR_AUC` (same model)
- **Sign Convention**: Positive = improvement, Negative = degradation
- **Precision**: 4 decimal places

#### **Lift_Improvement**
- **Definition**: `Variant_Lift - Traditional_Lift` (same model)
- **Sign Convention**: Positive = improvement, Negative = degradation
- **Precision**: 2 decimal places

### **Statistical Testing**

#### **DeLong Test p-value**
- **Definition**: Statistical test for comparing ROC AUCs
- **Range**: 0.0 to 1.0
- **Interpretation**: p < 0.05 indicates significant difference
- **Precision**: Scientific notation for very small values (< 0.001), 4 decimals otherwise
- **Example**: < 1e-15, 0.9469

#### **FDR-Adjusted p-value**
- **Definition**: Benjamini-Hochberg multiple comparison correction
- **Range**: 0.0 to 1.0
- **Interpretation**: Controls false discovery rate across multiple comparisons
- **Precision**: Same as raw p-values

#### **Confidence Intervals**
- **Definition**: Bootstrap 95% confidence intervals
- **Method**: 1000 resamples, percentile method
- **Precision**: 4 decimal places
- **Example**: (0.5758, 0.5965)

### **Significance Levels**
- `***`: p < 0.001 (highly significant)
- `**`: p < 0.01 (significant)
- `*`: p < 0.05 (marginally significant)
- `ns`: p ≥ 0.05 (not significant)

### **Decimal Precision Standards**

#### **Primary Metrics (4 decimals)**
- AUC, PR-AUC, KS Statistic
- Brier Score, ECE
- All improvement deltas (absolute values)
- Confidence interval bounds

#### **Percentage Metrics (2 decimals)**
- Improvement_Percent
- Lift@k% values
- Lift improvements

#### **Rates and Proportions (3 decimals)**
- Default rates
- Capture rates
- Gain@k% values

#### **P-values**
- Scientific notation: < 1e-15, 3.44e-14
- Regular notation: 0.9469, 0.1234

### **Effect Size Interpretation**

#### **AUC Improvements**
- **Small**: 0.001-0.01 (1-10 basis points)
- **Medium**: 0.01-0.03 (10-30 basis points)
- **Large**: > 0.03 (30+ basis points)

#### **Brier Improvements**
- **Small**: 0.0001-0.001
- **Medium**: 0.001-0.005
- **Large**: > 0.005

#### **Lift Improvements**
- **Small**: 0.1-0.5
- **Medium**: 0.5-1.0
- **Large**: > 1.0

### **Industry Benchmarks**

#### **AUC Benchmarks**
- **Credit Cards**: ~0.65
- **Personal Loans**: ~0.70
- **Mortgage**: ~0.75
- **Current Best**: 0.6327 (5% default rate)

#### **Lift Benchmarks**
- **Good**: 1.5-2.0
- **Very Good**: 2.0-3.0
- **Excellent**: > 3.0

### **Data Regimes**

#### **Balanced Experimental Regime**
- **Default Rate**: 51.3% (artificial balance)
- **Purpose**: Internal benchmarking only
- **Limitation**: Inflates relative improvements

#### **Realistic Prevalence Regimes**
- **Default Rates**: 5%, 10%, 15%
- **Purpose**: External interpretation
- **Advantage**: Representative of real-world scenarios

### **Reporting Standards**

#### **Required Information**
1. **Absolute improvements** before percentage improvements
2. **Statistical significance** with p-values
3. **Confidence intervals** for primary metrics
4. **Effect size context** (small/medium/large)
5. **Industry benchmark comparison**

#### **Transparency Requirements**
1. **Sample sizes** for each regime
2. **Feature counts** by variant
3. **Random seeds** for reproducibility
4. **Methodological limitations**
5. **Multiple comparison corrections** 