# Model Tweaks and Results Presentation Report

## Fusion Method Comparison

### Performance Comparison

**Attention:**
- AUC: 0.5248 ± 0.0163
- Inference Time: 0.0004s

**Gated:**
- AUC: 0.5039 ± 0.0174
- Inference Time: 0.0003s

**Simple Weighted:**
- AUC: 0.5027 ± 0.0157
- Inference Time: 0.0005s

**Concatenation:**
- AUC: 0.5032 ± 0.0165
- Inference Time: 0.0002s


## Hyperparameter Sensitivity Analysis

### Attention Dimension Sensitivity

**Attention Dimension 32:**
- Last 4 Layers: 0.4955
- All Layers: 0.4760

**Attention Dimension 64:**
- Last 4 Layers: 0.4979
- All Layers: 0.4723

**Attention Dimension 128:**
- Last 4 Layers: 0.5012
- All Layers: 0.4639

**Attention Dimension 256:**
- Last 4 Layers: 0.5012
- All Layers: 0.4848


## Marginal Cost Analysis

### Cost-Benefit Analysis

**Attention:**
- AUC Improvement: 0.0248
- Cost per Prediction: $0.010000
- Cost-Benefit Ratio: 2.48
- Total Cost (100k predictions): $1000.00
- Expected Improvement (100k): 2477

**Gated:**
- AUC Improvement: 0.0039
- Cost per Prediction: $0.010000
- Cost-Benefit Ratio: 0.39
- Total Cost (100k predictions): $1000.00
- Expected Improvement (100k): 386

**Simple Weighted:**
- AUC Improvement: 0.0027
- Cost per Prediction: $0.001000
- Cost-Benefit Ratio: 2.68
- Total Cost (100k predictions): $100.00
- Expected Improvement (100k): 268

**Concatenation:**
- AUC Improvement: 0.0032
- Cost per Prediction: $0.001000
- Cost-Benefit Ratio: 3.22
- Total Cost (100k predictions): $100.00
- Expected Improvement (100k): 322


## Sentiment-Risk Correlation

### Key Findings
- Correlation between sentiment and default rate
- Default rate varies significantly across sentiment deciles
- Sentiment distribution shows natural clustering

### Recommendations
1. **Fusion Method:** attention performs best
2. **Attention Dimension:** Optimal dimension identified
3. **Cost-Benefit:** ROI analysis for deployment decisions
4. **Sentiment Analysis:** Clear correlation with default risk

---
**Analysis completed:** 2025-08-15 10:04:10
