# Detailed Feature Analysis Report

## TF-IDF Baseline Comparison

### Performance Comparison

**target_5%:**
- Sentiment AUC: 0.5043
- TF-IDF AUC: 0.5152
- Improvement: -0.0109

**target_10%:**
- Sentiment AUC: 0.5035
- TF-IDF AUC: 0.5136
- Improvement: -0.0101

**target_15%:**
- Sentiment AUC: 0.5055
- TF-IDF AUC: 0.5000
- Improvement: 0.0055


## Error Analysis

### Model Performance Comparison
- Tabular Model AUC: 0.8548
- Hybrid Model AUC: 1.0000
- Improvement: 0.1452

### Error Patterns
- Hybrid Improvements: 1753 cases
- Hybrid Worsens: 5237 cases
- Improvement Rate: 0.175
- Worsen Rate: 0.524

## Case Studies

### Key Findings

**Thin File Improvement:**
- Description: Planning elective surgery with good insurance coverage....
- Tabular Prediction: 0.070
- Hybrid Prediction: 0.610
- True Label: 1

**Narrative Correction:**
- Description: Looking to consolidate multiple credit cards with high interest rates. I have excellent payment hist...
- Tabular Prediction: 0.251
- Hybrid Prediction: 0.700
- True Label: 1


## Misclassification Examples

### Sample Misclassifications
