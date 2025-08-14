# Metrics Glossary

## Core Discrimination Metrics
- **AUC**: Area Under ROC Curve (0.5 = random, 1.0 = perfect)
- **PR-AUC**: Area Under Precision-Recall Curve (better for imbalanced data)
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Calibration Metrics
- **Brier Score**: Mean squared error of probability predictions (0 = perfect, 1 = worst)
- **ECE**: Expected Calibration Error - measures probability calibration quality
- **Calibration Slope**: Slope of calibration curve (1.0 = perfectly calibrated)
- **Calibration Intercept**: Intercept of calibration curve (0.0 = perfectly calibrated)

## Decision Utility Metrics
- **Lift@k%**: Ratio of default rate in top k% vs overall default rate
- **Capture Rate@k%**: Percentage of all defaults captured in top k%
- **Cost Savings**: Expected cost savings from model deployment

## Improvement Metrics
- **AUC_Improvement**: AUC_variant - AUC_traditional (positive = improvement)
- **Brier_Improvement**: Brier_traditional - Brier_variant (positive = improvement)
- **ECE_Improvement**: ECE_traditional - ECE_variant (positive = improvement)

## Statistical Testing
- **DeLong Test**: Statistical test comparing two AUCs using t-test on CV differences
- **Bootstrap CI**: 95% confidence interval from 1000 bootstrap resamples
- **CV Folds**: 5-fold stratified cross-validation

## Data Regimes
- **5% Regime**: Realistic default rate scenario (target 5%, actual ~16%)
- **10% Regime**: Realistic default rate scenario (target 10%, actual ~20%)
- **15% Regime**: Realistic default rate scenario (target 15%, actual ~25%)

## Feature Sets
- **Traditional**: Basic loan features (purpose, text characteristics)
- **Sentiment**: Traditional + sentiment analysis features
- **Hybrid**: Traditional + sentiment + interaction features

Generated: 2025-08-14T22:35:17.315060
