# Lending Club Sentiment Analysis for Credit Risk Modeling

**A Methodologically Rigorous Evaluation of Sentiment Features in Credit Risk Prediction**

## Project Overview

This dissertation project provides a comprehensive, methodologically rigorous evaluation of sentiment analysis features in credit risk modeling. While the results show modest improvements with no statistical significance, the methodological framework and statistical rigor represent a significant contribution to the field.

## Key Contributions

### Methodology
- **Rigorous statistical testing** with proper multiple comparison correction
- **Temporal validation** to prevent data leakage
- **Comprehensive baseline comparisons** (TF-IDF, lexicon, embeddings)
- **Feature isolation and ablation** studies
- **Proper calibration** with fold-based training
- **Permutation tests** to validate signal above noise floor

### Academic Value
- **Established boundaries** on sentiment feature effectiveness in credit risk
- **Reproducible methodology** with full documentation
- **Transparent reporting** of negative results
- **Framework for future research** in text-based credit modeling

## Results Summary

### Honest Assessment
- **No statistically significant improvements** after multiple comparison correction
- **Modest effect sizes** (ΔAUC < 0.025) that barely meet practical thresholds
- **Negative business impact** in incremental defaults captured
- **Weak overall discrimination** (AUC ~0.51-0.52)

### Key Findings
| Regime | Best Model | ΔAUC | Statistical Significance | Practical Threshold | Business Impact |
|--------|------------|------|------------------------|-------------------|-----------------|
| 5% Default | Hybrid | 0.0052 | No (p=0.414) | No | No (-29 defaults) |
| 10% Default | Hybrid | 0.0139 | No (p=0.080) | Yes | No (-14 defaults) |
| 15% Default | Hybrid | 0.0227 | No (p=0.054) | Yes | No (-16 defaults) |

## Methodological Rigor

### Statistical Testing
- **Primary test:** Label permutation (preregistered)
- **Secondary test:** Feature permutation
- **Multiple comparison correction:** Holm method
- **Bootstrap confidence intervals:** 1000 resamples with BCa method

### Temporal Validation
- **Time series cross-validation:** 5-fold splits
- **No future leakage:** Strict temporal ordering
- **Date range:** 2020-01-01 to 2047-05-18 (synthetic future dates)

### Feature Engineering
- **Traditional features:** Purpose, text metrics, financial indicators
- **Sentiment features:** LLM-based sentiment scores, confidence, polarity
- **Text baselines:** TF-IDF (879 dimensions), lexicon-based, embeddings
- **Interaction terms:** Sentiment × text length, word count, purpose

### Model Evaluation
- **Calibration:** Platt scaling within folds only
- **Metrics:** AUC, PR-AUC, Brier score, ECE, calibration slope/intercept
- **Business metrics:** Incremental defaults, cost-sensitive evaluation
- **Robustness:** Subgroup analysis, stability testing

## Project Structure

```
dissertation_results/
├── analysis_modules/
│   ├── corrected_rigorous_analysis.py      # Main corrected analysis
│   ├── methodologically_rigorous_analysis.py # Comprehensive implementation
│   ├── realistic_target_creation.py        # Target variable generation
│   ├── enhanced_metrics_implementation.py  # Enhanced metrics
│   └── main.py                            # Original main script
├── final_results/
│   ├── corrected_rigorous/                 # Corrected results
│   │   ├── consolidated_results.csv        # Main results table
│   │   ├── detailed_results.json           # Complete analysis
│   │   └── manifest.json                   # Reproducibility
│   └── methodologically_rigorous/          # Comprehensive results
├── data/
│   └── synthetic_loan_descriptions_with_realistic_targets.csv
├── CORRECTED_ANALYSIS_SUMMARY.md           # Corrected findings
├── METHODOLOGICALLY_RIGOROUS_ANALYSIS_SUMMARY.md
├── run_all.py                             # Main pipeline
├── requirements.txt                        # Dependencies
├── seeds.json                             # Reproducibility
└── metrics.json                           # Metrics snapshot
```

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Complete Analysis
```bash
python run_all.py
```

### Run Corrected Analysis
```bash
python analysis_modules/corrected_rigorous_analysis.py
```

## Key Insights

### What Works
- **Methodological framework** for rigorous evaluation
- **Statistical testing** with proper corrections
- **Temporal validation** to prevent leakage
- **Comprehensive baselines** for comparison

### What Doesn't Work
- **Current sentiment features** provide no significant improvements
- **LLM-based sentiment** may not be optimal for credit risk
- **Short text descriptions** limit sentiment extraction
- **Domain mismatch** between general sentiment and credit language

### Future Directions
- **Domain-specific sentiment** models for financial text
- **Longer, richer descriptions** with more emotional content
- **Alternative text features** (embeddings, topic modeling)
- **Different modeling approaches** (regression, survival analysis)

## Academic Contribution

### Methodological Innovation
- **Rigorous evaluation framework** for text features in credit risk
- **Proper temporal validation** methodology
- **Comprehensive baseline comparison** approach
- **Statistical rigor** with multiple testing corrections

### Scientific Value
- **Established upper bounds** on sentiment feature effectiveness
- **Negative results** that guide future research
- **Reproducible methodology** for the field
- **Transparent reporting** of limitations

## Reproducibility

### Version Control
- **Git repository** with complete history
- **Frozen configuration** for all analyses
- **Seeded randomness** for reproducible results

### Documentation
- **Manifest files** with complete configuration
- **Package versions** pinned for reproducibility
- **Analysis logs** with timestamps and hashes

## Contributing

This is a dissertation project, but the methodological framework and code are available for:
- **Academic research** in credit risk modeling
- **Text analysis** in financial applications
- **Statistical methodology** development
- **Reproducibility** studies

## License

Academic use permitted. Please cite appropriately if using the methodology or code.

## Author

**Aadhira Chavan**  
MSc Dissertation 2025
Lending Club Sentiment Analysis for Credit Risk Modeling

---

## Limitations


**Results Quality: HONESTLY NEGATIVE**  
**Business Value: NOT DEMONSTRATED**

**This project demonstrates that rigorous methodology and honest reporting of negative results are valuable contributions to the field, even when the primary hypothesis is not supported by the data.**
