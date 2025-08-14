# Lending Club Sentiment Analysis for Credit Risk Modeling

## Dissertation Project Overview

This project investigates the integration of sentiment analysis features into traditional credit risk modeling frameworks using Lending Club data.

## Project Structure

```
dissertation_results/
├── final_results/          # All final CSV results
│   ├── realistic_prevalence_results.csv    # Realistic default rate scenarios
│   ├── balanced_regime_results.csv         # Balanced experimental regime
│   ├── fixed_results_no_leakage.csv        # Clean results without leakage
│   └── enhanced_results_with_validation.csv # Statistically validated results
├── visualizations/         # All final plots and charts
│   ├── comprehensive_lift_charts.png       # Lift analysis visualizations
│   ├── comprehensive_calibration_plots.png # Calibration analysis
│   └── comprehensive_roc_curves.png        # ROC curve comparisons
├── methodology/            # Documentation and conclusions
│   ├── methodological_documentation.txt    # Complete methodology
│   ├── synthetic_text_methodology_report.txt # Synthetic text generation
│   └── revised_conclusions.txt             # Final conclusions
├── analysis_modules/       # Analysis code
│   ├── main.py                            # Main execution script
│   ├── fix_leakage_and_rebalance.py       # Leakage fix and rebalancing
│   ├── synthetic_text_documentation.py    # Synthetic text documentation
│   └── revised_conclusions_analysis.py    # Revised conclusions analysis
├── data/                   # Data files
├── models/                 # Model files
├── README.md               # This file
└── DISSERTATION_SUMMARY.md # Complete dissertation summary
```

## Key Findings

- **Modest but consistent improvements** in credit risk modeling with sentiment analysis
- **Realistic prevalence scenarios** show incremental rather than transformative gains
- **Statistical significance** confirmed with proper validation
- **Methodological rigor** maintained throughout analysis

## Usage

1. **Run Analysis**: `python analysis_modules/main.py --fix-leakage-and-rebalance`
2. **Generate Documentation**: `python analysis_modules/main.py --document-synthetic-text`
3. **View Results**: Check `final_results/` directory
4. **Review Visualizations**: Check `visualizations/` directory

## Requirements

See `requirements.txt` for all dependencies.

## Academic Contribution

This work provides:
- Empirical investigation of sentiment analysis in credit modeling
- Methodological innovation in synthetic text generation
- Framework for addressing data scarcity in financial research
- Honest assessment of modest but meaningful improvements

## Status

✅ **DISSERTATION READY** - All analysis complete, validated, and documented
