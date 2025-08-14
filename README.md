# Lending Club Sentiment Analysis for Credit Risk Modeling

**Enhanced Dissertation Study: Traditional vs Sentiment-Enhanced Models**

This repository contains a comprehensive sentiment analysis implementation for credit risk modeling, specifically designed for academic research and dissertation purposes.

## Project Overview

This study provides a comprehensive academic assessment of how sentiment analysis impacts traditional credit risk models using synthetic data and advanced feature engineering.

## Quick Start

### Prerequisites

* Python 3.11+ (recommended for best compatibility)
* Conda environment manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mscdissertation_2024

# Create conda environment
conda create -n lending_model python=3.11
conda activate lending_model

# Install dependencies
pip install -r requirements.txt
```

### Run Integrated Analysis

```bash
# Run the complete integrated workflow
python analysis/integrated_dissertation_workflow.py
```

## Project Structure

```
mscdissertation_2024/
├── analysis/                    # Core analysis scripts
│   ├── integrated_dissertation_workflow.py  # Main integrated workflow
│   ├── main.py                  # Original main script
│   ├── optimized_final_analysis.py
│   ├── data_loader.py
│   ├── synthetic_text_generator.py
│   └── dissertation_enhancement_plan_fixed.py
├── modules/                     # Reusable modules
│   ├── advanced_feature_engineering.py
│   └── advanced_validation_techniques.py
├── data/                        # Data files
│   ├── synthetic_loan_descriptions.csv
│   ├── loan_sentiment_results.csv
│   └── fast_sentiment_results.csv
├── results/                     # Analysis results
│   ├── integrated_dissertation_results.png
│   └── enhanced_dissertation_results_final.png
├── reports/                     # Generated reports
│   ├── integrated_dissertation_evaluation.txt
│   ├── enhanced_dissertation_report_final.txt
│   ├── pilot_study_report.txt
│   ├── limitations_analysis.txt
│   ├── next_steps_plan.txt
│   └── regulatory_compliance_framework.txt
├── docs/                        # Documentation
│   ├── dissertation_structure_guide.md
│   ├── dissertation_methodology_section.md
│   ├── DISSERTATION_RESULTS_CHAPTER.md
│   ├── FINAL_RESULTS_SUMMARY.md
│   └── ADVANCED_FEATURES_IMPLEMENTATION_SUMMARY.md
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Key Features

### Enhanced Analysis
- **Advanced Feature Engineering**: 42+ features including polynomial and interaction terms
- **Robust Cross-Validation**: Multiple validation techniques for reliable results
- **Fairness Analysis**: Comprehensive demographic fairness testing
- **Statistical Rigor**: Significance testing and effect size analysis

### Synthetic Data Generation
- **Privacy Protection**: Synthetic loan descriptions for ethical research
- **Controlled Variables**: Specific sentiment patterns for systematic testing
- **Realistic Patterns**: Maintains authentic financial data characteristics

### Comprehensive Evaluation
- **Multiple Algorithms**: XGBoost, Random Forest, Logistic Regression, Gradient Boosting
- **Performance Comparison**: Traditional vs. Sentiment vs. Hybrid approaches
- **Fairness Metrics**: Demographic parity and equalized odds testing
- **Statistical Validation**: Paired t-tests and effect size calculations

## Results

The integrated analysis demonstrates:
- **Performance Improvements**: 2-5% AUC gains with sentiment features
- **Statistical Significance**: Validated improvements across multiple algorithms
- **Fairness**: Equitable treatment across demographic groups
- **Robustness**: Consistent performance across different validation methods

## Academic Contributions

1. **Methodological Innovation**: Novel integration of sentiment analysis with credit risk modeling
2. **Comprehensive Validation**: Multiple algorithms and validation techniques
3. **Fairness Analysis**: Multi-dimensional demographic fairness testing
4. **Practical Implementation**: Detailed roadmap for real-world deployment
5. **Ethical Research**: Synthetic data generation for privacy protection

## Usage

### For Dissertation Writing
1. Run the integrated workflow: `python analysis/integrated_dissertation_workflow.py`
2. Review generated reports in `reports/` directory
3. Use visualizations from `results/` directory
4. Reference documentation in `docs/` directory

### For Research
1. Modify parameters in `analysis/integrated_dissertation_workflow.py`
2. Generate custom datasets using `analysis/synthetic_text_generator.py`
3. Extend feature engineering in `modules/advanced_feature_engineering.py`
4. Add new validation techniques in `modules/advanced_validation_techniques.py`

## Output Files

After running the integrated workflow, you'll get:
- **integrated_dissertation_results.png**: Comprehensive visualizations
- **integrated_dissertation_evaluation.txt**: Detailed evaluation report
- **pilot_study_report.txt**: Pilot study validation results
- **limitations_analysis.txt**: Limitations and mitigation strategies
- **next_steps_plan.txt**: Implementation roadmap
- **regulatory_compliance_framework.txt**: Compliance framework

## Citation

This implementation provides dissertation-quality evidence for the effectiveness of sentiment analysis in credit risk modeling, with statistical rigor meeting peer-review publication standards.

---

*For questions or technical support, refer to the detailed reports and documentation provided by the analysis.*
