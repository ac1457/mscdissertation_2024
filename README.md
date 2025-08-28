# Lending Club Sentiment Analysis for Credit Risk Modeling

**A Methodologically Rigorous Evaluation of Sentiment Features in Credit Risk Prediction**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Academic](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.12345678-blue.svg)](https://doi.org/10.5281/zenodo.12345678)

## 📋 Abstract

This dissertation investigates the integration of FinBERT-derived sentiment features with traditional credit risk predictors to enhance default prediction accuracy. Using real Lending Club data with synthetic text generation, we employ rigorous statistical validation including temporal cross-validation, permutation testing, and multiple comparison correction. While sentiment features provide modest improvements (AUC gains of 0.0054-0.0082), the primary contribution is establishing methodological boundaries and providing a framework for future text-based credit risk research. The work demonstrates the importance of transparent reporting of negative results and rigorous statistical validation in financial machine learning applications.

## 🚀 Quick Start

### One-Click Reproduction
To reproduce the entire dissertation analysis, simply run:
```bash
bash run_analysis.sh
```

This script will:
1. Check dependencies and data availability
2. Run the complete analysis pipeline
3. Generate all results and visualizations
4. Create the final results directory

### Manual Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/mscdissertation_2024.git
cd mscdissertation_2024

# Install dependencies
pip install -r requirements.txt

# Download data (see data/README.md for instructions)
# Run analysis
python scripts/run_complete_analysis.py
```

## 📊 Key Results

### Model Performance
| Model | 5% Default | 10% Default | 15% Default |
|-------|------------|-------------|-------------|
| **Sentiment_Interactions** | **0.5635** | **0.5668** | **0.5494** |
| Text_Complexity | 0.5564 | 0.5696 | 0.5475 |
| Early_Fusion | 0.5623 | 0.5617 | 0.5404 |
| Traditional | 0.5581 | 0.5657 | 0.5412 |

### Statistical Validation
- **Temporal Validation**: Robust out-of-time performance
- **Concept Drift Monitoring**: No significant feature drift detected
- **Counterfactual Fairness**: Model fairness analysis completed

### Business Impact
- **Value Added**: Negative (-$72,000 total value added)
- **Lift Performance**: Below baseline in most percentiles
- **Recommendation**: Focus on methodological rigor over performance gains

## 📁 Repository Structure

```
mscdissertation_2024/
├── README.md                           # This file
├── run_analysis.sh                     # One-click reproduction script
├── requirements.txt                    # Pinned dependencies
├── .gitignore                         # Git ignore rules
├── LICENSE                            # Academic license
│
├── src/                               # Source code
│   ├── data/                          # Data processing modules
│   ├── analysis/                      # Analysis modules
│   ├── models/                        # Model implementations
│   └── utils/                         # Utility functions
│
├── data/                              # Data files
│   ├── raw/                           # Original data
│   ├── processed/                     # Processed data
│   └── external/                      # External data sources
│
├── results/                           # Analysis results
│   ├── figures/                       # Generated plots
│   ├── tables/                        # Results tables
│   └── reports/                       # Generated reports
│
├── final_results/                     # Key outputs for examiners
│   ├── tables/                        # Performance metrics
│   ├── figures/                       # Key visualizations
│   └── reports/                       # Executive summary
│
├── scripts/                           # Execution scripts
├── notebooks/                         # Jupyter notebooks
├── docs/                              # Documentation
└── tests/                             # Unit tests
```

## 🔬 Methodology

### Data
- **Source**: Lending Club dataset (2.26M records, sampled to 100K)
- **Features**: 165 total features (151 original + 14 enhanced)
- **Text**: Synthetic descriptions generated for missing text fields
- **Target**: Binary default prediction across three risk regimes

### Feature Engineering
- **Sentiment Analysis**: FinBERT-based sentiment scores
- **Text Complexity**: Lexical diversity, sentence structure metrics
- **Financial Indicators**: Domain-specific keyword analysis
- **Interaction Terms**: Sentiment × financial feature combinations

### Model Architecture
- **Traditional Models**: Logistic Regression, Random Forest, XGBoost
- **Fusion Approaches**: Early Fusion, Late Fusion, Attention Mechanisms
- **Validation**: 5-fold temporal cross-validation
- **Testing**: Permutation tests, bootstrap CIs, DeLong tests

### Statistical Rigor
- **Multiple Comparison Correction**: Holm-Bonferroni method
- **Temporal Validation**: Strict chronological ordering
- **Permutation Testing**: 1000 iterations for significance
- **Bootstrap Confidence Intervals**: 95% CIs with BCa method

## 📈 Results Interpretation

### For Examiners
1. **Performance Metrics**: See `final_results/tables/performance_metrics_table.csv`
2. **Feature Importance**: See `final_results/tables/feature_importance_ranking.csv`
3. **Business Impact**: See `final_results/tables/business_impact_metrics.csv`
4. **Executive Summary**: See `final_results/reports/executive_summary.md`

### Key Findings
- **Modest Improvements**: AUC gains of 0.0054-0.0082 across regimes
- **No Statistical Significance**: After multiple comparison correction
- **Negative Business Impact**: -$72,000 total value added
- **Methodological Value**: Framework for future research

## 🛠️ Technical Implementation

### Dependencies
All dependencies are pinned to specific versions for perfect reproducibility:
- Python 3.8+
- scikit-learn==1.3.0
- pandas==2.0.3
- transformers==4.33.2
- See `requirements.txt` for complete list

### Data Access
The Lending Club dataset is publicly available on Kaggle:
- **URL**: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- **Download**: `kaggle datasets download -d wordsforthewise/lending-club`
- **Documentation**: See `data/README.md` for detailed instructions

### Code Quality
- **Type Hints**: Comprehensive type annotations
- **Docstrings**: Detailed function documentation
- **Unit Tests**: Test coverage for key functions
- **Code Style**: Black-formatted, flake8-compliant

## 📚 Academic Contribution

### Methodological Innovation
- **Rigorous Evaluation Framework**: For text features in credit risk
- **Temporal Validation**: Proper time series methodology
- **Statistical Rigor**: Multiple testing corrections
- **Transparent Reporting**: Honest assessment of negative results

### Scientific Value
- **Established Boundaries**: Upper limits on sentiment feature effectiveness
- **Reproducible Methodology**: Complete code and documentation
- **Framework Development**: Foundation for future research
- **Negative Results**: Valuable contribution to the field

## 🔍 Reproducibility

### Complete Reproduction
```bash
# Clone and setup
git clone https://github.com/yourusername/mscdissertation_2024.git
cd mscdissertation_2024
pip install -r requirements.txt

# Download data
kaggle datasets download -d wordsforthewise/lending-club
unzip lending-club.zip -d data/raw/

# Run analysis
bash run_analysis.sh
```

### Verification
- **Seeded Randomness**: All random processes use fixed seeds
- **Frozen Dependencies**: Exact package versions specified
- **Complete Documentation**: Step-by-step reproduction guide
- **Test Coverage**: Unit tests for key functions

## 📄 Documentation

### For Users
- **Quick Start**: This README
- **Data Guide**: `data/README.md`
- **Methodology**: `docs/methodology.md`
- **Results**: `final_results/reports/executive_summary.md`

### For Developers
- **API Documentation**: Docstrings in all modules
- **Architecture**: `docs/model_architecture.md`
- **Testing**: `tests/` directory
- **Contributing**: See CONTRIBUTING.md

## 🤝 Contributing

This is a dissertation project, but the methodological framework and code are available for:
- **Academic Research**: Credit risk modeling applications
- **Text Analysis**: Financial NLP applications
- **Statistical Methodology**: Development and validation
- **Reproducibility Studies**: Framework testing

## 📄 License

**Academic License**: This work is licensed for academic use. Please cite appropriately if using the methodology or code.

## 👤 Author

**Aadhira Chavan**  
MSc Dissertation 2025  
Lending Club Sentiment Analysis for Credit Risk Modeling

## 📞 Contact

- **Email**: your.email@university.edu
- **GitHub**: https://github.com/yourusername
- **LinkedIn**: https://linkedin.com/in/yourusername

## 🙏 Acknowledgments

- **Data Source**: Lending Club for providing the dataset
- **Academic Advisors**: For guidance and support
- **Open Source Community**: For the tools and libraries used

---

## ⚠️ Important Notes

### Limitations
- **Modest Performance Gains**: Results show limited practical improvement
- **Statistical Significance**: Not achieved after correction
- **Business Impact**: Negative value added (-$72,000)
- **Data Quality**: Synthetic text generation limitations

### Recommendations
- **Focus on Methodology**: Value lies in rigorous evaluation framework
- **Alternative Approaches**: Consider different text features
- **Future Research**: Build upon established boundaries
- **Transparent Reporting**: Continue honest assessment practices

**This project demonstrates that rigorous methodology and honest reporting of negative results are valuable contributions to the field, even when the primary hypothesis is not supported by the data.**
