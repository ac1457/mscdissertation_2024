# Repository Organization Plan
## Lending Club Sentiment Analysis for Credit Risk Modeling

### Current Issues Identified:
1. **Scattered files** across multiple directories
2. **Duplicate content** in different locations
3. **Unclear naming conventions**
4. **Missing clear documentation**
5. **Mixed analysis scripts and results**
6. **No clear entry points**

### Proposed New Structure:

```
mscdissertation_2024/
├── README.md                           # Main project overview
├── requirements.txt                    # Dependencies
├── .gitignore                         # Git ignore rules
├── LICENSE                            # Academic license
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── data/                          # Data processing modules
│   │   ├── __init__.py
│   │   ├── data_loader.py             # Real data loading
│   │   ├── target_creation.py         # Target variable generation
│   │   └── preprocessing.py           # Data preprocessing
│   │
│   ├── analysis/                      # Analysis modules
│   │   ├── __init__.py
│   │   ├── eda.py                     # Exploratory data analysis
│   │   ├── feature_engineering.py     # Feature creation
│   │   ├── model_integration.py       # Model fusion approaches
│   │   ├── hyperparameter_analysis.py # Hyperparameter sensitivity
│   │   ├── ablation_studies.py        # Feature ablation
│   │   ├── statistical_validation.py  # Statistical testing
│   │   └── business_impact.py         # Business metrics
│   │
│   ├── models/                        # Model implementations
│   │   ├── __init__.py
│   │   ├── baseline_models.py         # Traditional models
│   │   ├── hybrid_models.py           # Text + tabular models
│   │   └── evaluation.py              # Model evaluation
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       ├── metrics.py                 # Custom metrics
│       ├── visualization.py           # Plotting functions
│       └── helpers.py                 # Helper functions
│
├── data/                              # Data files
│   ├── raw/                           # Original data
│   │   ├── lending_club_data.csv      # Raw Lending Club data
│   │   └── metadata.json              # Data metadata
│   ├── processed/                     # Processed data
│   │   ├── cleaned_data.csv           # Cleaned dataset
│   │   ├── synthetic_descriptions.csv # Generated descriptions
│   │   └── feature_engineered.csv     # Final features
│   └── external/                      # External data sources
│
├── results/                           # Analysis results
│   ├── figures/                       # Generated plots
│   │   ├── eda/                       # EDA plots
│   │   ├── model_performance/         # Model comparison plots
│   │   ├── feature_importance/        # SHAP plots
│   │   └── business_impact/           # Business metrics plots
│   │
│   ├── tables/                        # Results tables
│   │   ├── model_performance.csv      # Model comparison results
│   │   ├── feature_importance.csv     # Feature rankings
│   │   ├── statistical_tests.csv      # Statistical validation
│   │   └── business_metrics.csv       # Business impact metrics
│   │
│   └── reports/                       # Generated reports
│       ├── comprehensive_analysis.md  # Main analysis report
│       ├── methodology.md             # Methodology documentation
│       └── case_studies.md            # Case study examples
│
├── scripts/                           # Execution scripts
│   ├── run_complete_analysis.py       # Main analysis pipeline
│   ├── run_eda.py                     # EDA only
│   ├── run_model_comparison.py        # Model comparison
│   ├── run_feature_analysis.py        # Feature analysis
│   └── run_business_impact.py         # Business impact analysis
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb      # Data exploration
│   ├── 02_feature_engineering.ipynb   # Feature creation
│   ├── 03_model_development.ipynb     # Model development
│   └── 04_results_analysis.ipynb      # Results analysis
│
├── docs/                              # Documentation
│   ├── methodology.md                 # Detailed methodology
│   ├── data_dictionary.md             # Feature descriptions
│   ├── model_architecture.md          # Model details
│   ├── results_interpretation.md      # How to interpret results
│   └── reproducibility.md             # Reproducibility guide
│
└── tests/                             # Unit tests
    ├── __init__.py
    ├── test_data_processing.py        # Data processing tests
    ├── test_models.py                 # Model tests
    └── test_metrics.py                # Metrics tests
```

### Cleanup Tasks:

#### 1. **File Consolidation**
- [ ] Move all analysis modules to `src/analysis/`
- [ ] Consolidate duplicate files
- [ ] Remove unused files
- [ ] Standardize naming conventions

#### 2. **Data Organization**
- [ ] Organize data files by stage (raw, processed, external)
- [ ] Create clear data documentation
- [ ] Remove duplicate data files

#### 3. **Results Organization**
- [ ] Consolidate all results into `results/` directory
- [ ] Organize by type (figures, tables, reports)
- [ ] Create clear naming conventions

#### 4. **Documentation**
- [ ] Update main README.md
- [ ] Create comprehensive documentation
- [ ] Add clear entry points
- [ ] Document methodology

#### 5. **Code Quality**
- [ ] Add proper imports and dependencies
- [ ] Create clear entry point scripts
- [ ] Add unit tests
- [ ] Standardize code style

### Implementation Steps:

1. **Create new directory structure**
2. **Move and rename files systematically**
3. **Update imports and references**
4. **Create comprehensive documentation**
5. **Test all functionality**
6. **Update README and entry points**

### Benefits of New Structure:

1. **Clear separation** of concerns
2. **Easy navigation** for new users
3. **Reproducible analysis** pipeline
4. **Professional appearance**
5. **Academic standards** compliance
6. **Easy maintenance** and updates
