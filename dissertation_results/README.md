# Lending Club Sentiment Analysis for Credit Risk Modeling

## Real Data Analysis with Synthetic Text Generation

This project analyzes the **real Kaggle Lending Club dataset** (2.2M+ loans) with synthetic text generation for missing descriptions and other text fields.

### Key Features

- **Real Data**: Uses actual Lending Club loan data from Kaggle
- **Synthetic Text Generation**: Generates realistic loan descriptions for missing text fields
- **Enhanced Features**: Comprehensive feature engineering including sentiment analysis
- **Advanced Models**: Multiple fusion approaches and hyperparameter optimization
- **Robust Evaluation**: Statistical rigor with bootstrap CIs, permutation tests, and temporal validation

### Dataset Information

- **Source**: Kaggle Lending Club dataset (wordsforthewise/lending-club)
- **Size**: ~2.2M loan records (sampled for computational efficiency)
- **Text Generation**: Synthetic descriptions generated for missing text fields
- **Features**: 50+ engineered features including sentiment, text complexity, and financial indicators

### Analysis Pipeline

1. **Real Data Processing**: Download and clean Kaggle dataset
2. **Text Generation**: Create synthetic descriptions for missing text fields
3. **Feature Engineering**: Extract sentiment, text complexity, and financial features
4. **Model Development**: Multiple fusion approaches and hyperparameter optimization
5. **Evaluation**: Comprehensive statistical evaluation with real-world metrics

### Results

- **Real-world applicability**: Results based on actual lending data
- **Methodological contribution**: Framework for combining real financial data with synthetic text
- **Business insights**: Practical implications for credit risk modeling
- **Academic rigor**: Robust statistical evaluation and validation

### Files Structure

```
dissertation_results/
├── analysis_modules/          # Analysis scripts
├── data/
│   └── real_lending_club/     # Real processed data
├── final_results/             # Analysis results
└── run_*.py                   # Execution scripts
```

### Quick Start

1. **Process Real Data**:
   ```bash
   python run_real_data_processing.py
   ```

2. **Run Enhanced Analysis**:
```bash
python run_enhanced_analysis.py
```

3. **Run Advanced Integration**:
```bash
python run_advanced_integration.py
```

### Key Improvements

- **Real Data Foundation**: Based on actual Lending Club loans
- **Synthetic Text Enhancement**: Realistic descriptions for missing text fields
- **Comprehensive Evaluation**: Statistical rigor and business metrics
- **Methodological Innovation**: Framework for real+synthetic data integration

### Academic Contribution

This work demonstrates a novel approach to credit risk modeling by:
- Combining real financial data with synthetic text generation
- Providing a framework for handling missing text data in financial applications
- Delivering robust statistical evaluation of sentiment-enhanced credit models
- Contributing to the literature on multi-modal financial data analysis

### Future Work

- Scale to full 2.2M record dataset
- Implement real-time text generation
- Explore additional text features and embeddings
- Develop production-ready deployment pipeline
