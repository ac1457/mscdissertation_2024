#!/usr/bin/env python3
"""
EDA and Data Preprocessing Pipeline
Runs comprehensive exploratory data analysis and data preprocessing including:
1. Data exploration and summary statistics
2. Missing data analysis and handling
3. Text data cleaning and preprocessing
4. Feature engineering and enhancement
5. Comprehensive EDA visualizations
"""

import sys
import time
from pathlib import Path

def main():
    print("EDA and Data Preprocessing Pipeline")
    print("=" * 60)
    print("This will run comprehensive EDA and data preprocessing:")
    print("  - Data exploration and summary statistics")
    print("  - Missing data analysis and handling")
    print("  - Text data cleaning and preprocessing")
    print("  - Feature engineering and enhancement")
    print("  - Comprehensive EDA visualizations")
    print()
    
    # Add analysis modules to path
    sys.path.append('analysis_modules')
    
    try:
        # Run EDA and preprocessing
        print("Running EDA and Data Preprocessing...")
        start_time = time.time()
        
        from eda_and_preprocessing import EDAAndPreprocessing
        
        analysis = EDAAndPreprocessing(random_state=42)
        results = analysis.run_comprehensive_eda_and_preprocessing()
        
        analysis_time = time.time() - start_time
        print(f"   EDA and preprocessing completed in {analysis_time:.1f} seconds")
        
        # Summary
        print("\n" + "=" * 60)
        print("EDA AND DATA PREPROCESSING COMPLETE!")
        print("=" * 60)
        
        if results:
            cleaned_df, enhanced_df, eda_results = results
            
            print(f"Dataset Information:")
            print(f"  Original shape: {eda_results['dataset_info']['shape']}")
            print(f"  Cleaned shape: {cleaned_df.shape}")
            print(f"  Enhanced shape: {enhanced_df.shape}")
            print(f"  Memory usage: {eda_results['dataset_info']['memory_usage_mb']:.2f} MB")
            
            print(f"\nData Cleaning Results:")
            missing_cols = eda_results['missing_data_analysis']['columns_with_missing']
            print(f"  Columns with missing data: {len(missing_cols)}")
            if missing_cols:
                print(f"  Missing columns: {missing_cols}")
            
            print(f"\nFeature Engineering Results:")
            new_features = len(enhanced_df.columns) - len(cleaned_df.columns)
            print(f"  New features created: {new_features}")
            print(f"  Total features: {len(enhanced_df.columns)}")
            
            print(f"\nText Analysis Results:")
            if 'text_analysis' in eda_results:
                text_stats = eda_results['text_analysis']['text_length_stats']
                print(f"  Average text length: {text_stats['mean']:.1f} characters")
                print(f"  Average word count: {eda_results['text_analysis']['word_count_stats']['mean']:.1f} words")
        
        print(f"\nKey Files Generated:")
        print(f"  - eda_plots/ (comprehensive EDA visualizations)")
        print(f"  - fast_eda_plots/ (quick EDA insights)")
        print(f"  - final_results/eda_and_preprocessing/ (preprocessing results)")
        print(f"  - cleaned_data.csv (cleaned dataset)")
        print(f"  - enhanced_data.csv (enhanced dataset)")
        print(f"  - eda_results.json (comprehensive EDA results)")
        print(f"  - preprocessing_summary.md (preprocessing summary)")
        
        print(f"\nEDA and data preprocessing pipeline completed successfully!")
        print(f"Check the generated files for comprehensive data insights.")
        
    except ImportError as e:
        print(f"Error: Could not import required modules. {e}")
        print("Make sure you're running this from the dissertation_results directory.")
        return 1
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 