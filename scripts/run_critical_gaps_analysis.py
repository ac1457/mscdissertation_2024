#!/usr/bin/env python3
"""
Critical Gaps Analysis Runner
Runs comprehensive analysis to address identified methodological gaps:
1. Synthetic Text Validity Validation
2. Data Leakage Prevention in Temporal Split
3. Comprehensive Fairness Evaluation
4. Synthetic Data Contamination Analysis
5. Sensitivity Analysis at Different Risk Thresholds
6. Error Analysis and Misclassification Forensics
"""

import sys
import time
from pathlib import Path

def main():
    print("Critical Gaps Analysis Runner")
    print("=" * 60)
    print("This will address the following critical methodological gaps:")
    print("1. Synthetic Text Validity Risk - KS tests for NLP features")
    print("2. Data Leakage in Temporal Split - Strict as-of-date engineering")
    print("3. Incomplete Fairness Evaluation - Individual fairness measures")
    print("4. Synthetic Data Contamination - Ablation studies")
    print("5. Sensitivity Analysis - Precision-recall at risk thresholds")
    print("6. Error Analysis - Misclassification forensics")
    print()
    
    # Add analysis modules to path
    sys.path.append('analysis_modules')
    
    try:
        # Run critical gaps analysis
        print("Running Critical Gaps Analysis...")
        start_time = time.time()
        
        from critical_gaps_fixes import CriticalGapsFixes
        
        analyzer = CriticalGapsFixes(random_state=42)
        results = analyzer.run_comprehensive_analysis(None)
        
        analysis_time = time.time() - start_time
        print(f"   Critical gaps analysis completed in {analysis_time:.1f} seconds")
        
        # Summary
        if results:
            print("\n" + "=" * 60)
            print("CRITICAL GAPS ANALYSIS COMPLETE!")
            print("=" * 60)
            
            print("Analysis Results:")
            
            # Synthetic Text Validation
            if results['synthetic_validation']:
                print("  Synthetic Text Validation:")
                for feature, metrics in results['synthetic_validation'].items():
                    status = "SIGNIFICANT DIFFERENCE" if metrics['significant'] else "SIMILAR DISTRIBUTIONS"
                    print(f"    {feature}: {status} (p={metrics['p_value']:.4f})")
            else:
                print("  Synthetic Text Validation: No data available")
            
            # Temporal Leakage Prevention
            print("  Temporal Leakage Prevention:")
            print(f"    Features cleaned: {len(results['temporal_leakage_prevention']['cleaned_features'])}")
            print(f"    Future features removed: {len(results['temporal_leakage_prevention']['removed_future_features'])}")
            print(f"    Temporal splits created: {len(results['temporal_leakage_prevention']['temporal_splits'])}")
            
            # Other analyses
            print("  Fairness Evaluation: Requires model predictions")
            print("  Ablation Study: Requires model results")
            print("  Sensitivity Analysis: Requires model predictions")
            print("  Error Analysis: Requires model predictions")
            
            print(f"\nResults saved to: final_results/critical_gaps_analysis/")
            
        else:
            print("\nCritical gaps analysis failed!")
            print("Please check the error messages above.")
        
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