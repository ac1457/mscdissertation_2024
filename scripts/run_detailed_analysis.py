#!/usr/bin/env python3
"""
Detailed Feature Analysis Pipeline
Runs comprehensive detailed analysis including:
1. TF-IDF baseline comparison
2. Detailed SHAP feature importance analysis
3. Text feature breakdown (sentiment, structure, entity, style)
4. Error analysis with case studies
5. Misclassification examples
6. Comprehensive reporting
"""

import sys
import time
from pathlib import Path

def main():
    print("Detailed Feature Analysis Pipeline")
    print("=" * 60)
    print("This will run comprehensive detailed analysis:")
    print("  - TF-IDF baseline comparison")
    print("  - SHAP feature importance analysis")
    print("  - Text feature breakdown")
    print("  - Error analysis with case studies")
    print("  - Misclassification examples")
    print("  - Comprehensive reporting")
    print()
    
    # Add analysis modules to path
    sys.path.append('analysis_modules')
    
    try:
        # Run detailed feature analysis
        print("Running Detailed Feature Analysis...")
        start_time = time.time()
        
        from detailed_feature_analysis import DetailedFeatureAnalysis
        
        analysis = DetailedFeatureAnalysis(random_state=42)
        results = analysis.run_comprehensive_analysis()
        
        analysis_time = time.time() - start_time
        print(f"   Detailed analysis completed in {analysis_time:.1f} seconds")
        
        # Summary
        print("\n" + "=" * 60)
        print("DETAILED FEATURE ANALYSIS COMPLETE!")
        print("=" * 60)
        
        if results:
            tfidf_results, shap_results, error_results, misclassification_examples = results
            
            print(f"TF-IDF Comparison Results:")
            for regime, data in tfidf_results.items():
                print(f"  {regime}: Sentiment vs TF-IDF improvement = {data['improvement']:.4f}")
            
            if shap_results:
                print(f"\nSHAP Analysis Results:")
                top_features = shap_results['feature_importance'].head(5)
                for _, row in top_features.iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
            
            print(f"\nError Analysis Results:")
            error_data = error_results['error_analysis']
            print(f"  Hybrid vs Tabular AUC improvement: {error_data['improvement_auc']:.4f}")
            print(f"  Cases where hybrid improves: {error_data['hybrid_improvements_count']}")
            print(f"  Cases where hybrid worsens: {error_data['hybrid_worsens_count']}")
            
            print(f"\nCase Studies Generated: {len(error_results['case_studies'])}")
            print(f"Misclassification Examples: {len(misclassification_examples)}")
        
        print(f"\nKey Files Generated:")
        print(f"  - final_results/detailed_analysis/ (detailed results)")
        print(f"  - tfidf_comparison.json (TF-IDF vs sentiment comparison)")
        print(f"  - shap_analysis.json (SHAP feature importance)")
        print(f"  - error_analysis.json (error patterns and case studies)")
        print(f"  - misclassification_examples.json (example misclassifications)")
        print(f"  - detailed_summary_report.md (comprehensive report)")
        print(f"  - shap_feature_importance.png (SHAP visualization)")
        print(f"  - shap_top_features.png (top features plot)")
        
        print(f"\nDetailed feature analysis pipeline completed successfully!")
        print(f"Check the generated files for comprehensive insights.")
        
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