#!/usr/bin/env python3
"""
Enhanced Analysis Pipeline
Runs the complete enhanced analysis including:
1. Advanced text preprocessing
2. Entity extraction and fine-grained sentiment
3. Comprehensive ablation studies
4. Enhanced model integration
5. Detailed results analysis and visualization
"""

import sys
import time
from pathlib import Path

def main():
    print("Enhanced Analysis Pipeline")
    print("=" * 60)
    print("This will run the complete enhanced analysis with:")
    print("  - Advanced text preprocessing")
    print("  - Entity extraction for financial indicators")
    print("  - Fine-grained sentiment analysis")
    print("  - Comprehensive ablation studies")
    print("  - Enhanced model integration")
    print("  - Detailed results analysis")
    print()
    
    # Add analysis modules to path
    sys.path.append('analysis_modules')
    
    try:
        # Step 1: Run enhanced comprehensive analysis
        print("Step 1: Running Enhanced Comprehensive Analysis...")
        start_time = time.time()
        
        from enhanced_comprehensive_analysis import EnhancedComprehensiveAnalysis
        
        analysis = EnhancedComprehensiveAnalysis(random_state=42)
        results, ablation_results = analysis.run_enhanced_analysis()
        
        analysis_time = time.time() - start_time
        print(f"   Enhanced analysis completed in {analysis_time:.1f} seconds")
        
        # Step 2: Analyze results
        print("\nStep 2: Analyzing Results...")
        start_time = time.time()
        
        from enhanced_results_analyzer import EnhancedResultsAnalyzer
        
        analyzer = EnhancedResultsAnalyzer()
        report = analyzer.run_complete_analysis()
        
        analysis_time = time.time() - start_time
        print(f"   Results analysis completed in {analysis_time:.1f} seconds")
        
        # Step 3: Summary
        print("\n" + "=" * 60)
        print("ENHANCED ANALYSIS COMPLETE!")
        print("=" * 60)
        
        if report:
            summary = report['analysis_summary']
            print(f"Results Summary:")
            print(f"  - Total regimes analyzed: {summary['total_regimes']}")
            print(f"  - Total models tested: {summary['total_models']}")
            print(f"  - Best overall model: {summary['best_overall_model']}")
            print(f"  - Best improvement: {summary['best_overall_improvement']:.2f}%")
            print(f"  - Regimes with improvement: {summary['regimes_with_improvement']}/{summary['total_regimes']}")
            
            print(f"\nKey Files Generated:")
            print(f"  - final_results/enhanced_comprehensive/ (raw results)")
            print(f"  - final_results/enhanced_analysis/ (analysis and visualizations)")
            print(f"  - comprehensive_report.md (detailed report)")
            print(f"  - feature_importance.png (top features)")
            print(f"  - feature_group_performance.png (group comparison)")
            print(f"  - regime_performance.png (model comparison)")
            print(f"  - improvements.png (improvement visualization)")
        
        print(f"\nEnhanced analysis pipeline completed successfully!")
        print(f"Check the generated files for detailed insights.")
        
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