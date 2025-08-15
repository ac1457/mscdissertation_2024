#!/usr/bin/env python3
"""
Decision Threshold Analysis Pipeline
Runs comprehensive decision threshold analysis including:
1. Optimal decision thresholds for text features
2. Feature importance at different thresholds
3. Interpretable decision rules creation
4. Threshold stability and robustness analysis
5. Comprehensive visualizations and reporting
"""

import sys
import time
from pathlib import Path

def main():
    print("Decision Threshold Analysis Pipeline")
    print("=" * 60)
    print("This will run comprehensive decision threshold analysis:")
    print("  - Optimal decision thresholds for text features")
    print("  - Feature importance at different thresholds")
    print("  - Interpretable decision rules creation")
    print("  - Threshold stability and robustness analysis")
    print("  - Comprehensive visualizations")
    print()
    
    # Add analysis modules to path
    sys.path.append('analysis_modules')
    
    try:
        # Run decision threshold analysis
        print("Running Decision Threshold Analysis...")
        start_time = time.time()
        
        from decision_threshold_analysis import DecisionThresholdAnalysis
        
        analysis = DecisionThresholdAnalysis(random_state=42)
        results = analysis.run_comprehensive_threshold_analysis()
        
        analysis_time = time.time() - start_time
        print(f"   Decision threshold analysis completed in {analysis_time:.1f} seconds")
        
        # Summary
        print("\n" + "=" * 60)
        print("DECISION THRESHOLD ANALYSIS COMPLETE!")
        print("=" * 60)
        
        if results:
            threshold_results, importance_results, decision_rules, stability_results = results
            
            print(f"Threshold Analysis Results:")
            best_feature = max(threshold_results.items(), key=lambda x: x[1]['optimal_auc'])
            print(f"  Best performing feature: {best_feature[0]} (AUC: {best_feature[1]['optimal_auc']:.4f})")
            
            for feature, data in threshold_results.items():
                print(f"  {feature}: threshold = {data['optimal_threshold']:.3f}, AUC = {data['optimal_auc']:.4f}")
            
            print(f"\nFeature Importance Analysis:")
            print(f"  Thresholds analyzed: {len(importance_results)}")
            print(f"  Features tracked: {len(importance_results[list(importance_results.keys())[0]]['top_features'])}")
            
            print(f"\nDecision Rules Created:")
            total_rules = sum(len(rules_data['rules']) for rules_data in decision_rules.values())
            print(f"  Total rules: {total_rules}")
            print(f"  Features with rules: {len(decision_rules)}")
            
            print(f"\nThreshold Stability Analysis:")
            most_stable = min(stability_results.items(), key=lambda x: x[1]['threshold_stability']['cv'])
            print(f"  Most stable feature: {most_stable[0]} (CV: {most_stable[1]['threshold_stability']['cv']:.3f})")
            
            for feature, stability in stability_results.items():
                cv = stability['threshold_stability']['cv']
                stability_level = "High" if cv < 0.1 else "Medium" if cv < 0.2 else "Low"
                print(f"  {feature}: {stability_level} stability (CV: {cv:.3f})")
        
        print(f"\nKey Files Generated:")
        print(f"  - final_results/decision_thresholds/ (threshold analysis results)")
        print(f"  - threshold_analysis.json (optimal thresholds for all features)")
        print(f"  - feature_importance_at_thresholds.json (importance evolution)")
        print(f"  - decision_rules.json (interpretable decision rules)")
        print(f"  - threshold_stability.json (bootstrap stability analysis)")
        print(f"  - decision_thresholds_analysis.png (comprehensive visualization)")
        print(f"  - detailed_threshold_analysis.png (detailed feature analysis)")
        print(f"  - decision_rules_summary.png (rules distribution)")
        print(f"  - threshold_summary_report.md (comprehensive report)")
        
        print(f"\nDecision threshold analysis pipeline completed successfully!")
        print(f"Check the generated files for comprehensive threshold insights.")
        
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