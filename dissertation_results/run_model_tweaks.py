#!/usr/bin/env python3
"""
Model Tweaks and Results Presentation Pipeline
Runs comprehensive model tweaks analysis including:
1. Fusion method comparison (attention vs gated vs simple weighted)
2. Hyperparameter sensitivity analysis
3. Marginal cost quantification and ROI analysis
4. Sentiment-risk correlation visualization
5. Comprehensive reporting
"""

import sys
import time
from pathlib import Path

def main():
    print("Model Tweaks and Results Presentation Pipeline")
    print("=" * 60)
    print("This will run comprehensive model tweaks analysis:")
    print("  - Fusion method comparison")
    print("  - Hyperparameter sensitivity analysis")
    print("  - Marginal cost quantification")
    print("  - Sentiment-risk correlation visualization")
    print("  - ROI analysis for deployment decisions")
    print()
    
    # Add analysis modules to path
    sys.path.append('analysis_modules')
    
    try:
        # Run model tweaks analysis
        print("Running Model Tweaks Analysis...")
        start_time = time.time()
        
        from advanced_model_tweaks import AdvancedModelTweaks
        
        analysis = AdvancedModelTweaks(random_state=42)
        results = analysis.run_comprehensive_tweaks_analysis()
        
        analysis_time = time.time() - start_time
        print(f"   Model tweaks analysis completed in {analysis_time:.1f} seconds")
        
        # Summary
        print("\n" + "=" * 60)
        print("MODEL TWEAKS ANALYSIS COMPLETE!")
        print("=" * 60)
        
        if results:
            fusion_results, hyperparameter_results, cost_analysis, correlation_results = results
            
            print(f"Fusion Method Comparison Results:")
            best_fusion = max(fusion_results.items(), key=lambda x: x[1]['mean_auc'])
            print(f"  Best method: {best_fusion[0]} (AUC: {best_fusion[1]['mean_auc']:.4f})")
            
            for method, data in fusion_results.items():
                print(f"  {method}: AUC = {data['mean_auc']:.4f}, Time = {data['inference_time']:.4f}s")
            
            print(f"\nHyperparameter Sensitivity Results:")
            for dim, layer_results in hyperparameter_results.items():
                best_layers = max(layer_results.items(), key=lambda x: x[1])
                print(f"  Attention dim {dim}: {best_layers[0]} layers best (AUC: {best_layers[1]:.4f})")
            
            print(f"\nCost-Benefit Analysis:")
            best_roi = max(cost_analysis.items(), key=lambda x: x[1]['cost_benefit_ratio'])
            print(f"  Best ROI: {best_roi[0]} (ratio: {best_roi[1]['cost_benefit_ratio']:.2f})")
            
            for method, costs in cost_analysis.items():
                print(f"  {method}: ${costs['cost_per_prediction']:.6f}/pred, ROI = {costs['cost_benefit_ratio']:.2f}")
            
            print(f"\nSentiment-Risk Correlation:")
            print(f"  Correlation analysis completed")
            print(f"  Visualization saved to sentiment_risk_correlation.png")
        
        print(f"\nKey Files Generated:")
        print(f"  - final_results/model_tweaks/ (model tweaks results)")
        print(f"  - fusion_comparison.json (fusion method comparison)")
        print(f"  - hyperparameter_sensitivity.json (sensitivity analysis)")
        print(f"  - cost_analysis.json (ROI and cost analysis)")
        print(f"  - sentiment_risk_correlation.csv (correlation data)")
        print(f"  - sentiment_risk_correlation.png (visualization)")
        print(f"  - tweaks_summary_report.md (comprehensive report)")
        
        print(f"\nModel tweaks analysis pipeline completed successfully!")
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