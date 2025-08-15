#!/usr/bin/env python3
"""
Hyperparameter Sensitivity Analysis Pipeline
Runs comprehensive hyperparameter sensitivity analysis including:
1. Attention heads sensitivity (1, 2, 4, 8, 16, 32 heads)
2. FinBERT layer selection sensitivity (last 1, 2, 4, 8, all layers)
3. Combined hyperparameter analysis
4. Comprehensive visualizations and reporting
"""

import sys
import time
from pathlib import Path

def main():
    print("Hyperparameter Sensitivity Analysis Pipeline")
    print("=" * 60)
    print("This will run comprehensive hyperparameter sensitivity analysis:")
    print("  - Attention heads sensitivity (1-32 heads)")
    print("  - FinBERT layer selection sensitivity")
    print("  - Combined hyperparameter analysis")
    print("  - Performance stability analysis")
    print("  - Comprehensive visualizations")
    print()
    
    # Add analysis modules to path
    sys.path.append('analysis_modules')
    
    try:
        # Run hyperparameter sensitivity analysis
        print("Running Hyperparameter Sensitivity Analysis...")
        start_time = time.time()
        
        from hyperparameter_sensitivity_analysis import HyperparameterSensitivityAnalysis
        
        analysis = HyperparameterSensitivityAnalysis(random_state=42)
        results = analysis.run_comprehensive_sensitivity_analysis()
        
        analysis_time = time.time() - start_time
        print(f"   Hyperparameter sensitivity analysis completed in {analysis_time:.1f} seconds")
        
        # Summary
        print("\n" + "=" * 60)
        print("HYPERPARAMETER SENSITIVITY ANALYSIS COMPLETE!")
        print("=" * 60)
        
        if results:
            attention_heads_results, finbert_layers_results, combined_results = results
            
            print(f"Attention Heads Sensitivity Results:")
            best_heads = max(attention_heads_results.items(), key=lambda x: x[1]['mean_auc'])
            print(f"  Best configuration: {best_heads[0]} heads (AUC: {best_heads[1]['mean_auc']:.4f})")
            
            for n_heads, data in attention_heads_results.items():
                print(f"  {n_heads} heads: AUC = {data['mean_auc']:.4f} ± {data['std_auc']:.4f}")
            
            print(f"\nFinBERT Layer Selection Results:")
            best_layers = max(finbert_layers_results.items(), key=lambda x: x[1]['mean_auc'])
            print(f"  Best configuration: {best_layers[0]} (AUC: {best_layers[1]['mean_auc']:.4f})")
            
            for layer_config, data in finbert_layers_results.items():
                print(f"  {layer_config}: AUC = {data['mean_auc']:.4f} ± {data['std_auc']:.4f}")
            
            print(f"\nCombined Analysis Results:")
            best_combined = max(combined_results.items(), key=lambda x: x[1]['auc'])
            print(f"  Best combined: {best_combined[0]} (AUC: {best_combined[1]['auc']:.4f})")
            
            for config, data in combined_results.items():
                print(f"  {config}: AUC = {data['auc']:.4f}")
        
        print(f"\nKey Files Generated:")
        print(f"  - final_results/hyperparameter_sensitivity/ (sensitivity results)")
        print(f"  - attention_heads_sensitivity.json (attention heads analysis)")
        print(f"  - finbert_layers_sensitivity.json (layer selection analysis)")
        print(f"  - combined_sensitivity.json (combined analysis)")
        print(f"  - hyperparameter_sensitivity_analysis.png (comprehensive visualization)")
        print(f"  - attention_heads_detailed.png (detailed attention heads plot)")
        print(f"  - finbert_layers_detailed.png (detailed layer selection plot)")
        print(f"  - sensitivity_summary_report.md (comprehensive report)")
        
        print(f"\nHyperparameter sensitivity analysis pipeline completed successfully!")
        print(f"Check the generated files for detailed hyperparameter insights.")
        
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