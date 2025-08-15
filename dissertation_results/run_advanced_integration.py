#!/usr/bin/env python3
"""
Advanced Model Integration Pipeline
Runs comprehensive model integration analysis including:
1. Late fusion (concatenation)
2. Early fusion (separate models)
3. Attention fusion (weighted combination)
4. Model stacking (meta-learner)
5. Feature selection (best features)
6. Advanced ablation studies
"""

import sys
import time
from pathlib import Path

def main():
    print("Advanced Model Integration Pipeline")
    print("=" * 60)
    print("This will run comprehensive model integration analysis:")
    print("  - Late fusion (feature concatenation)")
    print("  - Early fusion (separate models)")
    print("  - Attention fusion (weighted combination)")
    print("  - Model stacking (meta-learner)")
    print("  - Feature selection (best features)")
    print("  - Advanced ablation studies")
    print()
    
    # Add analysis modules to path
    sys.path.append('analysis_modules')
    
    try:
        # Run advanced integration analysis
        print("Running Advanced Model Integration Analysis...")
        start_time = time.time()
        
        from advanced_model_integration import AdvancedModelIntegration
        
        analysis = AdvancedModelIntegration(random_state=42)
        results, ablation_results = analysis.run_advanced_integration_analysis()
        
        analysis_time = time.time() - start_time
        print(f"   Advanced integration analysis completed in {analysis_time:.1f} seconds")
        
        # Create comprehensive comparison
        print("\nCreating comprehensive comparison...")
        start_time = time.time()
        
        comparison_results = create_comprehensive_comparison(results, ablation_results)
        
        comparison_time = time.time() - start_time
        print(f"   Comparison analysis completed in {comparison_time:.1f} seconds")
        
        # Summary
        print("\n" + "=" * 60)
        print("ADVANCED MODEL INTEGRATION COMPLETE!")
        print("=" * 60)
        
        if comparison_results:
            print(f"Integration Approaches Tested:")
            approaches = comparison_results.get('approaches_tested', [])
            for approach in approaches:
                print(f"  - {approach}")
            
            print(f"\nBest Performing Approach:")
            best_approach = comparison_results.get('best_approach', 'Unknown')
            best_auc = comparison_results.get('best_auc', 0)
            print(f"  {best_approach}: AUC = {best_auc:.4f}")
            
            print(f"\nKey Findings:")
            findings = comparison_results.get('key_findings', [])
            for finding in findings:
                print(f"  - {finding}")
        
        print(f"\nKey Files Generated:")
        print(f"  - final_results/advanced_integration/ (integration results)")
        print(f"  - integration_results.json (detailed results)")
        print(f"  - ablation_results.json (ablation studies)")
        print(f"  - integration_summary.csv (comparison table)")
        
        print(f"\nAdvanced model integration pipeline completed successfully!")
        print(f"Check the generated files for detailed insights.")
        
    except ImportError as e:
        print(f"Error: Could not import required modules. {e}")
        print("Make sure you're running this from the dissertation_results directory.")
        return 1
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0

def create_comprehensive_comparison(results, ablation_results):
    """Create comprehensive comparison of integration approaches"""
    print("Creating comprehensive comparison...")
    
    comparison_data = []
    approaches_tested = []
    best_approach = None
    best_auc = 0
    
    for regime, regime_results in results.items():
        for approach, approach_results in regime_results.items():
            if approach_results is not None:
                auc = approach_results['mean_auc']
                comparison_data.append({
                    'Regime': regime,
                    'Approach': approach,
                    'AUC': auc,
                    'PR_AUC': approach_results['mean_pr_auc'],
                    'Std_AUC': approach_results['std_auc']
                })
                
                approaches_tested.append(approach)
                
                if auc > best_auc:
                    best_auc = auc
                    best_approach = approach
    
    # Calculate average performance by approach
    approach_performance = {}
    for data in comparison_data:
        approach = data['Approach']
        if approach not in approach_performance:
            approach_performance[approach] = []
        approach_performance[approach].append(data['AUC'])
    
    avg_performance = {approach: np.mean(aucs) for approach, aucs in approach_performance.items()}
    
    # Key findings
    key_findings = []
    
    # Find best approach
    best_avg_approach = max(avg_performance.items(), key=lambda x: x[1])
    key_findings.append(f"Best average approach: {best_avg_approach[0]} (AUC: {best_avg_approach[1]:.4f})")
    
    # Compare with late fusion baseline
    if 'Late_Fusion' in avg_performance:
        late_fusion_auc = avg_performance['Late_Fusion']
        for approach, auc in avg_performance.items():
            if approach != 'Late_Fusion':
                improvement = auc - late_fusion_auc
                if improvement > 0.01:
                    key_findings.append(f"{approach} improves over Late_Fusion by {improvement:.4f} AUC")
    
    # Ablation insights
    ablation_insights = []
    for regime, ablation in ablation_results.items():
        for ablation_name, ablation_result in ablation.items():
            if ablation_result is not None:
                ablation_auc = ablation_result['mean_auc']
                # Compare with best approach for this regime
                regime_best = max([r['AUC'] for r in comparison_data if r['Regime'] == regime])
                impact = regime_best - ablation_auc
                if impact > 0.01:
                    ablation_insights.append(f"Removing {ablation_name} reduces AUC by {impact:.4f} in {regime}")
    
    key_findings.extend(ablation_insights[:3])  # Top 3 ablation insights
    
    comparison_results = {
        'approaches_tested': list(set(approaches_tested)),
        'best_approach': best_approach,
        'best_auc': best_auc,
        'average_performance': avg_performance,
        'key_findings': key_findings,
        'comparison_data': comparison_data
    }
    
    # Save comparison results
    results_dir = Path('final_results/advanced_integration')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(results_dir / 'comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    # Create comparison table
    import pandas as pd
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(results_dir / 'comprehensive_comparison.csv', index=False)
    
    print(f"   Comparison results saved to {results_dir}")
    
    return comparison_results

if __name__ == "__main__":
    import numpy as np
    exit_code = main()
    sys.exit(exit_code) 