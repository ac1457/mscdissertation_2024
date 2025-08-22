"""
Enhanced Results Analyzer
Processes and analyzes results from the enhanced comprehensive analysis,
providing detailed insights into feature importance, ablation studies,
and model performance comparisons.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EnhancedResultsAnalyzer:
    def __init__(self):
        self.results_dir = Path('final_results/enhanced_comprehensive')
        self.output_dir = Path('final_results/enhanced_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def load_results(self):
        """Load enhanced analysis results"""
        try:
            with open(self.results_dir / 'enhanced_results.json', 'r') as f:
                all_results = json.load(f)
            
            with open(self.results_dir / 'ablation_results.json', 'r') as f:
                ablation_results = json.load(f)
            
            summary_df = pd.read_csv(self.results_dir / 'enhanced_summary.csv')
            
            return all_results, ablation_results, summary_df
        except FileNotFoundError:
            print("Enhanced results not found. Please run enhanced_comprehensive_analysis.py first.")
            return None, None, None

    def analyze_feature_importance(self, ablation_results):
        """Analyze individual feature importance from ablation study"""
        print("\nAnalyzing feature importance...")
        
        feature_importance = {}
        
        for regime, results in ablation_results.items():
            individual_results = results.get('individual_features', {})
            
            for feature, metrics in individual_results.items():
                auc = metrics['mean_auc']
                
                if feature not in feature_importance:
                    feature_importance[feature] = []
                
                feature_importance[feature].append({
                    'regime': regime,
                    'auc': auc
                })
        
        # Calculate average importance across regimes
        avg_importance = {}
        for feature, results in feature_importance.items():
            avg_auc = np.mean([r['auc'] for r in results])
            avg_importance[feature] = avg_auc
        
        # Sort by importance
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        features, aucs = zip(*sorted_features[:15])  # Top 15 features
        
        plt.barh(range(len(features)), aucs)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Average AUC')
        plt.title('Top 15 Individual Feature Performance')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save feature importance data
        importance_df = pd.DataFrame(sorted_features, columns=['Feature', 'Average_AUC'])
        importance_df.to_csv(self.output_dir / 'feature_importance.csv', index=False)
        
        print(f"   Top 5 features by AUC:")
        for feature, auc in sorted_features[:5]:
            print(f"      {feature}: {auc:.4f}")
        
        return importance_df

    def analyze_feature_groups(self, ablation_results):
        """Analyze feature group performance"""
        print("\nAnalyzing feature group performance...")
        
        group_performance = {}
        
        for regime, results in ablation_results.items():
            group_results = results.get('feature_groups', {})
            
            for group, metrics in group_results.items():
                auc = metrics['mean_auc']
                
                if group not in group_performance:
                    group_performance[group] = []
                
                group_performance[group].append({
                    'regime': regime,
                    'auc': auc
                })
        
        # Calculate average performance
        avg_group_performance = {}
        for group, results in group_performance.items():
            avg_auc = np.mean([r['auc'] for r in results])
            avg_group_performance[group] = avg_auc
        
        # Sort by performance
        sorted_groups = sorted(avg_group_performance.items(), key=lambda x: x[1], reverse=True)
        
        # Create group performance plot
        plt.figure(figsize=(12, 8))
        groups, aucs = zip(*sorted_groups)
        
        plt.barh(range(len(groups)), aucs)
        plt.yticks(range(len(groups)), groups)
        plt.xlabel('Average AUC')
        plt.title('Feature Group Performance Comparison')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_group_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save group performance data
        group_df = pd.DataFrame(sorted_groups, columns=['Feature_Group', 'Average_AUC'])
        group_df.to_csv(self.output_dir / 'feature_group_performance.csv', index=False)
        
        print(f"   Feature group ranking:")
        for group, auc in sorted_groups:
            print(f"      {group}: {auc:.4f}")
        
        return group_df

    def analyze_regime_performance(self, all_results):
        """Analyze performance across different regimes"""
        print("\nAnalyzing regime performance...")
        
        regime_data = []
        
        for regime, results in all_results.items():
            regime_results = results['regime_results']
            
            for model, metrics in regime_results.items():
                regime_data.append({
                    'Regime': regime,
                    'Model': model,
                    'AUC': metrics['mean_auc'],
                    'PR_AUC': metrics['mean_pr_auc'],
                    'AUC_Std': metrics['std_auc'],
                    'PR_AUC_Std': metrics['std_pr_auc']
                })
        
        regime_df = pd.DataFrame(regime_data)
        
        # Create regime comparison plot
        plt.figure(figsize=(15, 8))
        
        # Pivot for plotting
        pivot_df = regime_df.pivot(index='Model', columns='Regime', values='AUC')
        
        pivot_df.plot(kind='bar', figsize=(15, 8))
        plt.title('Model Performance Across Regimes')
        plt.xlabel('Model')
        plt.ylabel('AUC')
        plt.legend(title='Regime')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'regime_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save regime data
        regime_df.to_csv(self.output_dir / 'regime_performance.csv', index=False)
        
        return regime_df

    def analyze_improvements(self, summary_df):
        """Analyze improvements over baseline"""
        print("\nAnalyzing improvements over baseline...")
        
        # Calculate improvement percentages
        summary_df['Improvement_Percent'] = (
            (summary_df['Best_AUC'].astype(float) - summary_df['Baseline_AUC'].astype(float)) / 
            summary_df['Baseline_AUC'].astype(float) * 100
        )
        
        # Create improvement plot
        plt.figure(figsize=(10, 6))
        
        regimes = summary_df['Regime']
        improvements = summary_df['Improvement_Percent']
        colors = ['green' if x >= 0 else 'red' for x in improvements]
        
        plt.bar(regimes, improvements, color=colors)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Percentage Improvement Over Baseline')
        plt.xlabel('Regime')
        plt.ylabel('Improvement (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'improvements.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save improvement data
        summary_df.to_csv(self.output_dir / 'improvements_analysis.csv', index=False)
        
        print(f"   Improvement percentages:")
        for _, row in summary_df.iterrows():
            print(f"      {row['Regime']}: {row['Improvement_Percent']:.2f}%")
        
        return summary_df

    def convert_to_serializable(self, obj):
        """Convert numpy arrays and other objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'tolist'):  # Handle pandas Series
            return obj.tolist()
        else:
            return str(obj)  # Convert any other objects to string

    def create_comprehensive_report(self, all_results, ablation_results, summary_df):
        """Create comprehensive analysis report"""
        print("\nCreating comprehensive report...")
        
        # Analyze feature importance
        importance_df = self.analyze_feature_importance(ablation_results)
        
        # Analyze feature groups
        group_df = self.analyze_feature_groups(ablation_results)
        
        # Analyze regime performance
        regime_df = self.analyze_regime_performance(all_results)
        
        # Analyze improvements
        improved_summary = self.analyze_improvements(summary_df)
        
        # Create comprehensive report
        report = {
            'analysis_summary': {
                'total_regimes': len(all_results),
                'total_models': len(set([model for regime in all_results.values() 
                                       for model in regime['regime_results'].keys()])),
                'best_overall_model': improved_summary.loc[improved_summary['Improvement_Percent'].idxmax(), 'Best_Model'],
                'best_overall_improvement': float(improved_summary['Improvement_Percent'].max()),
                'regimes_with_improvement': int((improved_summary['Improvement_Percent'] > 0).sum())
            },
            'feature_insights': {
                'top_individual_features': importance_df.head(10).to_dict('records'),
                'top_feature_groups': group_df.head(5).to_dict('records'),
                'most_important_feature': importance_df.iloc[0]['Feature'],
                'best_feature_group': group_df.iloc[0]['Feature_Group']
            },
            'model_insights': {
                'best_model_by_regime': improved_summary[['Regime', 'Best_Model', 'Best_AUC']].to_dict('records'),
                'improvement_summary': improved_summary[['Regime', 'Improvement_Percent', 'Meets_Practical_Threshold']].to_dict('records')
            }
        }
        
        # Save comprehensive report
        with open(self.output_dir / 'comprehensive_report.json', 'w') as f:
            json.dump(self.convert_to_serializable(report), f, indent=2)
        
        # Create markdown report
        self.create_markdown_report(report, importance_df, group_df, improved_summary)
        
        return report

    def create_markdown_report(self, report, importance_df, group_df, improved_summary):
        """Create markdown report"""
        markdown_content = f"""
# Enhanced Comprehensive Analysis Report

## Executive Summary

This enhanced analysis implements advanced text preprocessing, entity extraction, fine-grained sentiment analysis, and comprehensive ablation studies to provide deeper insights into text-based credit risk modeling.

### Key Findings

**Overall Performance:**
- **Total regimes analyzed:** {report['analysis_summary']['total_regimes']}
- **Total models tested:** {report['analysis_summary']['total_models']}
- **Best overall model:** {report['analysis_summary']['best_overall_model']}
- **Best improvement:** {report['analysis_summary']['best_overall_improvement']:.2f}%
- **Regimes with improvement:** {report['analysis_summary']['regimes_with_improvement']}/{report['analysis_summary']['total_regimes']}

## Feature Analysis

### Top Individual Features

The following features showed the strongest individual predictive power:

"""
        
        for i, row in importance_df.head(10).iterrows():
            markdown_content += f"{i+1}. **{row['Feature']}**: AUC = {row['Average_AUC']:.4f}\n"
        
        markdown_content += f"""

### Top Feature Groups

The following feature groups performed best:

"""
        
        for i, row in group_df.head(5).iterrows():
            markdown_content += f"{i+1}. **{row['Feature_Group']}**: AUC = {row['Average_AUC']:.4f}\n"
        
        markdown_content += f"""

## Model Performance by Regime

| Regime | Best Model | Best AUC | Improvement | Meets Threshold |
|--------|------------|----------|-------------|-----------------|
"""
        
        for _, row in improved_summary.iterrows():
            markdown_content += f"| {row['Regime']} | {row['Best_Model']} | {row['Best_AUC']} | {row['Improvement_Percent']:.2f}% | {row['Meets_Practical_Threshold']} |\n"
        
        markdown_content += f"""

## Key Insights

### 1. Feature Importance
- **Most important individual feature:** {report['feature_insights']['most_important_feature']}
- **Best feature group:** {report['feature_insights']['best_feature_group']}

### 2. Model Performance
- **Best overall model:** {report['analysis_summary']['best_overall_model']}
- **Best improvement:** {report['analysis_summary']['best_overall_improvement']:.2f}%

### 3. Practical Significance
- **Regimes meeting practical threshold:** {report['analysis_summary']['regimes_with_improvement']}/{report['analysis_summary']['total_regimes']}

## Recommendations

### For Academic Contribution
1. **Emphasize methodological innovation** in text preprocessing and feature extraction
2. **Highlight comprehensive ablation studies** showing feature importance
3. **Document advanced entity extraction** for financial indicators

### For Future Research
1. **Focus on top-performing features** identified in ablation studies
2. **Explore combinations** of best individual features
3. **Investigate domain-specific** text preprocessing techniques

## Files Generated

- `feature_importance.png` - Individual feature performance visualization
- `feature_group_performance.png` - Feature group comparison
- `regime_performance.png` - Model performance across regimes
- `improvements.png` - Improvement over baseline visualization
- `comprehensive_report.json` - Detailed analysis results
- `feature_importance.csv` - Individual feature performance data
- `feature_group_performance.csv` - Feature group performance data
- `regime_performance.csv` - Regime performance data
- `improvements_analysis.csv` - Improvement analysis data

---

**Analysis completed:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(self.output_dir / 'comprehensive_report.md', 'w') as f:
            f.write(markdown_content)

    def run_complete_analysis(self):
        """Run complete enhanced results analysis"""
        print("Enhanced Results Analyzer")
        print("=" * 50)
        
        # Load results
        all_results, ablation_results, summary_df = self.load_results()
        
        if all_results is None:
            return
        
        # Create comprehensive report
        report = self.create_comprehensive_report(all_results, ablation_results, summary_df)
        
        print(f"\nEnhanced analysis complete!")
        print(f"Results saved to: {self.output_dir}")
        print(f"Key files generated:")
        print(f"  - comprehensive_report.md")
        print(f"  - feature_importance.png")
        print(f"  - feature_group_performance.png")
        print(f"  - regime_performance.png")
        print(f"  - improvements.png")
        
        return report

if __name__ == "__main__":
    analyzer = EnhancedResultsAnalyzer()
    report = analyzer.run_complete_analysis() 