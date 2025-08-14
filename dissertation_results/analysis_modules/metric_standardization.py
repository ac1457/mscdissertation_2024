#!/usr/bin/env python3
"""
Metric Standardization - Lending Club Sentiment Analysis
======================================================
Standardizes metric formatting and provides consistent legend injection
across all reports. Ensures uniform presentation of results.
"""

import pandas as pd
import numpy as np

class MetricStandardization:
    """
    Standardize metric formatting and provide consistent legends
    """
    
    def __init__(self):
        self.metric_definitions = self.get_metric_definitions()
        self.formatting_standards = self.get_formatting_standards()
    
    def get_metric_definitions(self):
        """
        Get standardized metric definitions
        """
        return {
            'AUC': {
                'definition': 'Area Under ROC Curve - probability that a randomly selected positive instance is ranked higher than a randomly selected negative instance',
                'range': '0.0 to 1.0 (0.5 = random, 1.0 = perfect)',
                'interpretation': 'Higher is better',
                'precision': 4
            },
            'AUC_Improvement': {
                'definition': 'AUC_Variant - AUC_Traditional',
                'sign_convention': 'POSITIVE values indicate improvement (higher AUC is better)',
                'precision': 4,
                'example': 'AUC_Improvement = 0.0234 means variant has higher AUC'
            },
            'Improvement_Percent': {
                'definition': '(AUC_Improvement / AUC_Traditional) * 100',
                'sign_convention': 'POSITIVE values indicate improvement',
                'precision': 2,
                'example': 'Improvement_Percent = 3.45% means 3.45% relative improvement'
            },
            'Brier_Score': {
                'definition': 'Mean squared error between predicted probabilities and actual outcomes',
                'range': '0.0 to 1.0 (0.0 = perfect calibration)',
                'interpretation': 'Lower is better',
                'precision': 4
            },
            'Brier_Improvement': {
                'definition': 'Brier_Traditional - Brier_Variant',
                'sign_convention': 'NEGATIVE values indicate improvement (lower Brier is better)',
                'precision': 4,
                'example': 'Brier_Improvement = -0.0023 means variant has better calibration'
            },
            'PR_AUC': {
                'definition': 'Area Under Precision-Recall Curve - measures precision-recall trade-off',
                'range': '0.0 to 1.0',
                'interpretation': 'Higher is better, especially important for imbalanced datasets',
                'precision': 4
            },
            'Lift_10': {
                'definition': 'Ratio of default rate in top 10% of predictions vs overall default rate',
                'range': '0.0 to ∞ (1.0 = no lift, >1.0 = positive lift)',
                'interpretation': 'Higher is better',
                'precision': 2
            },
            'DeLong_p_value': {
                'definition': 'p-value from DeLong test comparing two ROC AUCs',
                'range': '0.0 to 1.0',
                'interpretation': '<0.05 = statistically significant difference',
                'precision': 'scientific_notation'
            }
        }
    
    def get_formatting_standards(self):
        """
        Get formatting standards for different metric types
        """
        return {
            'AUC': {
                'format': '{:.4f}',
                'example': '0.6234'
            },
            'AUC_Improvement': {
                'format': '{:+.4f}',
                'example': '+0.0234'
            },
            'Improvement_Percent': {
                'format': '{:+.2f}%',
                'example': '+3.45%'
            },
            'Brier_Score': {
                'format': '{:.4f}',
                'example': '0.2345'
            },
            'Brier_Improvement': {
                'format': '{:+.4f}',
                'example': '-0.0023'
            },
            'PR_AUC': {
                'format': '{:.4f}',
                'example': '0.5678'
            },
            'Lift_10': {
                'format': '{:.2f}',
                'example': '1.23'
            },
            'DeLong_p_value': {
                'format': 'scientific_notation',
                'example': '1.23e-05'
            },
            'confidence_interval': {
                'format': '[{:.4f}, {:.4f}]',
                'example': '[0.6123, 0.6345]'
            }
        }
    
    def format_metric(self, value, metric_type):
        """
        Format a metric value according to standards
        """
        if pd.isna(value):
            return 'N/A'
        
        if metric_type == 'DeLong_p_value':
            if value < 1e-15:
                return '<1e-15'
            elif value < 0.001:
                return f'{value:.2e}'
            else:
                return f'{value:.6f}'
        
        if metric_type in self.formatting_standards:
            format_str = self.formatting_standards[metric_type]['format']
            if format_str == 'scientific_notation':
                if value < 0.001:
                    return f'{value:.2e}'
                else:
                    return f'{value:.6f}'
            else:
                return format_str.format(value)
        
        # Default formatting
        return f'{value:.4f}'
    
    def generate_legend(self, metrics_used=None):
        """
        Generate standardized legend for reports
        """
        if metrics_used is None:
            metrics_used = list(self.metric_definitions.keys())
        
        legend = []
        legend.append("METRIC DEFINITIONS")
        legend.append("-" * 20)
        
        for metric in metrics_used:
            if metric in self.metric_definitions:
                definition = self.metric_definitions[metric]
                legend.append(f"{metric}:")
                legend.append(f"  Definition: {definition['definition']}")
                
                if 'range' in definition:
                    legend.append(f"  Range: {definition['range']}")
                
                if 'interpretation' in definition:
                    legend.append(f"  Interpretation: {definition['interpretation']}")
                
                if 'sign_convention' in definition:
                    legend.append(f"  Sign Convention: {definition['sign_convention']}")
                
                if 'example' in definition:
                    legend.append(f"  Example: {definition['example']}")
                
                legend.append("")
        
        return "\n".join(legend)
    
    def standardize_dataframe(self, df):
        """
        Standardize metric formatting in a dataframe
        """
        df_standardized = df.copy()
        
        # Standardize AUC columns
        auc_columns = [col for col in df.columns if 'AUC' in col and 'Improvement' not in col and 'PR' not in col]
        for col in auc_columns:
            if col in df.columns:
                df_standardized[col] = df[col].apply(lambda x: self.format_metric(x, 'AUC'))
        
        # Standardize AUC Improvement columns
        auc_improvement_columns = [col for col in df.columns if 'AUC_Improvement' in col and 'Percent' not in col]
        for col in auc_improvement_columns:
            if col in df.columns:
                df_standardized[col] = df[col].apply(lambda x: self.format_metric(x, 'AUC_Improvement'))
        
        # Standardize Improvement Percent columns
        percent_columns = [col for col in df.columns if 'Percent' in col or 'percent' in col]
        for col in percent_columns:
            if col in df.columns:
                df_standardized[col] = df[col].apply(lambda x: self.format_metric(x, 'Improvement_Percent'))
        
        # Standardize Brier columns
        brier_columns = [col for col in df.columns if 'Brier' in col]
        for col in brier_columns:
            if col in df.columns:
                if 'Improvement' in col:
                    df_standardized[col] = df[col].apply(lambda x: self.format_metric(x, 'Brier_Improvement'))
                else:
                    df_standardized[col] = df[col].apply(lambda x: self.format_metric(x, 'Brier_Score'))
        
        # Standardize PR-AUC columns
        pr_auc_columns = [col for col in df.columns if 'PR_AUC' in col]
        for col in pr_auc_columns:
            if col in df.columns:
                df_standardized[col] = df[col].apply(lambda x: self.format_metric(x, 'PR_AUC'))
        
        # Standardize Lift columns
        lift_columns = [col for col in df.columns if 'Lift' in col]
        for col in lift_columns:
            if col in df.columns:
                df_standardized[col] = df[col].apply(lambda x: self.format_metric(x, 'Lift_10'))
        
        # Standardize DeLong p-value columns
        delong_columns = [col for col in df.columns if 'DeLong' in col and 'p' in col]
        for col in delong_columns:
            if col in df.columns:
                df_standardized[col] = df[col].apply(lambda x: self.format_metric(x, 'DeLong_p_value'))
        
        return df_standardized
    
    def add_negative_delta_flags(self, df):
        """
        Add explicit flags for negative deltas
        """
        df_flagged = df.copy()
        
        # Flag negative AUC improvements
        if 'AUC_Improvement' in df.columns:
            df_flagged['AUC_Improvement_Flag'] = df['AUC_Improvement'].apply(
                lambda x: 'NEGATIVE' if x < 0 else 'POSITIVE'
            )
        
        # Flag negative Brier improvements (which are actually good)
        if 'Brier_Improvement' in df.columns:
            df_flagged['Brier_Improvement_Flag'] = df['Brier_Improvement'].apply(
                lambda x: 'IMPROVEMENT' if x < 0 else 'DEGRADATION'
            )
        
        return df_flagged
    
    def generate_standardized_report_header(self):
        """
        Generate standardized report header with metric definitions
        """
        header = []
        header.append("LENDING CLUB SENTIMENT ANALYSIS - STANDARDIZED REPORT")
        header.append("=" * 60)
        header.append("")
        header.append("This report follows standardized metric definitions and formatting.")
        header.append("All improvements are calculated relative to Traditional baseline.")
        header.append("")
        
        # Add key metric definitions
        header.append("KEY METRIC DEFINITIONS:")
        header.append("-" * 25)
        header.append("• AUC_Improvement = AUC_Variant - AUC_Traditional (positive = better)")
        header.append("• Brier_Improvement = Brier_Traditional - Brier_Variant (negative = better)")
        header.append("• Improvement_Percent = (AUC_Improvement / AUC_Traditional) * 100")
        header.append("• DeLong p-value < 0.05 indicates statistical significance")
        header.append("• Confidence intervals are 95% bootstrap intervals")
        header.append("")
        
        return "\n".join(header)
    
    def inject_legend_into_report(self, report_content, metrics_used=None):
        """
        Inject standardized legend into existing report
        """
        # Find a good place to insert the legend (before conclusions)
        lines = report_content.split('\n')
        
        # Look for conclusion section
        conclusion_index = -1
        for i, line in enumerate(lines):
            if 'CONCLUSION' in line.upper() or 'CONCLUSIONS' in line.upper():
                conclusion_index = i
                break
        
        # Generate legend
        legend = self.generate_legend(metrics_used)
        
        # Insert legend before conclusions
        if conclusion_index > 0:
            lines.insert(conclusion_index, "")
            lines.insert(conclusion_index, legend)
        else:
            # If no conclusion section, add at the end
            lines.append("")
            lines.append(legend)
        
        return '\n'.join(lines)

if __name__ == "__main__":
    standardizer = MetricStandardization()
    
    # Example usage
    print("Metric Standardization Module")
    print("=" * 40)
    
    # Show metric definitions
    print("\nAvailable Metrics:")
    for metric, definition in standardizer.metric_definitions.items():
        print(f"• {metric}")
    
    # Show formatting examples
    print("\nFormatting Examples:")
    print(f"AUC: {standardizer.format_metric(0.6234, 'AUC')}")
    print(f"AUC Improvement: {standardizer.format_metric(0.0234, 'AUC_Improvement')}")
    print(f"Improvement Percent: {standardizer.format_metric(3.45, 'Improvement_Percent')}")
    print(f"Brier Improvement: {standardizer.format_metric(-0.0023, 'Brier_Improvement')}")
    print(f"DeLong p-value: {standardizer.format_metric(1.23e-05, 'DeLong_p_value')}")
    
    print("\n✅ Metric standardization module ready!") 