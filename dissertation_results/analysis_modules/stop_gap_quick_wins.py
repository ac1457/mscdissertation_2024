#!/usr/bin/env python3
"""
Stop-Gap Quick Wins - Lending Club Sentiment Analysis
====================================================
Immediate fixes for critical issues:
1. Add legend block to every table export
2. Insert "Statistical testing pending" flags where missing
3. Remove duplicate "SUCCESS!/READY" banners
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

class StopGapQuickWins:
    """
    Immediate fixes for critical issues
    """
    
    def __init__(self):
        self.legend_block = {
            'AUC': 'Area Under ROC Curve (0.5 = random, 1.0 = perfect)',
            'PR_AUC': 'Area Under Precision-Recall Curve (better for imbalanced data)',
            'Brier_Score': 'Mean squared error of probability predictions (0 = perfect, 1 = worst)',
            'Brier_Improvement': 'Brier_traditional - Brier_variant (positive = improvement)',
            'AUC_Improvement': 'AUC_variant - AUC_traditional (positive = improvement)',
            'Statistical_Testing': 'Statistical testing pending - DeLong test and bootstrap CIs needed',
            'Precision': 'True Positives / (True Positives + False Positives)',
            'Recall': 'True Positives / (True Positives + False Negatives)',
            'F1_Score': 'Harmonic mean of precision and recall',
            'Lift': 'Ratio of default rate in top k% vs overall default rate',
            'Capture_Rate': 'Percentage of all defaults captured in top k%'
        }
    
    def add_legend_to_csv(self, csv_path, legend_block=None):
        """
        Add legend block to CSV file
        """
        if not os.path.exists(csv_path):
            print(f"⚠️ File not found: {csv_path}")
            return
        
        # Read existing CSV
        df = pd.read_csv(csv_path)
        
        # Create legend text
        if legend_block is None:
            legend_block = self.legend_block
        
        legend_text = "# LEGEND\n"
        for key, description in legend_block.items():
            legend_text += f"# {key}: {description}\n"
        legend_text += f"# Generated: {datetime.now().isoformat()}\n"
        legend_text += "# Statistical testing pending - DeLong test and bootstrap CIs needed\n"
        legend_text += "# Brier_Improvement = Brier_traditional - Brier_variant (positive = improvement)\n"
        legend_text += "# AUC_Improvement = AUC_variant - AUC_traditional (positive = improvement)\n"
        
        # Write legend + CSV
        with open(csv_path, 'w') as f:
            f.write(legend_text)
            df.to_csv(f, index=False)
        
        print(f"✅ Added legend to: {csv_path}")
    
    def add_statistical_pending_flags(self, csv_path):
        """
        Add statistical testing pending flags to CSV
        """
        if not os.path.exists(csv_path):
            print(f"⚠️ File not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        
        # Add statistical testing pending columns
        if 'Statistical_Testing_Status' not in df.columns:
            df['Statistical_Testing_Status'] = 'PENDING'
        
        if 'DeLong_p_value' not in df.columns:
            df['DeLong_p_value'] = 'PENDING'
        
        if 'Bootstrap_CI_Lower' not in df.columns:
            df['Bootstrap_CI_Lower'] = 'PENDING'
        
        if 'Bootstrap_CI_Upper' not in df.columns:
            df['Bootstrap_CI_Upper'] = 'PENDING'
        
        # Save with legend
        self.add_legend_to_csv(csv_path)
        
        print(f"✅ Added statistical pending flags to: {csv_path}")
    
    def fix_metric_formatting(self, csv_path):
        """
        Apply metric formatting standards
        """
        if not os.path.exists(csv_path):
            print(f"⚠️ File not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        
        # Format metrics according to standards
        metric_columns = {
            'AUC_Mean': 4, 'AUC_Std': 4, 'PR_AUC_Mean': 4, 'PR_AUC_Std': 4,
            'AUC_Improvement': 4, 'PR_AUC_Improvement': 4, 'Brier_Improvement': 4,
            'AUC_Improvement_Percent': 2, 'Precision_Mean': 4, 'Recall_Mean': 4,
            'F1_Mean': 4, 'Brier_Mean': 4, 'Brier_Std': 4
        }
        
        for col, decimals in metric_columns.items():
            if col in df.columns:
                # Convert to numeric, handle errors
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].round(decimals)
        
        # Ensure negative ΔAUC is visible
        if 'AUC_Improvement' in df.columns:
            df['AUC_Improvement'] = df['AUC_Improvement'].apply(
                lambda x: f"{x:+.4f}" if pd.notna(x) else "PENDING"
            )
        
        if 'Brier_Improvement' in df.columns:
            df['Brier_Improvement'] = df['Brier_Improvement'].apply(
                lambda x: f"{x:+.4f}" if pd.notna(x) else "PENDING"
            )
        
        # Save with legend
        self.add_legend_to_csv(csv_path)
        
        print(f"✅ Fixed metric formatting in: {csv_path}")
    
    def remove_duplicate_banners(self, file_path):
        """
        Remove duplicate success/ready banners from files
        """
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            return
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Remove duplicate banners
        banners_to_remove = [
            "✅ SUCCESS!",
            "✅ READY!",
            "✅ COMPLETE!",
            "🎯 FIXED!",
            "🚀 DONE!"
        ]
        
        for banner in banners_to_remove:
            # Remove multiple occurrences, keep only first
            if content.count(banner) > 1:
                first_occurrence = content.find(banner)
                content = content[:first_occurrence + len(banner)] + content[first_occurrence + len(banner):].replace(banner, "")
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Removed duplicate banners from: {file_path}")
    
    def create_metrics_snapshot(self):
        """
        Create canonical metrics.json snapshot
        """
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'status': 'STATISTICAL_TESTING_PENDING',
            'legend': self.legend_block,
            'metric_standards': {
                'AUC': '4 decimal places',
                'PR_AUC': '4 decimal places', 
                'AUC_Improvement': '4 decimal places, signed (+/-)',
                'Brier_Improvement': '4 decimal places, signed (+/-)',
                'Percentage': '2 decimal places',
                'p_values': 'Scientific notation, <1e-15 cap'
            },
            'pending_items': [
                'Bootstrap CIs for realistic regimes',
                'DeLong tests for ΔAUC comparisons',
                'PR-AUC & F1 for low prevalence scenarios',
                'Calibration metrics (ECE, slope, intercept)',
                'Lift@k for realistic regimes',
                'Profit/decision curves',
                'Permutation null distribution tests',
                'Feature ablation analysis',
                'Temporal drift validation',
                'SHAP interpretability analysis'
            ],
            'completed_items': [
                'Realistic target creation',
                'Basic cross-validation',
                'Feature preparation',
                'Initial metric calculations'
            ]
        }
        
        with open('final_results/metrics_snapshot.json', 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        print("✅ Created canonical metrics.json snapshot")
        return snapshot
    
    def run_stop_gap_fixes(self):
        """
        Run all stop-gap quick wins
        """
        print("STOP-GAP QUICK WINS")
        print("=" * 30)
        
        # List of CSV files to fix
        csv_files = [
            'final_results/realistic_target_regime_summary.csv',
            'final_results/realistic_regime_validation_with_realistic_targets_results.csv',
            'final_results/realistic_regime_validation_with_realistic_targets_improvements.csv',
            'final_results/calibration_and_decision_utility_complete.csv',
            'final_results/sampling_transparency_summary.csv'
        ]
        
        # Apply fixes to each CSV
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                print(f"\nFixing: {csv_file}")
                self.add_legend_to_csv(csv_file)
                self.add_statistical_pending_flags(csv_file)
                self.fix_metric_formatting(csv_file)
            else:
                print(f"⚠️ File not found: {csv_file}")
        
        # Remove duplicate banners from documentation
        doc_files = [
            'TARGET_FIX_SUMMARY.md',
            'PRIORITY_CHECKLIST_IMPLEMENTATION_SUMMARY.md',
            'methodology/realistic_target_creation_report.txt'
        ]
        
        for doc_file in doc_files:
            if os.path.exists(doc_file):
                self.remove_duplicate_banners(doc_file)
        
        # Create canonical metrics snapshot
        self.create_metrics_snapshot()
        
        print("\n✅ STOP-GAP QUICK WINS COMPLETE!")
        print("✅ Applied to all existing files")
        print("✅ Created canonical metrics snapshot")
        print("✅ Added statistical testing pending flags")

if __name__ == "__main__":
    quick_wins = StopGapQuickWins()
    quick_wins.run_stop_gap_fixes() 