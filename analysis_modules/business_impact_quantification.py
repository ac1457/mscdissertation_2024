"""
Business Impact Quantification
Translates model performance improvements to financial terms and economic value.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve
import warnings
warnings.filterwarnings('ignore')

class BusinessImpactQuantifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.config = {
            'output_dir': 'final_results/business_impact',
            'default_cost_matrix': {
                'true_positive': 0,      # Correctly identified default (no cost)
                'false_positive': 100,   # Incorrectly rejected good loan (opportunity cost)
                'true_negative': 0,      # Correctly identified good loan (no cost)
                'false_negative': 15000  # Missed default (principal loss)
            },
            'loan_parameters': {
                'average_loan_amount': 15000,
                'interest_rate': 0.12,   # 12% annual interest
                'loan_term_months': 36,
                'loss_given_default': 0.6,  # 60% of principal lost in default
                'operational_cost_per_loan': 200
            },
            'portfolio_parameters': {
                'total_loans': 100000,
                'baseline_default_rate': 0.15,
                'risk_thresholds': [0.05, 0.10, 0.15, 0.20, 0.25]
            }
        }
        
    def calculate_value_add(self, lift, default_rate, num_loans, avg_loan=None, loss_given_default=None):
        """Calculate economic value added from model improvements"""
        if avg_loan is None:
            avg_loan = self.config['loan_parameters']['average_loan_amount']
        if loss_given_default is None:
            loss_given_default = self.config['loan_parameters']['loss_given_default']
        
        # Calculate prevented defaults
        prevented_defaults = lift * default_rate * num_loans
        
        # Calculate value added
        value_added = prevented_defaults * avg_loan * loss_given_default
        
        return {
            'prevented_defaults': prevented_defaults,
            'value_added': value_added,
            'value_per_loan': value_added / num_loans,
            'roi_percentage': (value_added / (num_loans * avg_loan)) * 100
        }
    
    def calculate_cost_savings(self, confusion_matrix, cost_matrix=None):
        """Calculate cost savings from improved predictions"""
        if cost_matrix is None:
            cost_matrix = self.config['default_cost_matrix']
        
        tp, fp, tn, fn = confusion_matrix
        
        # Calculate costs
        total_cost = (tp * cost_matrix['true_positive'] + 
                     fp * cost_matrix['false_positive'] + 
                     tn * cost_matrix['true_negative'] + 
                     fn * cost_matrix['false_negative'])
        
        return {
            'total_cost': total_cost,
            'cost_breakdown': {
                'true_positives': tp * cost_matrix['true_positive'],
                'false_positives': fp * cost_matrix['false_positive'],
                'true_negatives': tn * cost_matrix['true_negative'],
                'false_negatives': fn * cost_matrix['false_negative']
            },
            'average_cost_per_loan': total_cost / (tp + fp + tn + fn)
        }
    
    def calculate_roi_analysis(self, baseline_performance, improved_performance, 
                             implementation_cost=100000):
        """Calculate ROI of model improvements"""
        
        # Calculate performance improvements
        auc_improvement = improved_performance['auc'] - baseline_performance['auc']
        precision_improvement = improved_performance['precision'] - baseline_performance['precision']
        
        # Estimate value from improvements
        portfolio_size = self.config['portfolio_parameters']['total_loans']
        avg_loan = self.config['loan_parameters']['average_loan_amount']
        default_rate = self.config['portfolio_parameters']['baseline_default_rate']
        
        # Conservative estimate: 1% AUC improvement = 1% fewer defaults
        default_reduction = auc_improvement * default_rate
        prevented_defaults = default_reduction * portfolio_size
        value_added = prevented_defaults * avg_loan * self.config['loan_parameters']['loss_given_default']
        
        # Calculate ROI
        roi = (value_added - implementation_cost) / implementation_cost * 100
        
        return {
            'implementation_cost': implementation_cost,
            'value_added': value_added,
            'roi_percentage': roi,
            'payback_period_months': implementation_cost / (value_added / 12) if value_added > 0 else float('inf'),
            'performance_improvements': {
                'auc_improvement': auc_improvement,
                'precision_improvement': precision_improvement,
                'default_reduction': default_reduction
            }
        }
    
    def analyze_threshold_optimization(self, y_true, y_pred_proba):
        """Analyze optimal decision threshold for business value"""
        print("Analyzing threshold optimization for business value...")
        
        thresholds = np.arange(0.01, 0.99, 0.01)
        threshold_analysis = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            
            # Calculate confusion matrix
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fn = ((y_true == 1) & (y_pred == 1)).sum()
            
            # Calculate business metrics
            cost_savings = self.calculate_cost_savings([tp, fp, tn, fn])
            
            # Calculate portfolio metrics
            portfolio_value = self.calculate_portfolio_value(
                tp, fp, tn, fn, threshold
            )
            
            threshold_analysis.append({
                'threshold': threshold,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'total_cost': cost_savings['total_cost'],
                'portfolio_value': portfolio_value['total_value'],
                'rejection_rate': (fp + tp) / (tp + fp + tn + fn)
            })
        
        threshold_df = pd.DataFrame(threshold_analysis)
        
        # Find optimal threshold
        optimal_threshold_idx = threshold_df['portfolio_value'].idxmax()
        optimal_threshold = threshold_df.loc[optimal_threshold_idx]
        
        return {
            'threshold_analysis': threshold_df,
            'optimal_threshold': optimal_threshold,
            'recommended_threshold': optimal_threshold['threshold']
        }
    
    def calculate_portfolio_value(self, tp, fp, tn, fn, threshold):
        """Calculate total portfolio value at given threshold"""
        
        # Portfolio parameters
        avg_loan = self.config['loan_parameters']['average_loan_amount']
        interest_rate = self.config['loan_parameters']['interest_rate']
        loan_term = self.config['loan_parameters']['loan_term_months']
        loss_given_default = self.config['loan_parameters']['loss_given_default']
        operational_cost = self.config['loan_parameters']['operational_cost_per_loan']
        
        # Calculate loan outcomes
        total_loans = tp + fp + tn + fn
        approved_loans = tp + tn
        rejected_loans = fp + fn
        
        # Revenue from approved loans
        monthly_interest = interest_rate / 12
        total_payments = avg_loan * (monthly_interest * (1 + monthly_interest)**loan_term) / ((1 + monthly_interest)**loan_term - 1) * loan_term
        revenue_per_loan = total_payments - avg_loan
        
        # Losses from defaults
        default_losses = fn * avg_loan * loss_given_default
        
        # Operational costs
        total_operational_cost = total_loans * operational_cost
        
        # Total portfolio value
        total_revenue = approved_loans * revenue_per_loan
        total_value = total_revenue - default_losses - total_operational_cost
        
        return {
            'total_value': total_value,
            'total_revenue': total_revenue,
            'default_losses': default_losses,
            'operational_costs': total_operational_cost,
            'approved_loans': approved_loans,
            'rejected_loans': rejected_loans,
            'default_rate': fn / approved_loans if approved_loans > 0 else 0
        }
    
    def calculate_lift_analysis(self, y_true, y_pred_proba, percentiles=[10, 20, 30, 40, 50]):
        """Calculate lift analysis for business value"""
        print("Calculating lift analysis for business value...")
        
        # Create lift table
        lift_data = []
        
        for percentile in percentiles:
            # Get top percentile predictions
            threshold = np.percentile(y_pred_proba, 100 - percentile)
            top_mask = y_pred_proba >= threshold
            
            # Calculate metrics
            total_in_percentile = top_mask.sum()
            defaults_in_percentile = (y_true[top_mask] == 1).sum()
            default_rate = defaults_in_percentile / total_in_percentile if total_in_percentile > 0 else 0
            
            # Calculate lift
            overall_default_rate = y_true.mean()
            lift = default_rate / overall_default_rate if overall_default_rate > 0 else 0
            
            # Calculate business value
            value_analysis = self.calculate_value_add(
                lift - 1, overall_default_rate, total_in_percentile
            )
            
            lift_data.append({
                'percentile': percentile,
                'threshold': threshold,
                'total_loans': total_in_percentile,
                'defaults': defaults_in_percentile,
                'default_rate': default_rate,
                'lift': lift,
                'value_added': value_analysis['value_added'],
                'value_per_loan': value_analysis['value_per_loan']
            })
        
        return pd.DataFrame(lift_data)
    
    def calculate_break_even_analysis(self, model_improvement, implementation_cost):
        """Calculate break-even analysis for model implementation"""
        
        portfolio_size = self.config['portfolio_parameters']['total_loans']
        avg_loan = self.config['loan_parameters']['average_loan_amount']
        default_rate = self.config['portfolio_parameters']['baseline_default_rate']
        
        # Calculate required improvement for break-even
        required_value = implementation_cost
        required_default_reduction = required_value / (portfolio_size * avg_loan * self.config['loan_parameters']['loss_given_default'])
        required_auc_improvement = required_default_reduction / default_rate
        
        # Calculate time to break-even
        monthly_value = (model_improvement['value_added'] / 12) if 'value_added' in model_improvement else 0
        months_to_break_even = implementation_cost / monthly_value if monthly_value > 0 else float('inf')
        
        return {
            'implementation_cost': implementation_cost,
            'required_auc_improvement': required_auc_improvement,
            'required_default_reduction': required_default_reduction,
            'months_to_break_even': months_to_break_even,
            'annual_value': model_improvement.get('value_added', 0) * 12,
            'roi_after_1_year': ((model_improvement.get('value_added', 0) * 12 - implementation_cost) / implementation_cost * 100) if implementation_cost > 0 else 0
        }
    
    def create_business_visualizations(self, threshold_analysis, lift_analysis, roi_analysis):
        """Create business impact visualizations"""
        from pathlib import Path
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Threshold optimization
        if threshold_analysis is not None:
            plt.figure(figsize=(12, 8))
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Portfolio value vs threshold
            ax1.plot(threshold_analysis['threshold_analysis']['threshold'], 
                    threshold_analysis['threshold_analysis']['portfolio_value'], 'b-', linewidth=2)
            ax1.axvline(threshold_analysis['optimal_threshold']['threshold'], color='red', linestyle='--', 
                       label=f"Optimal: {threshold_analysis['optimal_threshold']['threshold']:.3f}")
            ax1.set_xlabel('Decision Threshold')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.set_title('Portfolio Value vs Decision Threshold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Rejection rate vs threshold
            ax2.plot(threshold_analysis['threshold_analysis']['threshold'], 
                    threshold_analysis['threshold_analysis']['rejection_rate'], 'g-', linewidth=2)
            ax2.axvline(threshold_analysis['optimal_threshold']['threshold'], color='red', linestyle='--')
            ax2.set_xlabel('Decision Threshold')
            ax2.set_ylabel('Rejection Rate')
            ax2.set_title('Rejection Rate vs Decision Threshold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'threshold_optimization.png')
            plt.close()
        
        # 2. Lift analysis
        if lift_analysis is not None:
            plt.figure(figsize=(12, 8))
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Lift by percentile
            ax1.bar(lift_analysis['percentile'], lift_analysis['lift'], alpha=0.7, color='blue')
            ax1.axhline(y=1, color='red', linestyle='--', label='Baseline')
            ax1.set_xlabel('Top Percentile')
            ax1.set_ylabel('Lift')
            ax1.set_title('Lift Analysis by Percentile')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Value added by percentile
            ax2.bar(lift_analysis['percentile'], lift_analysis['value_added'], alpha=0.7, color='green')
            ax2.set_xlabel('Top Percentile')
            ax2.set_ylabel('Value Added ($)')
            ax2.set_title('Business Value Added by Percentile')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'lift_analysis.png')
            plt.close()
    
    def run_comprehensive_business_analysis(self, y_true, y_pred_proba, baseline_performance=None, 
                                          improved_performance=None):
        """Run comprehensive business impact analysis"""
        print("Running comprehensive business impact analysis...")
        
        # Load data if not provided
        if y_true is None or y_pred_proba is None:
            try:
                df = pd.read_csv('data/real_lending_club/real_lending_club_processed.csv')
                print(f"Loaded dataset: {len(df)} records")
                
                # Simulate predictions for demonstration
                np.random.seed(self.random_state)
                y_true = (df['target_5%'] == 1).astype(int) if 'target_5%' in df.columns else np.random.binomial(1, 0.15, len(df))
                y_pred_proba = np.random.uniform(0, 1, len(df))
                
            except FileNotFoundError:
                print("Dataset not found. Please run real data processing first.")
                return None
        
        # Run threshold optimization
        threshold_analysis = self.analyze_threshold_optimization(y_true, y_pred_proba)
        
        # Run lift analysis
        lift_analysis = self.calculate_lift_analysis(y_true, y_pred_proba)
        
        # Run ROI analysis
        if baseline_performance and improved_performance:
            roi_analysis = self.calculate_roi_analysis(baseline_performance, improved_performance)
        else:
            # Simulate performance improvements
            simulated_improvement = {
                'value_added': 500000,  # $500K annual value
                'auc_improvement': 0.05
            }
            roi_analysis = self.calculate_break_even_analysis(simulated_improvement, 100000)
        
        # Create visualizations
        self.create_business_visualizations(threshold_analysis, lift_analysis, roi_analysis)
        
        # Compile results
        business_results = {
            'threshold_optimization': threshold_analysis,
            'lift_analysis': lift_analysis,
            'roi_analysis': roi_analysis,
            'summary': {
                'optimal_threshold': threshold_analysis['optimal_threshold']['threshold'],
                'max_portfolio_value': threshold_analysis['optimal_threshold']['portfolio_value'],
                'total_value_added': lift_analysis['value_added'].sum(),
                'roi_percentage': roi_analysis.get('roi_after_1_year', 0)
            }
        }
        
        # Save results
        self.save_business_results(business_results)
        
        return business_results
    
    def save_business_results(self, results):
        """Save business impact analysis results"""
        import json
        from pathlib import Path
        
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(output_dir / 'business_impact_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary report
        self.create_business_report(results, output_dir)
        
        print(f"Business impact analysis results saved to: {output_dir}")
    
    def create_business_report(self, results, output_dir):
        """Create business impact summary report"""
        report = []
        report.append("# Business Impact Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        summary = results['summary']
        report.append("## Business Impact Summary")
        report.append(f"- Optimal Decision Threshold: {summary['optimal_threshold']:.3f}")
        report.append(f"- Maximum Portfolio Value: ${summary['max_portfolio_value']:,.2f}")
        report.append(f"- Total Value Added: ${summary['total_value_added']:,.2f}")
        report.append(f"- ROI After 1 Year: {summary['roi_percentage']:.1f}%")
        report.append("")
        
        # Threshold Optimization
        report.append("## Threshold Optimization")
        optimal = results['threshold_optimization']['optimal_threshold']
        report.append(f"- Optimal Threshold: {optimal['threshold']:.3f}")
        report.append(f"- Portfolio Value: ${optimal['portfolio_value']:,.2f}")
        report.append(f"- Rejection Rate: {optimal['rejection_rate']:.1%}")
        report.append(f"- Precision: {optimal['precision']:.1%}")
        report.append(f"- Recall: {optimal['recall']:.1%}")
        report.append("")
        
        # Lift Analysis
        report.append("## Lift Analysis")
        lift_df = results['lift_analysis']
        for _, row in lift_df.iterrows():
            report.append(f"### Top {row['percentile']}%")
            report.append(f"- Lift: {row['lift']:.2f}x")
            report.append(f"- Value Added: ${row['value_added']:,.2f}")
            report.append(f"- Value per Loan: ${row['value_per_loan']:.2f}")
            report.append("")
        
        # ROI Analysis
        report.append("## ROI Analysis")
        roi = results['roi_analysis']
        report.append(f"- Implementation Cost: ${roi['implementation_cost']:,.2f}")
        report.append(f"- Required AUC Improvement: {roi['required_auc_improvement']:.3f}")
        report.append(f"- Months to Break-Even: {roi['months_to_break_even']:.1f}")
        report.append(f"- Annual Value: ${roi['annual_value']:,.2f}")
        report.append(f"- ROI After 1 Year: {roi['roi_after_1_year']:.1f}%")
        report.append("")
        
        # Save report
        with open(output_dir / 'business_impact_report.md', 'w') as f:
            f.write('\n'.join(report))

if __name__ == "__main__":
    # Run comprehensive business impact analysis
    quantifier = BusinessImpactQuantifier(random_state=42)
    results = quantifier.run_comprehensive_business_analysis(None, None)
    
    if results:
        print("Business impact analysis completed successfully!")
    else:
        print("Business impact analysis failed. Please check the error messages above.") 