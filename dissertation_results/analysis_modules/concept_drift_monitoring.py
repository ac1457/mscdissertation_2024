"""
Concept Drift Monitoring System
Monitors text feature stability and model performance over time to detect concept drift.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ConceptDriftMonitor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.config = {
            'output_dir': 'final_results/concept_drift_monitoring',
            'drift_threshold': 0.05,
            'monitoring_window': 30,  # days
            'min_sample_size': 100,
            'features_to_monitor': [
                'sentiment_score', 'text_length', 'word_count', 
                'financial_keyword_count', 'sentiment_balance',
                'avg_word_length', 'type_token_ratio'
            ]
        }
        
        self.reference_distributions = {}
        self.drift_history = []
        
    def establish_baseline(self, reference_data):
        """Establish baseline distributions for drift detection"""
        print("Establishing baseline distributions for drift detection...")
        
        baseline_stats = {}
        
        for feature in self.config['features_to_monitor']:
            if feature in reference_data.columns:
                feature_data = reference_data[feature].dropna()
                
                if len(feature_data) > 0:
                    baseline_stats[feature] = {
                        'mean': feature_data.mean(),
                        'std': feature_data.std(),
                        'median': feature_data.median(),
                        'q25': feature_data.quantile(0.25),
                        'q75': feature_data.quantile(0.75),
                        'distribution': feature_data.values,
                        'sample_size': len(feature_data)
                    }
        
        self.reference_distributions = baseline_stats
        print(f"Baseline established for {len(baseline_stats)} features")
        
        return baseline_stats
    
    def detect_text_feature_drift(self, new_data, feature_name):
        """Detect drift in specific text feature using KS test"""
        if feature_name not in self.reference_distributions:
            return None
            
        reference_dist = self.reference_distributions[feature_name]['distribution']
        new_feature_data = new_data[feature_name].dropna()
        
        if len(new_feature_data) < self.config['min_sample_size']:
            return None
            
        # Perform KS test
        ks_stat, p_value = stats.ks_2samp(reference_dist, new_feature_data)
        
        # Calculate distribution statistics
        new_mean = new_feature_data.mean()
        new_std = new_feature_data.std()
        ref_mean = self.reference_distributions[feature_name]['mean']
        ref_std = self.reference_distributions[feature_name]['std']
        
        # Calculate drift magnitude
        mean_drift = abs(new_mean - ref_mean) / ref_std if ref_std > 0 else 0
        std_drift = abs(new_std - ref_std) / ref_std if ref_std > 0 else 0
        
        drift_result = {
            'feature': feature_name,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'significant_drift': p_value < self.config['drift_threshold'],
            'mean_drift': mean_drift,
            'std_drift': std_drift,
            'reference_mean': ref_mean,
            'new_mean': new_mean,
            'reference_std': ref_std,
            'new_std': new_std,
            'sample_size': len(new_feature_data)
        }
        
        return drift_result
    
    def detect_embedding_drift(self, new_embeddings, reference_embeddings):
        """Detect drift in FinBERT embeddings using statistical tests"""
        print("Detecting embedding drift...")
        
        if len(new_embeddings) < self.config['min_sample_size']:
            return None
            
        # Calculate embedding statistics
        new_mean = np.mean(new_embeddings, axis=0)
        new_std = np.std(new_embeddings, axis=0)
        ref_mean = np.mean(reference_embeddings, axis=0)
        ref_std = np.std(reference_embeddings, axis=0)
        
        # Calculate drift metrics
        mean_drift = np.mean(np.abs(new_mean - ref_mean))
        std_drift = np.mean(np.abs(new_std - ref_std))
        
        # Perform multivariate KS test (simplified)
        # Use first principal component for univariate test
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=1)
        ref_pc = pca.fit_transform(reference_embeddings).flatten()
        new_pc = pca.transform(new_embeddings).flatten()
        
        ks_stat, p_value = stats.ks_2samp(ref_pc, new_pc)
        
        embedding_drift = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'significant_drift': p_value < self.config['drift_threshold'],
            'mean_drift': mean_drift,
            'std_drift': std_drift,
            'reference_mean_norm': np.linalg.norm(ref_mean),
            'new_mean_norm': np.linalg.norm(new_mean)
        }
        
        return embedding_drift
    
    def detect_performance_drift(self, reference_predictions, new_predictions, 
                                reference_labels, new_labels):
        """Detect drift in model performance metrics"""
        print("Detecting performance drift...")
        
        if len(new_predictions) < self.config['min_sample_size']:
            return None
            
        # Calculate performance metrics
        ref_auc = roc_auc_score(reference_labels, reference_predictions)
        new_auc = roc_auc_score(new_labels, new_predictions)
        
        # Calculate precision-recall metrics
        ref_precision, ref_recall, _ = precision_recall_curve(reference_labels, reference_predictions)
        new_precision, new_recall, _ = precision_recall_curve(new_labels, new_predictions)
        
        ref_pr_auc = np.trapz(ref_precision, ref_recall)
        new_pr_auc = np.trapz(new_precision, new_recall)
        
        # Calculate drift
        auc_drift = abs(new_auc - ref_auc)
        pr_auc_drift = abs(new_pr_auc - ref_pr_auc)
        
        # Statistical significance test (bootstrap)
        auc_significant = self.bootstrap_performance_test(
            reference_predictions, new_predictions, 
            reference_labels, new_labels, 'auc'
        )
        
        performance_drift = {
            'auc_drift': auc_drift,
            'pr_auc_drift': pr_auc_drift,
            'reference_auc': ref_auc,
            'new_auc': new_auc,
            'reference_pr_auc': ref_pr_auc,
            'new_pr_auc': new_pr_auc,
            'auc_significant': auc_significant,
            'performance_degradation': new_auc < ref_auc
        }
        
        return performance_drift
    
    def bootstrap_performance_test(self, ref_pred, new_pred, ref_labels, new_labels, metric):
        """Bootstrap test for performance difference significance"""
        n_bootstrap = 1000
        differences = []
        
        for _ in range(n_bootstrap):
            # Bootstrap samples
            ref_idx = np.random.choice(len(ref_pred), len(ref_pred), replace=True)
            new_idx = np.random.choice(len(new_pred), len(new_pred), replace=True)
            
            ref_pred_boot = ref_pred[ref_idx]
            ref_labels_boot = ref_labels[ref_idx]
            new_pred_boot = new_pred[new_idx]
            new_labels_boot = new_labels[new_idx]
            
            # Calculate metrics
            ref_metric = roc_auc_score(ref_labels_boot, ref_pred_boot)
            new_metric = roc_auc_score(new_labels_boot, new_pred_boot)
            
            differences.append(new_metric - ref_metric)
        
        # Calculate p-value
        p_value = np.mean(np.array(differences) <= 0)  # One-sided test
        return p_value < self.config['drift_threshold']
    
    def monitor_temporal_drift(self, data_stream, date_column='issue_d'):
        """Monitor drift over time using sliding windows"""
        print("Monitoring temporal drift...")
        
        # Sort by date
        data_stream = data_stream.sort_values(date_column).reset_index(drop=True)
        
        drift_timeline = []
        window_size = self.config['monitoring_window']
        
        for i in range(window_size, len(data_stream), window_size//2):  # 50% overlap
            window_data = data_stream.iloc[i-window_size:i]
            
            # Detect drift in each feature
            window_drift = {}
            for feature in self.config['features_to_monitor']:
                drift_result = self.detect_text_feature_drift(window_data, feature)
                if drift_result:
                    window_drift[feature] = drift_result
            
            # Aggregate drift metrics
            if window_drift:
                significant_drifts = sum(1 for d in window_drift.values() if d['significant_drift'])
                avg_drift_magnitude = np.mean([d['mean_drift'] for d in window_drift.values()])
                
                drift_timeline.append({
                    'window_start': data_stream.iloc[i-window_size][date_column],
                    'window_end': data_stream.iloc[i-1][date_column],
                    'significant_drifts': significant_drifts,
                    'avg_drift_magnitude': avg_drift_magnitude,
                    'feature_drifts': window_drift
                })
        
        return drift_timeline
    
    def generate_drift_alert(self, drift_results):
        """Generate alerts based on drift detection results"""
        alerts = []
        
        for feature, result in drift_results.items():
            if result['significant_drift']:
                alert = {
                    'type': 'FEATURE_DRIFT',
                    'feature': feature,
                    'severity': 'HIGH' if result['mean_drift'] > 1.0 else 'MEDIUM',
                    'message': f"Significant drift detected in {feature}: "
                              f"mean drift = {result['mean_drift']:.3f}, "
                              f"p-value = {result['p_value']:.4f}",
                    'recommendation': 'Consider retraining model or updating feature engineering'
                }
                alerts.append(alert)
        
        return alerts
    
    def create_drift_visualizations(self, drift_results, drift_timeline):
        """Create visualizations for drift monitoring"""
        from pathlib import Path
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Feature drift summary
        if drift_results:
            plt.figure(figsize=(12, 8))
            
            features = list(drift_results.keys())
            drift_magnitudes = [drift_results[f]['mean_drift'] for f in features]
            significant = [drift_results[f]['significant_drift'] for f in features]
            
            colors = ['red' if sig else 'blue' for sig in significant]
            
            plt.bar(features, drift_magnitudes, color=colors, alpha=0.7)
            plt.axhline(y=1.0, color='red', linestyle='--', label='High Drift Threshold')
            plt.xlabel('Features')
            plt.ylabel('Drift Magnitude')
            plt.title('Text Feature Drift Detection')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'feature_drift_summary.png')
            plt.close()
        
        # 2. Temporal drift timeline
        if drift_timeline:
            plt.figure(figsize=(12, 6))
            
            dates = [d['window_start'] for d in drift_timeline]
            significant_drifts = [d['significant_drifts'] for d in drift_timeline]
            avg_magnitudes = [d['avg_drift_magnitude'] for d in drift_timeline]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            ax1.plot(dates, significant_drifts, 'o-', color='red')
            ax1.set_ylabel('Number of Significant Drifts')
            ax1.set_title('Temporal Drift Monitoring')
            ax1.grid(True)
            
            ax2.plot(dates, avg_magnitudes, 'o-', color='blue')
            ax2.set_ylabel('Average Drift Magnitude')
            ax2.set_xlabel('Time')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'temporal_drift_timeline.png')
            plt.close()
    
    def run_comprehensive_drift_monitoring(self, reference_data):
        """Run comprehensive drift monitoring analysis"""
        print("Running comprehensive concept drift monitoring...")
        
        # Load data if not provided
        if reference_data is None:
            try:
                # Try to load real data first, fall back to synthetic if not available
                try:
                    reference_data = pd.read_csv('data/real_lending_club/real_lending_club_processed.csv')
                    print(f"Loaded REAL Lending Club dataset: {len(reference_data)} records")
                except FileNotFoundError:
                    reference_data = pd.read_csv('data/synthetic_loan_descriptions_with_realistic_targets.csv')
                    print(f"Using SYNTHETIC data (real data not found): {len(reference_data)} records")
                
                print(f"Dataset loaded: {len(reference_data)} records, {len(reference_data.columns)} columns")
                
            except FileNotFoundError:
                print("No dataset found. Please run real data processing first.")
                return None
        
        # Establish baseline
        baseline = self.establish_baseline(reference_data)
        
        # Split data for demonstration
        split_point = len(reference_data) // 2
        new_data = reference_data.iloc[split_point:]
        reference_data = reference_data.iloc[:split_point]
        baseline = self.establish_baseline(reference_data)
        
        # Detect drift in text features
        drift_results = {}
        for feature in self.config['features_to_monitor']:
            drift_result = self.detect_text_feature_drift(new_data, feature)
            if drift_result:
                drift_results[feature] = drift_result
        
        # Monitor temporal drift
        if 'issue_d' in reference_data.columns:
            combined_data = pd.concat([reference_data, new_data], ignore_index=True)
            drift_timeline = self.monitor_temporal_drift(combined_data)
        else:
            drift_timeline = []
        
        # Generate alerts
        alerts = self.generate_drift_alert(drift_results)
        
        # Create visualizations
        self.create_drift_visualizations(drift_results, drift_timeline)
        
        # Compile results
        monitoring_results = {
            'baseline_stats': baseline,
            'drift_results': drift_results,
            'drift_timeline': drift_timeline,
            'alerts': alerts,
            'summary': {
                'total_features_monitored': len(self.config['features_to_monitor']),
                'features_with_drift': sum(1 for r in drift_results.values() if r['significant_drift']),
                'total_alerts': len(alerts),
                'high_severity_alerts': sum(1 for a in alerts if a['severity'] == 'HIGH')
            }
        }
        
        # Save results
        self.save_monitoring_results(monitoring_results)
        
        return monitoring_results
    
    def save_monitoring_results(self, results):
        """Save drift monitoring results"""
        import json
        from pathlib import Path
        
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(output_dir / 'drift_monitoring_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary report
        self.create_monitoring_report(results, output_dir)
        
        print(f"Drift monitoring results saved to: {output_dir}")
    
    def create_monitoring_report(self, results, output_dir):
        """Create monitoring summary report"""
        report = []
        report.append("# Concept Drift Monitoring Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        summary = results['summary']
        report.append("## Monitoring Summary")
        report.append(f"- Features Monitored: {summary['total_features_monitored']}")
        report.append(f"- Features with Drift: {summary['features_with_drift']}")
        report.append(f"- Total Alerts: {summary['total_alerts']}")
        report.append(f"- High Severity Alerts: {summary['high_severity_alerts']}")
        report.append("")
        
        # Drift Results
        report.append("## Feature Drift Results")
        for feature, drift in results['drift_results'].items():
            status = "SIGNIFICANT DRIFT" if drift['significant_drift'] else "No Drift"
            report.append(f"### {feature}")
            report.append(f"- Status: {status}")
            report.append(f"- Mean Drift: {drift['mean_drift']:.4f}")
            report.append(f"- P-value: {drift['p_value']:.4f}")
            report.append(f"- Sample Size: {drift['sample_size']}")
            report.append("")
        
        # Alerts
        if results['alerts']:
            report.append("## Drift Alerts")
            for alert in results['alerts']:
                report.append(f"### {alert['type']} - {alert['severity']}")
                report.append(f"- Feature: {alert['feature']}")
                report.append(f"- Message: {alert['message']}")
                report.append(f"- Recommendation: {alert['recommendation']}")
                report.append("")
        
        # Save report
        with open(output_dir / 'drift_monitoring_report.md', 'w') as f:
            f.write('\n'.join(report))

if __name__ == "__main__":
    # Example usage
    monitor = ConceptDriftMonitor(random_state=42)
    
    # Load data
    try:
        df = pd.read_csv('data/real_lending_club/real_lending_club_processed.csv')
        print(f"Loaded dataset: {len(df)} records")
        
        # Run drift monitoring
        results = monitor.run_comprehensive_drift_monitoring(df)
        
        if results:
            print("Concept drift monitoring completed successfully!")
            print(f"Alerts generated: {len(results['alerts'])}")
        else:
            print("Drift monitoring failed.")
            
    except FileNotFoundError:
        print("Dataset not found. Please run real data processing first.") 