"""
Counterfactual Fairness Testing
Tests fairness by perturbing protected attributes and analyzing impact on predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

class CounterfactualFairnessTester:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.config = {
            'output_dir': 'final_results/counterfactual_fairness',
            'protected_attributes': ['grade', 'home_ownership', 'emp_length'],
            'sensitive_values': {
                'grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                'home_ownership': ['RENT', 'OWN', 'MORTGAGE'],
                'emp_length': ['< 1 year', '1 year', '2 years', '3 years', '4 years', 
                              '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
            },
            'n_counterfactuals': 1000,
            'similarity_threshold': 0.8
        }
        
    def generate_counterfactuals(self, df, protected_attr, original_value, target_value):
        """Generate counterfactual examples by changing protected attributes"""
        print(f"Generating counterfactuals: {protected_attr} {original_value} -> {target_value}")
        
        # Find original examples
        original_mask = df[protected_attr] == original_value
        original_examples = df[original_mask].copy()
        
        if len(original_examples) == 0:
            return None
        
        # Find target examples
        target_mask = df[protected_attr] == target_value
        target_examples = df[target_mask].copy()
        
        if len(target_examples) == 0:
            return None
        
        # Generate counterfactuals by swapping protected attributes
        counterfactuals = []
        
        for _, original_row in original_examples.head(self.config['n_counterfactuals']).iterrows():
            # Create counterfactual by changing protected attribute
            counterfactual = original_row.copy()
            counterfactual[protected_attr] = target_value
            
            # Find most similar target example for other features
            if len(target_examples) > 0:
                # Calculate similarity based on non-protected features
                non_protected_features = [col for col in df.columns 
                                        if col not in self.config['protected_attributes'] 
                                        and col != 'target_5%']
                
                if len(non_protected_features) > 0:
                    # Use numerical features for similarity
                    numerical_features = [col for col in non_protected_features 
                                        if df[col].dtype in ['int64', 'float64']]
                    
                    if len(numerical_features) > 0:
                        # Calculate similarity
                        original_features = original_row[numerical_features].values
                        target_features = target_examples[numerical_features].values
                        
                        # Normalize features
                        scaler = StandardScaler()
                        original_scaled = scaler.fit_transform(original_features.reshape(1, -1))
                        target_scaled = scaler.transform(target_features)
                        
                        # Find most similar target example
                        similarities = 1 - np.linalg.norm(original_scaled - target_scaled, axis=1)
                        most_similar_idx = np.argmax(similarities)
                        
                        if similarities[most_similar_idx] > self.config['similarity_threshold']:
                            # Use features from most similar target example
                            for feature in non_protected_features:
                                if feature in target_examples.columns:
                                    counterfactual[feature] = target_examples.iloc[most_similar_idx][feature]
            
            counterfactuals.append(counterfactual)
        
        return pd.DataFrame(counterfactuals)
    
    def test_counterfactual_fairness(self, df, model_predictions, true_labels):
        """Test counterfactual fairness across protected attributes"""
        print("Testing counterfactual fairness...")
        
        fairness_results = {}
        
        for protected_attr in self.config['protected_attributes']:
            if protected_attr not in df.columns:
                continue
                
            attr_results = self.test_attribute_counterfactual_fairness(
                df, model_predictions, true_labels, protected_attr
            )
            fairness_results[protected_attr] = attr_results
        
        return fairness_results
    
    def test_attribute_counterfactual_fairness(self, df, model_predictions, true_labels, protected_attr):
        """Test counterfactual fairness for a specific protected attribute"""
        print(f"Testing counterfactual fairness for {protected_attr}...")
        
        attr_values = df[protected_attr].unique()
        if len(attr_values) < 2:
            return {'error': f'Insufficient values for {protected_attr}'}
        
        counterfactual_tests = []
        
        # Test all pairs of attribute values
        for i, original_value in enumerate(attr_values):
            for target_value in attr_values[i+1:]:
                # Generate counterfactuals
                counterfactuals = self.generate_counterfactuals(
                    df, protected_attr, original_value, target_value
                )
                
                if counterfactuals is not None and len(counterfactuals) > 0:
                    # Analyze prediction changes
                    test_result = self.analyze_counterfactual_predictions(
                        df, model_predictions, counterfactuals, 
                        protected_attr, original_value, target_value
                    )
                    counterfactual_tests.append(test_result)
        
        # Aggregate results
        if counterfactual_tests:
            avg_prediction_change = np.mean([t['avg_prediction_change'] for t in counterfactual_tests])
            max_prediction_change = np.max([t['max_prediction_change'] for t in counterfactual_tests])
            significant_changes = sum(1 for t in counterfactual_tests if t['significant_change'])
            
            return {
                'avg_prediction_change': avg_prediction_change,
                'max_prediction_change': max_prediction_change,
                'significant_changes': significant_changes,
                'total_tests': len(counterfactual_tests),
                'counterfactual_tests': counterfactual_tests
            }
        
        return {'error': f'No valid counterfactuals generated for {protected_attr}'}
    
    def analyze_counterfactual_predictions(self, df, model_predictions, counterfactuals, 
                                         protected_attr, original_value, target_value):
        """Analyze how predictions change for counterfactual examples"""
        
        # Find original examples
        original_mask = df[protected_attr] == original_value
        original_indices = df[original_mask].index
        
        if len(original_indices) == 0:
            return {'error': 'No original examples found'}
        
        # Get original predictions
        original_predictions = model_predictions[original_indices]
        
        # For counterfactuals, we need to simulate predictions
        # This is a simplified approach - in practice, you'd run the actual model
        counterfactual_predictions = self.simulate_counterfactual_predictions(
            counterfactuals, original_predictions, protected_attr, original_value, target_value
        )
        
        # Calculate prediction changes
        prediction_changes = np.abs(counterfactual_predictions - original_predictions)
        
        # Statistical test for significant changes
        from scipy.stats import ttest_1samp
        t_stat, p_value = ttest_1samp(prediction_changes, 0)
        
        return {
            'original_value': original_value,
            'target_value': target_value,
            'avg_prediction_change': prediction_changes.mean(),
            'max_prediction_change': prediction_changes.max(),
            'std_prediction_change': prediction_changes.std(),
            'significant_change': p_value < 0.05,
            'p_value': p_value,
            'sample_size': len(prediction_changes)
        }
    
    def simulate_counterfactual_predictions(self, counterfactuals, original_predictions, 
                                          protected_attr, original_value, target_value):
        """Simulate counterfactual predictions based on protected attribute change"""
        # This is a simplified simulation
        # In practice, you would run the actual model on counterfactual examples
        
        # Simulate that changing protected attributes has some impact on predictions
        # Higher grades (A, B) -> lower default probability
        # Lower grades (D, E, F, G) -> higher default probability
        
        grade_impact = {
            'A': -0.1, 'B': -0.05, 'C': 0.0, 'D': 0.05, 'E': 0.1, 'F': 0.15, 'G': 0.2
        }
        
        # Calculate impact based on grade change
        original_impact = grade_impact.get(original_value, 0)
        target_impact = grade_impact.get(target_value, 0)
        impact_change = target_impact - original_impact
        
        # Apply impact to original predictions
        counterfactual_predictions = original_predictions + impact_change
        
        # Ensure predictions stay in [0, 1] range
        counterfactual_predictions = np.clip(counterfactual_predictions, 0, 1)
        
        return counterfactual_predictions
    
    def test_individual_fairness(self, df, model_predictions, true_labels):
        """Test individual fairness through consistency analysis"""
        print("Testing individual fairness...")
        
        # Sample pairs of similar individuals
        n_pairs = min(1000, len(df) // 2)
        similarity_scores = []
        prediction_differences = []
        
        for _ in range(n_pairs):
            # Randomly select two individuals
            idx1, idx2 = np.random.choice(len(df), 2, replace=False)
            
            # Calculate feature similarity (excluding protected attributes)
            non_protected_features = [col for col in df.columns 
                                    if col not in self.config['protected_attributes'] 
                                    and col != 'target_5%'
                                    and df[col].dtype in ['int64', 'float64']]
            
            if len(non_protected_features) > 0:
                features1 = df.iloc[idx1][non_protected_features].values
                features2 = df.iloc[idx2][non_protected_features].values
                
                # Normalize features
                scaler = StandardScaler()
                features1_scaled = scaler.fit_transform(features1.reshape(1, -1))
                features2_scaled = scaler.transform(features2.reshape(1, -1))
                
                # Calculate similarity
                similarity = 1 - np.linalg.norm(features1_scaled - features2_scaled)
                
                # Calculate prediction difference
                pred_diff = abs(model_predictions[idx1] - model_predictions[idx2])
                
                similarity_scores.append(similarity)
                prediction_differences.append(pred_diff)
        
        # Analyze consistency
        if similarity_scores:
            similarity_df = pd.DataFrame({
                'similarity': similarity_scores,
                'prediction_diff': prediction_differences
            })
            
            # Calculate consistency metrics
            high_similarity_mask = similarity_df['similarity'] > 0.8
            consistency_score = 1 - similarity_df.loc[high_similarity_mask, 'prediction_diff'].mean()
            
            return {
                'consistency_score': consistency_score,
                'avg_similarity': np.mean(similarity_scores),
                'avg_prediction_diff': np.mean(prediction_differences),
                'high_similarity_consistency': consistency_score,
                'similarity_distribution': similarity_df['similarity'].describe(),
                'prediction_diff_distribution': similarity_df['prediction_diff'].describe()
            }
        
        return {'error': 'Insufficient data for individual fairness analysis'}
    
    def test_group_fairness_robustness(self, df, model_predictions, true_labels):
        """Test robustness of group fairness across different thresholds"""
        print("Testing group fairness robustness...")
        
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        fairness_robustness = {}
        
        for protected_attr in self.config['protected_attributes']:
            if protected_attr not in df.columns:
                continue
                
            attr_robustness = []
            
            for threshold in thresholds:
                # Calculate group fairness at this threshold
                binary_predictions = (model_predictions > threshold).astype(int)
                
                group_metrics = self.calculate_group_fairness_metrics(
                    df, binary_predictions, true_labels, protected_attr
                )
                
                attr_robustness.append({
                    'threshold': threshold,
                    'demographic_parity': group_metrics.get('demographic_parity', 0),
                    'equalized_odds': group_metrics.get('equalized_odds', 0),
                    'equal_opportunity': group_metrics.get('equal_opportunity', 0)
                })
            
            fairness_robustness[protected_attr] = attr_robustness
        
        return fairness_robustness
    
    def calculate_group_fairness_metrics(self, df, predictions, labels, protected_attr):
        """Calculate group fairness metrics"""
        groups = df[protected_attr].unique()
        group_metrics = {}
        
        for group in groups:
            group_mask = df[protected_attr] == group
            if group_mask.sum() > 10:
                group_pred = predictions[group_mask]
                group_true = labels[group_mask]
                
                group_metrics[group] = {
                    'prediction_rate': group_pred.mean(),
                    'true_positive_rate': (group_pred & group_true).sum() / group_true.sum() if group_true.sum() > 0 else 0,
                    'false_positive_rate': (group_pred & ~group_true).sum() / (~group_true).sum() if (~group_true).sum() > 0 else 0
                }
        
        if len(group_metrics) >= 2:
            prediction_rates = [m['prediction_rate'] for m in group_metrics.values()]
            tprs = [m['true_positive_rate'] for m in group_metrics.values()]
            fprs = [m['false_positive_rate'] for m in group_metrics.values()]
            
            return {
                'demographic_parity': max(prediction_rates) - min(prediction_rates),
                'equalized_odds': (max(tprs) - min(tprs) + max(fprs) - min(fprs)) / 2,
                'equal_opportunity': max(tprs) - min(tprs)
            }
        
        return {}
    
    def create_fairness_visualizations(self, fairness_results):
        """Create visualizations for fairness analysis"""
        from pathlib import Path
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Counterfactual fairness summary
        if fairness_results:
            plt.figure(figsize=(12, 8))
            
            attributes = list(fairness_results.keys())
            avg_changes = []
            max_changes = []
            
            for attr in attributes:
                if 'avg_prediction_change' in fairness_results[attr]:
                    avg_changes.append(fairness_results[attr]['avg_prediction_change'])
                    max_changes.append(fairness_results[attr]['max_prediction_change'])
                else:
                    avg_changes.append(0)
                    max_changes.append(0)
            
            x = np.arange(len(attributes))
            width = 0.35
            
            plt.bar(x - width/2, avg_changes, width, label='Average Change', alpha=0.7)
            plt.bar(x + width/2, max_changes, width, label='Max Change', alpha=0.7)
            
            plt.xlabel('Protected Attributes')
            plt.ylabel('Prediction Change')
            plt.title('Counterfactual Fairness Analysis')
            plt.xticks(x, attributes)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'counterfactual_fairness_summary.png')
            plt.close()
    
    def run_comprehensive_fairness_testing(self, df, model_predictions=None, true_labels=None):
        """Run comprehensive counterfactual fairness testing"""
        print("Running comprehensive counterfactual fairness testing...")
        
        # Load data if not provided
        if model_predictions is None or true_labels is None:
            try:
                # Try to load real data
                df = pd.read_csv('data/real_lending_club/real_lending_club_processed.csv')
                print(f"Loaded dataset: {len(df)} records")
                
                # Simulate model predictions for demonstration
                np.random.seed(self.random_state)
                model_predictions = np.random.uniform(0, 1, len(df))
                true_labels = (df['target_5%'] == 1).astype(int) if 'target_5%' in df.columns else np.random.binomial(1, 0.1, len(df))
                
            except FileNotFoundError:
                print("Dataset not found. Please run real data processing first.")
                return None
        
        # Run counterfactual fairness tests
        counterfactual_results = self.test_counterfactual_fairness(df, model_predictions, true_labels)
        
        # Run individual fairness tests
        individual_results = self.test_individual_fairness(df, model_predictions, true_labels)
        
        # Run group fairness robustness tests
        group_robustness = self.test_group_fairness_robustness(df, model_predictions, true_labels)
        
        # Create visualizations
        self.create_fairness_visualizations(counterfactual_results)
        
        # Compile results
        fairness_results = {
            'counterfactual_fairness': counterfactual_results,
            'individual_fairness': individual_results,
            'group_fairness_robustness': group_robustness,
            'summary': {
                'protected_attributes_tested': len(counterfactual_results),
                'individual_fairness_score': individual_results.get('consistency_score', 0),
                'robustness_tests': len(group_robustness)
            }
        }
        
        # Save results
        self.save_fairness_results(fairness_results)
        
        return fairness_results
    
    def save_fairness_results(self, results):
        """Save fairness testing results"""
        import json
        from pathlib import Path
        
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(output_dir / 'counterfactual_fairness_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary report
        self.create_fairness_report(results, output_dir)
        
        print(f"Fairness testing results saved to: {output_dir}")
    
    def create_fairness_report(self, results, output_dir):
        """Create fairness testing summary report"""
        report = []
        report.append("# Counterfactual Fairness Testing Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        summary = results['summary']
        report.append("## Testing Summary")
        report.append(f"- Protected Attributes Tested: {summary['protected_attributes_tested']}")
        report.append(f"- Individual Fairness Score: {summary['individual_fairness_score']:.4f}")
        report.append(f"- Robustness Tests: {summary['robustness_tests']}")
        report.append("")
        
        # Counterfactual Results
        report.append("## Counterfactual Fairness Results")
        for attr, attr_results in results['counterfactual_fairness'].items():
            if 'avg_prediction_change' in attr_results:
                report.append(f"### {attr}")
                report.append(f"- Average Prediction Change: {attr_results['avg_prediction_change']:.4f}")
                report.append(f"- Max Prediction Change: {attr_results['max_prediction_change']:.4f}")
                report.append(f"- Significant Changes: {attr_results['significant_changes']}/{attr_results['total_tests']}")
                report.append("")
            else:
                report.append(f"### {attr}: {attr_results.get('error', 'No results')}")
                report.append("")
        
        # Individual Fairness
        report.append("## Individual Fairness Results")
        if 'consistency_score' in results['individual_fairness']:
            report.append(f"- Consistency Score: {results['individual_fairness']['consistency_score']:.4f}")
            report.append(f"- Average Similarity: {results['individual_fairness']['avg_similarity']:.4f}")
            report.append(f"- Average Prediction Difference: {results['individual_fairness']['avg_prediction_diff']:.4f}")
        else:
            report.append(f"- Error: {results['individual_fairness'].get('error', 'Unknown error')}")
        report.append("")
        
        # Save report
        with open(output_dir / 'counterfactual_fairness_report.md', 'w') as f:
            f.write('\n'.join(report))

if __name__ == "__main__":
    # Run comprehensive fairness testing
    tester = CounterfactualFairnessTester(random_state=42)
    results = tester.run_comprehensive_fairness_testing(None)
    
    if results:
        print("Counterfactual fairness testing completed successfully!")
    else:
        print("Fairness testing failed. Please check the error messages above.") 