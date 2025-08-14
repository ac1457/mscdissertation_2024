#!/usr/bin/env python3
"""
Realistic Target Creation - Lending Club Sentiment Analysis
=========================================================
Creates realistic synthetic targets based on actual loan features
rather than random noise. This ensures meaningful relationships
between features and target variables.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class RealisticTargetCreation:
    """
    Creates realistic synthetic targets based on actual features
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def load_data(self):
        """
        Load synthetic loan descriptions data
        """
        try:
            df = pd.read_csv('data/synthetic_loan_descriptions.csv')
            print(f"✅ Loaded dataset: {len(df)} records")
            return df
        except FileNotFoundError:
            print("❌ synthetic_loan_descriptions.csv not found")
            return None
    
    def create_realistic_default_target(self, df, base_default_rate=0.08):
        """
        Create realistic default target based on actual loan features
        """
        print("Creating realistic default target based on actual features...")
        
        # Base default rate
        base_prob = base_default_rate
        
        # Create risk factors based on actual features
        risk_factors = []
        
        # 1. Sentiment-based risk
        if 'sentiment_score' in df.columns:
            # Negative sentiment increases default risk
            sentiment_risk = (0.5 - df['sentiment_score']) * 0.15
            risk_factors.append(sentiment_risk)
            print(f"  Sentiment risk factor: mean={sentiment_risk.mean():.4f}, std={sentiment_risk.std():.4f}")
        
        # 2. Text complexity risk
        if 'text_length' in df.columns and 'word_count' in df.columns:
            # Longer, more complex descriptions might indicate issues
            complexity_risk = (df['text_length'] / 100) * 0.02 + (df['word_count'] / 20) * 0.01
            risk_factors.append(complexity_risk)
            print(f"  Complexity risk factor: mean={complexity_risk.mean():.4f}, std={complexity_risk.std():.4f}")
        
        # 3. Financial terms risk
        if 'has_financial_terms' in df.columns:
            # Presence of financial terms might indicate higher risk
            financial_risk = df['has_financial_terms'] * 0.03
            risk_factors.append(financial_risk)
            print(f"  Financial terms risk factor: mean={financial_risk.mean():.4f}, std={financial_risk.std():.4f}")
        
        # 4. Purpose-based risk
        if 'purpose' in df.columns:
            # Different purposes have different risk levels
            purpose_risk_mapping = {
                'debt_consolidation': 0.05,  # Higher risk
                'credit_card': 0.04,         # Higher risk
                'home_improvement': 0.02,    # Medium risk
                'major_purchase': 0.02,      # Medium risk
                'small_business': 0.06,      # Higher risk
                'car': 0.01,                 # Lower risk
                'medical': 0.03,             # Medium risk
                'vacation': 0.01,            # Lower risk
                'moving': 0.02,              # Medium risk
                'house': 0.01,               # Lower risk
                'wedding': 0.02,             # Medium risk
                'educational': 0.01,         # Lower risk
                'renewable_energy': 0.02,    # Medium risk
                'other': 0.03                # Default medium risk
            }
            
            purpose_risk = df['purpose'].map(purpose_risk_mapping).fillna(0.03)
            risk_factors.append(purpose_risk)
            print(f"  Purpose risk factor: mean={purpose_risk.mean():.4f}, std={purpose_risk.std():.4f}")
        
        # 5. Sentiment confidence risk
        if 'sentiment_confidence' in df.columns:
            # Low confidence might indicate uncertainty/risk
            confidence_risk = (1 - df['sentiment_confidence']) * 0.02
            risk_factors.append(confidence_risk)
            print(f"  Confidence risk factor: mean={confidence_risk.mean():.4f}, std={confidence_risk.std():.4f}")
        
        # Combine all risk factors
        if risk_factors:
            total_risk = np.sum(risk_factors, axis=0)
        else:
            total_risk = np.zeros(len(df))
        
        # Create realistic default probability
        default_prob = base_prob + total_risk
        
        # Ensure reasonable bounds (1% to 30%)
        default_prob = np.clip(default_prob, 0.01, 0.30)
        
        # Add some realistic noise/variation
        noise = np.random.normal(0, 0.005, len(df))
        default_prob += noise
        default_prob = np.clip(default_prob, 0.01, 0.30)
        
        # Generate target based on realistic probabilities
        y = np.random.binomial(1, default_prob)
        
        # Calculate final statistics
        actual_default_rate = np.mean(y)
        print(f"  Base default rate: {base_prob:.1%}")
        print(f"  Actual default rate: {actual_default_rate:.1%}")
        print(f"  Total defaults: {np.sum(y):,}")
        print(f"  Total non-defaults: {len(y) - np.sum(y):,}")
        
        return y, default_prob
    
    def create_multiple_realistic_regimes(self, df):
        """
        Create multiple realistic regimes with different default rates
        """
        print("Creating multiple realistic regimes...")
        
        regimes = {}
        
        # Create different base rates
        base_rates = [0.05, 0.10, 0.15]  # 5%, 10%, 15%
        
        for base_rate in base_rates:
            print(f"\nCreating {base_rate*100:.0f}% regime...")
            
            # Create realistic target for this regime
            y, default_prob = self.create_realistic_default_target(df, base_rate)
            
            # Calculate regime statistics
            n_total = len(y)
            n_positives = np.sum(y)
            n_negatives = n_total - n_positives
            actual_rate = n_positives / n_total
            
            regimes[f"{base_rate*100:.0f}%"] = {
                'y': y,
                'default_prob': default_prob,
                'base_rate': base_rate,
                'actual_rate': actual_rate,
                'n_total': n_total,
                'n_positives': n_positives,
                'n_negatives': n_negatives
            }
            
            print(f"  Target: {base_rate*100:.0f}%, Actual: {actual_rate*100:.1f}%")
            print(f"  Positives: {n_positives:,}, Negatives: {n_negatives:,}")
        
        return regimes
    
    def analyze_target_relationships(self, df, y, default_prob):
        """
        Analyze relationships between features and target
        """
        print("\nAnalyzing feature-target relationships...")
        
        # Create analysis dataframe
        analysis_df = df.copy()
        analysis_df['target'] = y
        analysis_df['default_prob'] = default_prob
        
        # Analyze correlations
        correlations = {}
        
        numeric_features = ['sentiment_score', 'sentiment_confidence', 'text_length', 'word_count', 'sentence_count']
        
        for feature in numeric_features:
            if feature in analysis_df.columns:
                corr = analysis_df[feature].corr(analysis_df['target'])
                correlations[feature] = corr
                print(f"  {feature} correlation with target: {corr:.4f}")
        
        # Analyze by purpose
        if 'purpose' in analysis_df.columns:
            print("\n  Default rates by purpose:")
            purpose_defaults = analysis_df.groupby('purpose')['target'].agg(['mean', 'count'])
            purpose_defaults = purpose_defaults.sort_values('mean', ascending=False)
            
            for purpose, row in purpose_defaults.head(10).iterrows():
                print(f"    {purpose}: {row['mean']:.1%} ({row['count']} loans)")
        
        # Analyze by sentiment
        if 'sentiment' in analysis_df.columns:
            print("\n  Default rates by sentiment:")
            sentiment_defaults = analysis_df.groupby('sentiment')['target'].agg(['mean', 'count'])
            
            for sentiment, row in sentiment_defaults.iterrows():
                print(f"    {sentiment}: {row['mean']:.1%} ({row['count']} loans)")
        
        return correlations
    
    def save_realistic_targets(self, df, regimes):
        """
        Save realistic targets to file
        """
        print("\nSaving realistic targets...")
        
        # Create comprehensive dataset with all targets
        enhanced_df = df.copy()
        
        for regime_name, regime_data in regimes.items():
            enhanced_df[f'target_{regime_name}'] = regime_data['y']
            enhanced_df[f'default_prob_{regime_name}'] = regime_data['default_prob']
        
        # Save enhanced dataset
        enhanced_df.to_csv('data/synthetic_loan_descriptions_with_realistic_targets.csv', index=False)
        
        # Save regime summary
        regime_summary = []
        for regime_name, regime_data in regimes.items():
            regime_summary.append({
                'Regime': regime_name,
                'Base_Rate': regime_data['base_rate'],
                'Actual_Rate': regime_data['actual_rate'],
                'Total_Loans': regime_data['n_total'],
                'Defaults': regime_data['n_positives'],
                'Non_Defaults': regime_data['n_negatives']
            })
        
        regime_summary_df = pd.DataFrame(regime_summary)
        regime_summary_df.to_csv('final_results/realistic_target_regime_summary.csv', index=False)
        
        print("✅ Saved realistic targets:")
        print("  - data/synthetic_loan_descriptions_with_realistic_targets.csv")
        print("  - final_results/realistic_target_regime_summary.csv")
        
        return enhanced_df, regime_summary_df
    
    def run_complete_target_creation(self):
        """
        Run complete realistic target creation process
        """
        print("REALISTIC TARGET CREATION")
        print("=" * 40)
        
        # Load data
        df = self.load_data()
        if df is None:
            return None
        
        # Create multiple realistic regimes
        regimes = self.create_multiple_realistic_regimes(df)
        
        # Analyze relationships for 10% regime
        print("\n" + "="*50)
        print("DETAILED ANALYSIS: 10% REGIME")
        print("="*50)
        
        y_10 = regimes['10%']['y']
        default_prob_10 = regimes['10%']['default_prob']
        
        correlations = self.analyze_target_relationships(df, y_10, default_prob_10)
        
        # Save realistic targets
        enhanced_df, regime_summary = self.save_realistic_targets(df, regimes)
        
        # Generate report
        report = self.generate_target_creation_report(regimes, correlations)
        
        with open('methodology/realistic_target_creation_report.txt', 'w') as f:
            f.write(report)
        
        print("\n✅ Realistic target creation complete!")
        print("✅ Saved report: methodology/realistic_target_creation_report.txt")
        
        return regimes, enhanced_df, correlations
    
    def generate_target_creation_report(self, regimes, correlations):
        """
        Generate report on realistic target creation
        """
        report = []
        report.append("REALISTIC TARGET CREATION REPORT")
        report.append("=" * 40)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        report.append("This report documents the creation of realistic synthetic targets")
        report.append("based on actual loan features rather than random noise.")
        report.append("Targets are designed to reflect realistic relationships between")
        report.append("loan characteristics and default probability.")
        report.append("")
        
        # Methodology
        report.append("METHODOLOGY")
        report.append("-" * 15)
        report.append("• Base default rate established for each regime")
        report.append("• Risk factors calculated from actual features:")
        report.append("  - Sentiment score (negative sentiment increases risk)")
        report.append("  - Text complexity (longer descriptions indicate issues)")
        report.append("  - Financial terms (presence indicates higher risk)")
        report.append("  - Purpose (different purposes have different risk levels)")
        report.append("  - Sentiment confidence (low confidence indicates uncertainty)")
        report.append("• Realistic noise added to probabilities")
        report.append("• Reasonable bounds enforced (1% to 30% default rate)")
        report.append("")
        
        # Regime Summary
        report.append("REGIME SUMMARY")
        report.append("-" * 15)
        
        for regime_name, regime_data in regimes.items():
            report.append(f"\n{regime_name} Regime:")
            report.append(f"  Base rate: {regime_data['base_rate']:.1%}")
            report.append(f"  Actual rate: {regime_data['actual_rate']:.1%}")
            report.append(f"  Total loans: {regime_data['n_total']:,}")
            report.append(f"  Defaults: {regime_data['n_positives']:,}")
            report.append(f"  Non-defaults: {regime_data['n_negatives']:,}")
        
        # Feature Relationships
        report.append("\nFEATURE-TARGET RELATIONSHIPS")
        report.append("-" * 30)
        
        for feature, corr in correlations.items():
            report.append(f"  {feature}: {corr:+.4f}")
        
        # Limitations
        report.append("\nLIMITATIONS")
        report.append("-" * 12)
        report.append("• Synthetic targets based on assumptions about risk factors")
        report.append("• Real-world default rates may differ significantly")
        report.append("• Feature relationships are simplified approximations")
        report.append("• No temporal component (all loans treated as contemporaneous)")
        report.append("• No external economic factors considered")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 16)
        report.append("• Use real loan performance data when available")
        report.append("• Validate assumptions with domain experts")
        report.append("• Consider temporal and economic factors")
        report.append("• Test with external datasets")
        report.append("• Acknowledge synthetic nature in conclusions")
        
        return "\n".join(report)

if __name__ == "__main__":
    target_creator = RealisticTargetCreation()
    results = target_creator.run_complete_target_creation()
    print("✅ Realistic target creation execution complete!") 