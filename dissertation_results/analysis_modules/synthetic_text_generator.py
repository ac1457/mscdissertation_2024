#!/usr/bin/env python3
"""
Synthetic Text Generator for Loan Descriptions
==============================================
Generates realistic loan descriptions with controlled sentiment patterns
for ethical research and comprehensive testing
"""

import pandas as pd
import numpy as np
import random
from textblob import TextBlob
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SyntheticTextGenerator:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Loan purpose templates
        self.loan_purposes = {
            'debt_consolidation': {
                'positive': [
                    "I need to consolidate my high-interest credit card debt into a single, manageable payment. I have a stable job and am committed to improving my financial situation.",
                    "Looking to consolidate multiple credit cards with high interest rates. I have excellent payment history and want to simplify my finances.",
                    "I want to consolidate my debts to reduce monthly payments and interest costs. I have a good credit score and stable income."
                ],
                'negative': [
                    "I have multiple credit cards maxed out and need to consolidate before they go to collections. My income is irregular.",
                    "Need to consolidate debts because I'm struggling to make minimum payments. Have some late payments recently.",
                    "Trying to consolidate debts but my credit score has dropped due to missed payments. Need help getting back on track."
                ],
                'neutral': [
                    "I want to consolidate my existing debts into one loan. I have various credit cards and personal loans.",
                    "Looking to consolidate multiple debts for better organization. I have a mix of credit cards and other loans.",
                    "Need to consolidate my current debts. I have several credit accounts that I want to combine."
                ]
            },
            'home_improvement': {
                'positive': [
                    "Planning to renovate my kitchen to increase home value. I have a stable job and good credit history.",
                    "Want to upgrade my home's energy efficiency with new windows and insulation. I have excellent payment history.",
                    "Looking to remodel my bathroom to modernize the house. I have a good income and strong credit score."
                ],
                'negative': [
                    "Need emergency repairs on my home due to water damage. My income has been reduced recently.",
                    "Have to fix major structural issues in my house. My credit has suffered due to medical bills.",
                    "Need urgent home repairs but my financial situation is tight. Have some payment issues recently."
                ],
                'neutral': [
                    "Planning home improvements to update the property. I have various projects in mind.",
                    "Want to make some upgrades to my house. I have a few different improvement projects planned.",
                    "Looking to do some home renovations. I have several areas that need updating."
                ]
            },
            'business': {
                'positive': [
                    "Starting a new business venture with a solid business plan. I have strong financial backing and experience.",
                    "Expanding my successful small business. I have excellent credit and proven track record.",
                    "Investing in new equipment for my growing business. I have stable revenue and good credit history."
                ],
                'negative': [
                    "Trying to save my struggling business. Sales have been declining and I'm behind on some payments.",
                    "Need emergency funding for my business. Have been having cash flow problems recently.",
                    "Trying to keep my business afloat. Have had some financial difficulties and payment issues."
                ],
                'neutral': [
                    "Need funding for business expansion. I have various growth opportunities planned.",
                    "Looking to invest in my business. I have several areas that need funding.",
                    "Need capital for business development. I have different projects that require financing."
                ]
            },
            'medical': {
                'positive': [
                    "Planning elective surgery with good insurance coverage. I have stable income and excellent credit.",
                    "Need medical procedure that's well-planned. I have good health insurance and financial stability.",
                    "Scheduling important medical treatment. I have strong credit and stable employment."
                ],
                'negative': [
                    "Emergency medical expenses that insurance won't cover. My income has been reduced due to illness.",
                    "Need urgent medical treatment but struggling financially. Have missed some payments due to medical bills.",
                    "Medical emergency that has impacted my ability to work. Have some credit issues due to medical expenses."
                ],
                'neutral': [
                    "Need funding for medical expenses. I have various healthcare costs to cover.",
                    "Looking to finance medical treatment. I have several medical bills to pay.",
                    "Need loan for healthcare expenses. I have different medical costs to address."
                ]
            },
            'education': {
                'positive': [
                    "Pursuing advanced degree to advance my career. I have excellent academic record and stable job.",
                    "Investing in professional certification to increase earning potential. I have strong credit and good income.",
                    "Funding education to improve career prospects. I have stable employment and good payment history."
                ],
                'negative': [
                    "Need to finish degree but struggling financially. Have some payment issues due to reduced income.",
                    "Trying to pay for education but facing financial difficulties. Have missed some payments recently.",
                    "Need education funding but my financial situation is tight. Have some credit problems."
                ],
                'neutral': [
                    "Need funding for educational expenses. I have various education costs to cover.",
                    "Looking to finance my education. I have several educational expenses to pay.",
                    "Need loan for education. I have different education-related costs to address."
                ]
            }
        }
        
        # Financial situation templates
        self.financial_contexts = {
            'positive': [
                "I have a stable job with good income and excellent credit history.",
                "My employment is secure and I have strong payment history.",
                "I have stable income and good credit score with no late payments.",
                "My financial situation is stable with consistent income and good credit.",
                "I have excellent credit history and stable employment."
            ],
            'negative': [
                "My income has been reduced recently and I have some payment issues.",
                "I've been struggling financially and have missed some payments.",
                "My credit has suffered due to recent financial difficulties.",
                "I have some payment problems and my income is irregular.",
                "My financial situation has been challenging recently."
            ],
            'neutral': [
                "I have a regular income and standard credit history.",
                "My employment is stable and I have typical payment history.",
                "I have consistent income and average credit score.",
                "My financial situation is normal with regular income.",
                "I have standard credit history and stable employment."
            ]
        }
        
        # Sentiment modifiers
        self.sentiment_modifiers = {
            'positive': [
                "I am confident in my ability to repay this loan.",
                "I have a solid plan for managing this debt responsibly.",
                "I am committed to maintaining my good credit standing.",
                "I have carefully considered this financial decision.",
                "I am optimistic about my financial future."
            ],
            'negative': [
                "I am concerned about my ability to make payments.",
                "I am struggling to manage my current financial obligations.",
                "I am worried about my financial situation.",
                "I am uncertain about my ability to repay.",
                "I am facing financial challenges."
            ],
            'neutral': [
                "I plan to manage this loan responsibly.",
                "I will make payments as agreed.",
                "I understand my financial obligations.",
                "I will maintain my payment schedule.",
                "I am committed to meeting my financial commitments."
            ]
        }
    
    def generate_synthetic_description(self, purpose, sentiment, financial_context, length='medium'):
        """Generate synthetic loan description with controlled sentiment"""
        
        # Select base template
        if purpose in self.loan_purposes:
            templates = self.loan_purposes[purpose][sentiment]
            base_template = random.choice(templates)
        else:
            # Fallback template
            base_template = f"I need a loan for {purpose}. I have various expenses to cover."
        
        # Add financial context
        financial_context_template = random.choice(self.financial_contexts[sentiment])
        
        # Add sentiment modifier
        sentiment_modifier = random.choice(self.sentiment_modifiers[sentiment])
        
        # Combine elements
        description = f"{base_template} {financial_context_template} {sentiment_modifier}"
        
        # Adjust length if needed
        if length == 'short':
            # Keep only first sentence
            sentences = description.split('.')
            description = sentences[0] + '.'
        elif length == 'long':
            # Add additional context
            additional_contexts = [
                "I have researched various loan options and believe this is the best choice for my situation.",
                "I have discussed this with my financial advisor and feel confident about this decision.",
                "I have compared different lenders and found this to be the most suitable option.",
                "I have carefully planned my budget to accommodate this loan payment.",
                "I have considered the long-term implications of this financial decision."
            ]
            description += " " + random.choice(additional_contexts)
        
        return description.strip()
    
    def generate_dataset_with_synthetic_text(self, n_samples=10000, sentiment_distribution=None):
        """Generate complete dataset with synthetic text descriptions"""
        
        if sentiment_distribution is None:
            sentiment_distribution = {'positive': 0.2, 'negative': 0.3, 'neutral': 0.5}
        
        # Generate purposes
        purposes = list(self.loan_purposes.keys())
        
        # Generate synthetic data
        data = []
        
        for i in range(n_samples):
            # Select purpose
            purpose = random.choice(purposes)
            
            # Select sentiment based on distribution
            sentiment = np.random.choice(
                list(sentiment_distribution.keys()),
                p=list(sentiment_distribution.values())
            )
            
            # Generate description
            description = self.generate_synthetic_description(
                purpose, sentiment, sentiment, length=random.choice(['short', 'medium', 'long'])
            )
            
            # Calculate sentiment score
            blob = TextBlob(description)
            sentiment_score = (blob.sentiment.polarity + 1) / 2  # Normalize to 0-1
            
            # Generate confidence score
            confidence_score = random.uniform(0.6, 0.95)
            
            # Create record
            record = {
                'id': i + 1,
                'purpose': purpose,
                'description': description,
                'sentiment': sentiment.upper(),
                'sentiment_score': sentiment_score,
                'sentiment_confidence': confidence_score,
                'text_length': len(description),
                'word_count': len(description.split()),
                'sentence_count': len(description.split('.')),
                'has_positive_words': int(any(word in description.lower() for word in ['good', 'excellent', 'stable', 'strong', 'confident'])),
                'has_negative_words': int(any(word in description.lower() for word in ['struggling', 'difficult', 'worried', 'concerned', 'emergency'])),
                'has_financial_terms': int(any(word in description.lower() for word in ['credit', 'payment', 'income', 'debt', 'financial']))
            }
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def validate_synthetic_text(self, df):
        """Validate the quality and realism of synthetic text"""
        
        print("VALIDATING SYNTHETIC TEXT QUALITY")
        print("="*50)
        
        # Basic statistics
        print(f"Total descriptions: {len(df)}")
        print(f"Average text length: {df['text_length'].mean():.1f} characters")
        print(f"Average word count: {df['word_count'].mean():.1f} words")
        print(f"Average sentence count: {df['sentence_count'].mean():.1f} sentences")
        
        # Sentiment distribution
        print(f"\nSentiment Distribution:")
        sentiment_counts = df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count} ({count/len(df)*100:.1f}%)")
        
        # Purpose distribution
        print(f"\nPurpose Distribution:")
        purpose_counts = df['purpose'].value_counts()
        for purpose, count in purpose_counts.items():
            print(f"  {purpose}: {count} ({count/len(df)*100:.1f}%)")
        
        # Sentiment score analysis
        print(f"\nSentiment Score Analysis:")
        print(f"  Mean: {df['sentiment_score'].mean():.3f}")
        print(f"  Std: {df['sentiment_score'].std():.3f}")
        print(f"  Min: {df['sentiment_score'].min():.3f}")
        print(f"  Max: {df['sentiment_score'].max():.3f}")
        
        # Confidence score analysis
        print(f"\nConfidence Score Analysis:")
        print(f"  Mean: {df['sentiment_confidence'].mean():.3f}")
        print(f"  Std: {df['sentiment_confidence'].std():.3f}")
        
        # Text quality indicators
        print(f"\nText Quality Indicators:")
        print(f"  Contains positive words: {df['has_positive_words'].mean()*100:.1f}%")
        print(f"  Contains negative words: {df['has_negative_words'].mean()*100:.1f}%")
        print(f"  Contains financial terms: {df['has_financial_terms'].mean()*100:.1f}%")
        
        # Sample descriptions
        print(f"\nSample Descriptions:")
        for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            sample = df[df['sentiment'] == sentiment].iloc[0]
            print(f"\n  {sentiment} (Score: {sample['sentiment_score']:.3f}):")
            print(f"    {sample['description'][:100]}...")
        
        return True
    
    def save_synthetic_dataset(self, df, filename='synthetic_loan_descriptions.csv'):
        """Save synthetic dataset to file"""
        df.to_csv(filename, index=False)
        print(f"Synthetic dataset saved to {filename}")
        print(f"Dataset shape: {df.shape}")
        return filename

def generate_synthetic_text_dataset():
    """Generate and validate synthetic text dataset"""
    print("GENERATING SYNTHETIC TEXT DATASET")
    print("="*50)
    
    # Initialize generator
    generator = SyntheticTextGenerator()
    
    # Generate dataset
    print("Generating synthetic loan descriptions...")
    df = generator.generate_dataset_with_synthetic_text(
        n_samples=10000,
        sentiment_distribution={'positive': 0.2, 'negative': 0.3, 'neutral': 0.5}
    )
    
    # Validate quality
    generator.validate_synthetic_text(df)
    
    # Save dataset
    filename = generator.save_synthetic_dataset(df)
    
    print(f"\nSynthetic text generation complete!")
    print(f"Dataset saved to: {filename}")
    
    return df

if __name__ == "__main__":
    generate_synthetic_text_dataset() 