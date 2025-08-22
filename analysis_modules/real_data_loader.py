"""
Real Data Loader for Lending Club Dataset
Downloads the real Kaggle Lending Club dataset and generates synthetic text
for missing descriptions and other text fields.
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    print("kagglehub not available. Install with: pip install kagglehub")
    KAGGLEHUB_AVAILABLE = False

class RealDataLoader:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Configuration
        self.config = {
            'dataset_name': 'wordsforthewise/lending-club',
            'output_dir': 'data/real_lending_club',
            'sample_size': 100000,  # Use 100K records for computational efficiency
            'text_generation': True
        }
        
        # Loan purposes for text generation
        self.loan_purposes = [
            'debt_consolidation', 'credit_card', 'home_improvement', 
            'major_purchase', 'small_business', 'car', 'medical', 
            'vacation', 'moving', 'house', 'wedding', 'renewable_energy',
            'educational', 'other'
        ]
        
        # Text templates for different purposes
        self.text_templates = {
            'debt_consolidation': [
                "Looking to consolidate multiple credit card debts into one manageable payment.",
                "Need to consolidate high-interest debts for better financial management.",
                "Want to combine several loans into one with lower interest rate.",
                "Seeking debt consolidation to simplify my monthly payments.",
                "Need to consolidate debts to improve my credit score."
            ],
            'credit_card': [
                "Need to pay off high-interest credit card balances.",
                "Looking to consolidate credit card debt with better terms.",
                "Want to transfer credit card balances to a lower rate.",
                "Need funds to pay off multiple credit cards.",
                "Seeking loan to eliminate credit card debt."
            ],
            'home_improvement': [
                "Planning home renovation and improvement projects.",
                "Need funds for kitchen and bathroom upgrades.",
                "Looking to improve home value through renovations.",
                "Want to make energy-efficient home improvements.",
                "Need loan for major home repair work."
            ],
            'major_purchase': [
                "Planning to make a significant purchase for home or family.",
                "Need funds for a major appliance or furniture purchase.",
                "Looking to buy expensive equipment or tools.",
                "Want to make a large purchase for personal use.",
                "Need loan for a substantial investment."
            ],
            'small_business': [
                "Need capital for small business expansion and growth.",
                "Looking to invest in business equipment and supplies.",
                "Want to fund business operations and working capital.",
                "Need loan for business development and marketing.",
                "Seeking funds for business startup costs."
            ],
            'car': [
                "Need to purchase a reliable vehicle for transportation.",
                "Looking to buy a car for work and family needs.",
                "Want to finance a vehicle purchase with better terms.",
                "Need funds for car down payment and related costs.",
                "Seeking loan for automotive purchase."
            ],
            'medical': [
                "Need funds for medical procedures and healthcare costs.",
                "Looking to pay for medical bills and treatment.",
                "Want to finance healthcare expenses and procedures.",
                "Need loan for medical emergency costs.",
                "Seeking funds for dental work and medical care."
            ],
            'vacation': [
                "Planning a family vacation and need travel funds.",
                "Looking to finance a dream vacation experience.",
                "Want to take a trip and need travel expenses covered.",
                "Need funds for vacation planning and booking.",
                "Seeking loan for travel and leisure activities."
            ],
            'moving': [
                "Need funds for relocation and moving expenses.",
                "Looking to finance a move to a new location.",
                "Want to cover moving costs and relocation fees.",
                "Need loan for moving and settling in new place.",
                "Seeking funds for relocation and moving services."
            ],
            'house': [
                "Need funds for home purchase and related costs.",
                "Looking to finance home buying expenses.",
                "Want to cover down payment and closing costs.",
                "Need loan for home purchase and moving.",
                "Seeking funds for real estate investment."
            ],
            'wedding': [
                "Planning wedding and need funds for ceremony and reception.",
                "Looking to finance wedding expenses and celebration.",
                "Want to cover wedding costs and related events.",
                "Need loan for wedding planning and execution.",
                "Seeking funds for wedding and celebration costs."
            ],
            'renewable_energy': [
                "Want to install solar panels and renewable energy systems.",
                "Looking to finance green energy improvements.",
                "Need funds for renewable energy installation.",
                "Want to invest in sustainable energy solutions.",
                "Seeking loan for environmental improvements."
            ],
            'educational': [
                "Need funds for education and tuition expenses.",
                "Looking to finance educational programs and courses.",
                "Want to pay for school fees and educational costs.",
                "Need loan for academic pursuits and learning.",
                "Seeking funds for educational advancement."
            ],
            'other': [
                "Need funds for personal expenses and miscellaneous costs.",
                "Looking to finance various personal needs and projects.",
                "Want to cover unexpected expenses and costs.",
                "Need loan for personal financial needs.",
                "Seeking funds for general personal use."
            ]
        }

    def download_real_dataset(self):
        """Download the real Kaggle Lending Club dataset"""
        print("Downloading real Kaggle Lending Club dataset...")
        
        if not KAGGLEHUB_AVAILABLE:
            print("kagglehub not available. Please install with: pip install kagglehub")
            return None
        
        try:
            # Download the dataset
            path = kagglehub.dataset_download(self.config['dataset_name'])
            print(f"Dataset downloaded to: {path}")
            
            # List downloaded files
            dataset_files = list(Path(path).glob("*")) # List all items in the downloaded directory
            print(f"Found {len(dataset_files)} items:")
            for item in dataset_files:
                print(f"   - {item.name} ({item.stat().st_size / 1024**2:.1f} MB)")
            
            return path, dataset_files
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None

    def load_and_process_real_data(self, dataset_path, files):
        """Load and process the real Lending Club data"""
        print("Loading and processing real Lending Club data...")
        
        # Find the main loan data file - handle both files and directories
        main_file = None
        csv_files = []
        
        for item in files:
            if item.is_file() and item.suffix == '.csv':
                csv_files.append(item)
            elif item.is_dir():
                # Look inside directories for CSV files
                sub_files = list(item.glob("*.csv"))
                csv_files.extend(sub_files)
        
        if not csv_files:
            print("No CSV files found in the downloaded dataset")
            return None
        
        print(f"Found {len(csv_files)} CSV files:")
        for file in csv_files:
            print(f"   - {file.name} ({file.stat().st_size / 1024**2:.1f} MB)")
        
        # Prefer accepted loans over rejected loans
        accepted_file = None
        rejected_file = None
        
        for file in csv_files:
            if 'accepted' in file.name.lower():
                accepted_file = file
            elif 'rejected' in file.name.lower():
                rejected_file = file
        
        if accepted_file:
            main_file = accepted_file
            print(f"Using accepted loans file: {main_file.name}")
        elif rejected_file:
            main_file = rejected_file
            print(f"Using rejected loans file: {main_file.name}")
        else:
            # Use the largest CSV file
            main_file = max(csv_files, key=lambda x: x.stat().st_size)
            print(f"Using largest file: {main_file.name}")
        
        try:
            # Load the data
            print(f"Loading {main_file.name}...")
            df = pd.read_csv(main_file)
            print(f"Loaded {len(df)} records with {len(df.columns)} columns")
            
            # Display basic info
            print(f"Dataset shape: {df.shape}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Show column names
            print(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def sample_and_clean_data(self, df):
        """Sample and clean the real data"""
        print("Sampling and cleaning real data...")
        
        # Sample data for computational efficiency
        if len(df) > self.config['sample_size']:
            print(f"Sampling {self.config['sample_size']} records from {len(df)} total records")
            df_sample = df.sample(n=self.config['sample_size'], random_state=self.random_state).reset_index(drop=True)
        else:
            df_sample = df.copy()
        
        # Basic cleaning
        print("Performing basic data cleaning...")
        
        # Remove rows with too many missing values
        threshold = len(df_sample.columns) * 0.5
        df_clean = df_sample.dropna(thresh=threshold)
        print(f"   Removed {len(df_sample) - len(df_clean)} rows with too many missing values")
        
        # Handle missing values
        for col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                print(f"   {col}: {missing_count} missing values")
                
                if df_clean[col].dtype in ['object', 'category']:
                    # Fill categorical with mode
                    mode_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                    df_clean[col] = df_clean[col].fillna(mode_value)
                else:
                    # Fill numerical with median
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        print(f"Cleaned dataset: {len(df_clean)} records, {len(df_clean.columns)} columns")
        
        return df_clean

    def generate_synthetic_text(self, df):
        """Generate synthetic text for missing descriptions and text fields"""
        print("Generating synthetic text for missing fields...")
        
        # Check if description column exists (could be 'desc' or 'description')
        description_cols = [col for col in df.columns if 'desc' in col.lower() or 'text' in col.lower()]
        
        if not description_cols:
            print("No description columns found, creating synthetic descriptions...")
            df['description'] = self.generate_descriptions_from_purpose(df)
        else:
            print(f"Found description columns: {description_cols}")
            # Fill missing descriptions
            for col in description_cols:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    print(f"   Generating {missing_count} missing descriptions for {col}")
                    df[col] = df[col].fillna(df.apply(lambda row: self.generate_description_from_purpose(row), axis=1))
            
            # Ensure we have a 'description' column for consistency
            if 'desc' in df.columns and 'description' not in df.columns:
                df['description'] = df['desc']
            elif 'description' not in df.columns:
                df['description'] = self.generate_descriptions_from_purpose(df)
        
        # Generate synthetic text for other missing text fields
        text_cols = [col for col in df.columns if df[col].dtype == 'object' and col not in description_cols and col != 'description']
        
        for col in text_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                print(f"   Generating {missing_count} missing values for {col}")
                df[col] = df[col].fillna("Generated text for " + col)
        
        return df

    def generate_descriptions_from_purpose(self, df):
        """Generate descriptions based on loan purpose"""
        descriptions = []
        
        for _, row in df.iterrows():
            purpose = self.get_loan_purpose(row)
            description = self.generate_description_from_purpose(row)
            descriptions.append(description)
        
        return descriptions

    def generate_description_from_purpose(self, row):
        """Generate a single description based on row data"""
        purpose = self.get_loan_purpose(row)
        
        # Get template for this purpose
        templates = self.text_templates.get(purpose, self.text_templates['other'])
        
        # Select random template
        template = np.random.choice(templates)
        
        # Add some variation
        variations = [
            "I have stable employment and good credit history.",
            "I have considered this decision carefully and can afford the payments.",
            "This will help me achieve my financial goals.",
            "I have a solid repayment plan in place.",
            "This loan will improve my financial situation."
        ]
        
        variation = np.random.choice(variations)
        
        return f"{template} {variation}"

    def get_loan_purpose(self, row):
        """Extract loan purpose from row data"""
        # Look for purpose-related columns
        purpose_cols = [col for col in row.index if 'purpose' in col.lower()]
        
        if purpose_cols:
            for col in purpose_cols:
                if pd.notna(row[col]) and row[col] in self.loan_purposes:
                    return row[col]
        
        # Default to 'other' if no purpose found
        return 'other'

    def create_enhanced_features(self, df):
        """Create enhanced features for the real data"""
        print("Creating enhanced features for real data...")
        
        # Add text-based features
        if 'description' in df.columns:
            df['text_length'] = df['description'].str.len()
            df['word_count'] = df['description'].str.split().str.len()
            df['sentence_count'] = df['description'].str.count(r'[.!?]+')
            df['avg_word_length'] = df['description'].str.split().apply(
                lambda x: np.mean([len(word) for word in x]) if x else 0
            )
        
        # Add sentiment features (simulated for now)
        df['sentiment_score'] = np.random.uniform(-1, 1, len(df))
        df['sentiment_confidence'] = np.random.uniform(0.5, 1.0, len(df))
        
        # Add financial keyword features
        financial_keywords = ['loan', 'debt', 'credit', 'money', 'payment', 'interest', 'bank', 'financial']
        df['financial_keyword_count'] = df['description'].apply(
            lambda x: sum(1 for word in str(x).lower().split() if word in financial_keywords)
        )
        
        # Add positive/negative word features
        positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'improve', 'help', 'support']
        negative_words = ['bad', 'poor', 'negative', 'problem', 'issue', 'difficult', 'struggle', 'debt']
        
        df['positive_word_count'] = df['description'].apply(
            lambda x: sum(1 for word in str(x).lower().split() if word in positive_words)
        )
        df['negative_word_count'] = df['description'].apply(
            lambda x: sum(1 for word in str(x).lower().split() if word in negative_words)
        )
        
        df['sentiment_balance'] = df['positive_word_count'] - df['negative_word_count']
        
        # Add binary indicators
        df['has_positive_words'] = (df['positive_word_count'] > 0).astype(int)
        df['has_negative_words'] = (df['negative_word_count'] > 0).astype(int)
        df['has_financial_terms'] = (df['financial_keyword_count'] > 0).astype(int)
        
        print(f"Enhanced features created. Final shape: {df.shape}")
        
        return df

    def save_processed_data(self, df):
        """Save the processed real data"""
        print("Saving processed real data...")
        
        # Create output directory
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the processed data
        output_file = output_dir / 'real_lending_club_processed.csv'
        df.to_csv(output_file, index=False)
        
        # Save metadata
        metadata = {
            'dataset_name': self.config['dataset_name'],
            'original_records': len(df),
            'processed_records': len(df),
            'columns': list(df.columns),
            'processing_date': datetime.now().isoformat(),
            'sample_size': self.config['sample_size'],
            'text_generation': self.config['text_generation']
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Processed data saved to: {output_file}")
        print(f"Dataset summary:")
        print(f"   - Records: {len(df):,}")
        print(f"   - Columns: {len(df.columns)}")
        print(f"   - File size: {output_file.stat().st_size / 1024**2:.2f} MB")
        
        return output_file

    def run_complete_data_processing(self):
        """Run complete data processing pipeline"""
        print("=" * 60)
        print("REAL LENDING CLUB DATASET PROCESSING")
        print("=" * 60)
        
        # Step 1: Download dataset
        download_result = self.download_real_dataset()
        if download_result is None:
            return None
        
        dataset_path, files = download_result
        
        # Step 2: Load and process data
        df = self.load_and_process_real_data(dataset_path, files)
        if df is None:
            return None
        
        # Step 3: Sample and clean data
        df_clean = self.sample_and_clean_data(df)
        
        # Step 4: Generate synthetic text
        if self.config['text_generation']:
            df_enhanced = self.generate_synthetic_text(df_clean)
        else:
            df_enhanced = df_clean
        
        # Step 5: Create enhanced features
        df_final = self.create_enhanced_features(df_enhanced)
        
        # Step 6: Save processed data
        output_file = self.save_processed_data(df_final)
        
        print("\n" + "=" * 60)
        print("✅ REAL DATA PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Output file: {output_file}")
        print(f"Final dataset: {len(df_final):,} records, {len(df_final.columns)} columns")
        print(f"Ready for analysis with real Lending Club data!")
        
        return df_final, output_file

if __name__ == "__main__":
    # Run the complete data processing
    loader = RealDataLoader(random_state=42)
    result = loader.run_complete_data_processing()
    
    if result:
        df, output_file = result
        print(f"\nSuccessfully processed real Lending Club data!")
        print(f"Data saved to: {output_file}")
        print(f"Dataset ready for analysis: {len(df):,} records")
    else:
        print(f"\nFailed to process real data. Please check the error messages above.") 