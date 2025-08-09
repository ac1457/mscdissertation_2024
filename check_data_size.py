#!/usr/bin/env python3
"""
Check Data Size and Runtime Estimation
=====================================
"""

import os
import pandas as pd

def check_available_data():
    """Check what data is available and estimate runtime"""
    print("LENDING CLUB DATA SIZE CHECKER")
    print("="*50)
    
    # Check local files
    local_files = [
        "accepted_2007_to_2018q4.csv",
        "loan_sentiment_results.csv", 
        "fast_sentiment_results.csv"
    ]
    
    total_records = 0
    
    for file in local_files:
        if os.path.exists(file):
            print(f"\nFound: {file}")
            
            try:
                if file.endswith('sentiment_results.csv'):
                    df = pd.read_csv(file)
                    print(f"  Sentiment records: {len(df):,}")
                    if 'sentiment' in df.columns:
                        sentiment_counts = df['sentiment'].value_counts()
                        for sentiment, count in sentiment_counts.items():
                            print(f"    {sentiment}: {count}")
                else:
                    # For large CSV, just get info without loading all
                    df_sample = pd.read_csv(file, nrows=1000)
                    
                    # Estimate total rows by file size
                    file_size = os.path.getsize(file)
                    print(f"  File size: {file_size / (1024**3):.2f} GB")
                    
                    # Quick row count (this might take a moment for very large files)
                    print("  Counting rows (this may take a moment)...")
                    row_count = sum(1 for line in open(file)) - 1  # Subtract header
                    print(f"  Total records: {row_count:,}")
                    total_records = row_count
                    
                    print(f"  Columns: {len(df_sample.columns)}")
                    print(f"  Sample columns: {list(df_sample.columns[:10])}")
                    
            except Exception as e:
                print(f"  Error reading {file}: {e}")
        else:
            print(f"\nNot found: {file}")
    
    # Runtime estimates
    print(f"\nRUNTIME ESTIMATES")
    print("="*30)
    
    if total_records > 0:
        print(f"Total records available: {total_records:,}")
        
        # Estimate based on your recent 5k sample (293 seconds)
        base_time = 293  # seconds for 5k samples
        base_samples = 5000
        
        # Estimate scaling (not linear due to model complexity)
        if total_records <= 10000:
            estimated_time = (total_records / base_samples) * base_time
        elif total_records <= 100000:
            estimated_time = (total_records / base_samples) * base_time * 1.2  # 20% overhead
        else:
            estimated_time = (total_records / base_samples) * base_time * 1.5  # 50% overhead
        
        estimated_minutes = estimated_time / 60
        estimated_hours = estimated_minutes / 60
        
        print(f"\nEstimated runtime for full dataset:")
        if estimated_hours >= 2:
            print(f"  {estimated_hours:.1f} hours")
        elif estimated_minutes >= 90:
            print(f"  {estimated_minutes:.0f} minutes ({estimated_hours:.1f} hours)")
        else:
            print(f"  {estimated_minutes:.0f} minutes")
        
        print(f"\nRecommended approach:")
        if total_records > 1000000:
            print(f"  1. Start with: python main.py --samples 50000  (large sample)")
            print(f"  2. If satisfied: python main.py --full-data   (full analysis)")
        elif total_records > 100000:
            print(f"  1. Start with: python main.py --samples 20000  (medium sample)")
            print(f"  2. If satisfied: python main.py --full-data   (full analysis)")
        else:
            print(f"  Go ahead with: python main.py --full-data")
    
    else:
        print("No local data found. Will download from Kaggle when you run analysis.")
        print("Expected download: ~2.26 million records")
        print("Estimated full analysis time: 2-4 hours")
    
    print(f"\nCOMMAND OPTIONS:")
    print("="*20)
    print("python main.py                    # 5k sample (5 min)")
    print("python main.py --samples 20000    # 20k sample (20 min)")
    print("python main.py --samples 100000   # 100k sample (1-2 hours)")
    print("python main.py --full-data        # All data (comprehensive)")

if __name__ == "__main__":
    check_available_data() 