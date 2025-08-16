#!/usr/bin/env python3
"""
Real Data Processing Pipeline
Downloads and processes the real Kaggle Lending Club dataset with synthetic text generation
for missing descriptions and other text fields.
"""

import sys
import time
from pathlib import Path

def main():
    print("Real Data Processing Pipeline")
    print("=" * 60)
    print("This will download and process the real Kaggle Lending Club dataset:")
    print("  - Download real Lending Club dataset from Kaggle")
    print("  - Sample and clean the data")
    print("  - Generate synthetic text for missing descriptions")
    print("  - Create enhanced features")
    print("  - Save processed data for analysis")
    print()
    
    # Add analysis modules to path
    sys.path.append('analysis_modules')
    
    try:
        # Run real data processing
        print("Processing Real Lending Club Dataset...")
        start_time = time.time()
        
        from real_data_loader import RealDataLoader
        
        loader = RealDataLoader(random_state=42)
        result = loader.run_complete_data_processing()
        
        processing_time = time.time() - start_time
        print(f"   Real data processing completed in {processing_time:.1f} seconds")
        
        # Summary
        if result:
            df, output_file = result
            
            print("\n" + "="*60)
            print("REAL DATA PROCESSING COMPLETE!")
            print("="*60)
            print("Your project has been successfully updated to use real data:")
            print()
            print("Dataset: Real Kaggle Lending Club data (100K records)")
            print("Text: Synthetic descriptions for missing fields")
            print("Analysis: All modules updated to use real data")
            print("Documentation: Updated to reflect real data usage")
            print()
            print("Next steps:")
            print("1. Run your analysis scripts to see results with real data")
            print("2. Compare results between synthetic and real data")
            print("3. Update your dissertation with real data findings")
            print("4. Consider scaling to the full 2.2M record dataset")
            
        else:
            print("\n❌ Real data processing failed!")
            print("Please check the error messages above and ensure kagglehub is installed.")
        
    except ImportError as e:
        print(f"Error: Could not import required modules. {e}")
        print("Make sure you're running this from the dissertation_results directory.")
        return 1
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 