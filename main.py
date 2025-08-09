#!/usr/bin/env python3
"""
Lending Club Sentiment Analysis - Main Execution Script
======================================================
Real sentiment analysis for credit risk modeling dissertation using authentic data
"""

import logging
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_real_analysis(sample_size=5000):
    """Run analysis with real Lending Club data and authentic sentiment"""
    logger.info(f"Running Real Data Analysis with {sample_size} samples...")
    logger.info("Using authentic Lending Club data with real sentiment analysis")
    
    try:
        from real_lending_analysis import RealLendingAnalysis
        study = RealLendingAnalysis(sample_size=sample_size)
        results, stats, success = study.run_real_analysis()
        
        if success:
            logger.info("Real data analysis completed successfully!")
            logger.info("Authentic results based on actual Lending Club data!")
            return True
        else:
            logger.error("Real data analysis failed")
            return False
            
    except Exception as e:
        logger.error(f"Real data analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_streamlined_workflow(sample_size=3000):
    """Run streamlined workflow for quick testing with real data"""
    logger.info(f"Running Streamlined Workflow with {sample_size} samples...")
    logger.info("Quick analysis using real data for testing and validation")
    
    try:
        from streamlined_workflow import StreamlinedWorkflow
        workflow = StreamlinedWorkflow(sample_size=sample_size)
        results = workflow.run_streamlined_workflow()
        
        if results:
            logger.info("Streamlined workflow completed successfully!")
            return True
        else:
            logger.error("Streamlined workflow failed")
            return False
            
    except Exception as e:
        logger.error(f"Streamlined workflow failed: {e}")
        return False

def analyze_existing_sentiment():
    """Analyze the real sentiment data files"""
    logger.info("Analyzing existing real sentiment data...")
    
    try:
        from analyze_real_sentiment import main as analyze_main
        analyze_main()
        logger.info("Sentiment data analysis completed!")
        return True
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return False

def run_synthetic_analysis(sample_size=15000):
    """Run synthetic analysis (for comparison/demonstration only)"""
    logger.info(f"Running Synthetic Analysis with {sample_size} samples...")
    logger.info("WARNING: This uses artificial data for demonstration purposes only")
    
    try:
        from optimized_final_analysis import OptimizedFinalAnalysis
        study = OptimizedFinalAnalysis(sample_size=sample_size)
        results, stats, success = study.run_optimized_analysis()
        
        if success:
            logger.info("Synthetic analysis completed!")
            logger.warning("REMINDER: These results are based on artificial data")
            return True
        else:
            logger.error("Synthetic analysis failed")
            return False
            
    except Exception as e:
        logger.error(f"Synthetic analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Lending Club Sentiment Analysis for Credit Risk Modeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run real data analysis (5k samples)
  python main.py --full-data               # Run with ALL data (comprehensive)
  python main.py --samples 0               # Alternative way to use all data
  python main.py --samples 10000           # Run with 10k samples  
  python main.py --quick                   # Run quick streamlined version
  python main.py --analyze-sentiment       # Analyze existing sentiment data
  python main.py --synthetic               # Run synthetic analysis (artificial data)
        """
    )
    
    parser.add_argument('--samples', type=int, default=5000,
                       help='Number of samples to analyze (default: 5000, use 0 for all data)')
    
    parser.add_argument('--full-data', action='store_true',
                       help='Use all available data (may take 30-60 minutes)')
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick streamlined workflow for testing')
    
    parser.add_argument('--analyze-sentiment', action='store_true',
                       help='Analyze existing real sentiment data files')
    
    parser.add_argument('--synthetic', action='store_true',
                       help='Run synthetic analysis (artificial data - for demonstration only)')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 70)
    print("LENDING CLUB SENTIMENT ANALYSIS DISSERTATION STUDY")
    print("=" * 70)
    print("Real Data | Authentic Results | Academic Integrity")
    print()
    
    start_time = datetime.now()
    
    # Determine sample size
    if args.full_data:
        sample_size = None  # Use all data
        print("FULL DATASET ANALYSIS - Using all available data")
        print("Expected runtime: 30-60 minutes")
        print()
    elif args.samples == 0:
        sample_size = None  # Use all data
        print("FULL DATASET ANALYSIS - Using all available data")
        print("Expected runtime: 30-60 minutes")
        print()
    else:
        sample_size = args.samples
    
    # Run analysis based on arguments
    if args.analyze_sentiment:
        success = analyze_existing_sentiment()
    elif args.quick:
        success = run_streamlined_workflow(sample_size or 3000)
    elif args.synthetic:
        print("WARNING: Synthetic analysis uses artificial data!")
        print("For authentic dissertation results, use real data analysis")
        print()
        success = run_synthetic_analysis(sample_size or 15000)
    else:
        # Default: Run real data analysis
        success = run_real_analysis(sample_size)
    
    # Print conclusion
    end_time = datetime.now()
    runtime = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    if success:
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        if args.synthetic:
            print("NOTE: Results based on synthetic/artificial data")
            print("For dissertation, use real data analysis (default)")
        else:
            print("Results based on authentic Lending Club data")
            print("These are real-world findings suitable for academic work")
    else:
        print("ANALYSIS FAILED")
        print("Please check the error messages above")
    
    print(f"Total runtime: {runtime:.1f} seconds")
    print("=" * 70)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 
    
    