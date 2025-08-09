"""
Two-Stage Lending Model Package
==============================

A comprehensive machine learning solution for loan default prediction using
a two-stage modeling approach with sentiment analysis, cost-sensitive learning,
and ensemble methods.

This package provides:
- Two-stage modeling with sentiment analysis
- Class imbalance handling strategies
- Comprehensive evaluation framework
- Complete workflow automation

Author: AI Assistant
Date: 2024
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai.assistant@example.com"

# Import main components for easy access
try:
    from .two_stage_lending_model import TwoStageModel
    from .model_evaluation_workflow import ModelEvaluator
    from .class_imbalance_optimization import (
        xgb_with_class_weights,
        xgb_with_smote,
        xgb_with_combined_sampling,
        optimize_threshold,
        evaluate_imbalanced_model,
        optimize_for_class_imbalance
    )
except ImportError:
    # Handle case when running __init__.py directly
    try:
        from two_stage_lending_model import TwoStageModel
        from model_evaluation_workflow import ModelEvaluator
        from class_imbalance_optimization import (
            xgb_with_class_weights,
            xgb_with_smote,
            xgb_with_combined_sampling,
            optimize_threshold,
            evaluate_imbalanced_model,
            optimize_for_class_imbalance
        )
    except ImportError:
        # If modules not found, set to None
        TwoStageModel = None
        ModelEvaluator = None
        xgb_with_class_weights = None
        xgb_with_smote = None
        xgb_with_combined_sampling = None
        optimize_threshold = None
        evaluate_imbalanced_model = None
        optimize_for_class_imbalance = None

# Define what gets imported with "from package import *"
__all__ = [
    'TwoStageModel',
    'ModelEvaluator',
    'xgb_with_class_weights',
    'xgb_with_smote',
    'xgb_with_combined_sampling',
    'optimize_threshold',
    'evaluate_imbalanced_model',
    'optimize_for_class_imbalance'
]

# Package metadata
PACKAGE_NAME = "two_stage_lending_model"
DESCRIPTION = "Advanced loan default prediction with two-stage modeling and sentiment analysis"
KEYWORDS = ["machine learning", "lending", "default prediction", "sentiment analysis", "ensemble methods"]

# Version info
def get_version():
    """Get the package version."""
    return __version__

def get_info():
    """Get package information."""
    return {
        'name': PACKAGE_NAME,
        'version': __version__,
        'author': __author__,
        'description': DESCRIPTION,
        'keywords': KEYWORDS
    }

# Handle direct execution
if __name__ == "__main__":
    print("Two-Stage Lending Model Package")
    print("="*40)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"Description: {DESCRIPTION}")
    print()
    print("This is a package. To run the application, use:")
    print("  python main.py")
    print()
    print("To import components:")
    print("  from two_stage_lending_model import TwoStageModel")
    print("  from two_stage_lending_model import ModelEvaluator") 