"""
Data Loader for Lending Club Dataset
===================================

This module handles downloading and loading the Lending Club dataset using kagglehub.
It provides a clean interface for accessing the dataset in the two-stage lending model.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import kagglehub
import os
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class LendingClubDataLoader:
    """Data loader for Lending Club dataset using kagglehub"""
    
    def __init__(self, dataset_name: str = "wordsforthewise/lending-club"):
        """
        Initialize the data loader
        
        Args:
            dataset_name: Kaggle dataset name (default: "wordsforthewise/lending-club")
        """
        self.dataset_name = dataset_name
        self.dataset_path = None
        self.data = None
        
    def download_dataset(self, force_download: bool = False) -> str:
        """
        Download the dataset using kagglehub
        
        Args:
            force_download: Force re-download even if already exists
            
        Returns:
            Path to the downloaded dataset
        """
        try:
            logger.info(f"Downloading dataset: {self.dataset_name}")
            
            # Download dataset
            self.dataset_path = kagglehub.dataset_download(self.dataset_name)
            
            logger.info(f"Dataset downloaded to: {self.dataset_path}")
            return self.dataset_path
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise
    
    def load_data(self, file_name: str = "accepted_2007_to_2018Q4.csv") -> pd.DataFrame:
        """
        Load the dataset from the downloaded files
        
        Args:
            file_name: Name of the CSV file to load
            
        Returns:
            Loaded DataFrame
        """
        try:
            # If dataset not downloaded yet, download it
            if self.dataset_path is None:
                self.download_dataset()
            
            # List all files in the dataset directory
            available_files = os.listdir(self.dataset_path)
            logger.info(f"Available files in dataset directory: {available_files}")
            
            # Handle nested directory structure
            # The CSV file might be in a subdirectory with the same name
            possible_paths = [
                # Direct file in dataset directory
                os.path.join(self.dataset_path, file_name),
                # File in subdirectory with same name (nested structure)
                os.path.join(self.dataset_path, file_name, file_name),
                # Alternative case variations
                os.path.join(self.dataset_path, "accepted_2007_to_2018q4.csv"),
                os.path.join(self.dataset_path, "accepted_2007_to_2018q4.csv", "accepted_2007_to_2018q4.csv"),
            ]
            
            csv_path = None
            for path in possible_paths:
                if os.path.exists(path) and os.path.isfile(path):
                    csv_path = path
                    logger.info(f"Found CSV file: {path}")
                    break
            
            if csv_path is None:
                # Search recursively for CSV files
                logger.info("Searching recursively for CSV files...")
                for root, dirs, files in os.walk(self.dataset_path):
                    for file in files:
                        if file.endswith('.csv'):
                            full_path = os.path.join(root, file)
                            if os.path.isfile(full_path):
                                csv_path = full_path
                                logger.info(f"Found CSV file recursively: {full_path}")
                                break
                    if csv_path:
                        break
            
            if csv_path is None:
                logger.error(f"No CSV files found. Available files: {available_files}")
                raise FileNotFoundError(f"No CSV file found in {self.dataset_path}")
            
            logger.info(f"Loading data from: {csv_path}")
            
            # Load the CSV file
            self.data = pd.read_csv(csv_path, low_memory=False)
            
            logger.info(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_data_info(self) -> dict:
        """
        Get information about the loaded dataset
        
        Returns:
            Dictionary with dataset information
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        info = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "memory_usage": self.data.memory_usage(deep=True).sum(),
            "missing_values": self.data.isnull().sum().sum(),
            "dtypes": self.data.dtypes.value_counts().to_dict()
        }
        
        # Add target variable info if available
        if 'loan_status' in self.data.columns:
            info["loan_status_distribution"] = self.data['loan_status'].value_counts().to_dict()
        
        return info
    
    def save_local_copy(self, output_path: str = "accepted_2007_to_2018q4.csv") -> str:
        """
        Save a local copy of the dataset
        
        Args:
            output_path: Path where to save the local copy
            
        Returns:
            Path to the saved file
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        try:
            self.data.to_csv(output_path, index=False)
            logger.info(f"Local copy saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving local copy: {e}")
            raise

def load_lending_club_data(use_local_copy: bool = False, 
                          local_path: str = "accepted_2007_to_2018q4.csv",
                          sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Convenience function to load Lending Club data
    
    Args:
        use_local_copy: Whether to use a local copy instead of downloading
        local_path: Path to local copy if use_local_copy is True
        sample_size: Number of rows to sample (None for full dataset)
        
    Returns:
        Loaded DataFrame
    """
    if use_local_copy and os.path.exists(local_path):
        logger.info(f"Loading local copy from: {local_path}")
        df = pd.read_csv(local_path, low_memory=False)
    else:
        # Use kagglehub to download and load
        loader = LendingClubDataLoader()
        df = loader.load_data()
    
    # Apply sampling if requested
    if sample_size is not None and sample_size < len(df):
        logger.info(f"Sampling {sample_size:,} records from {len(df):,} total records")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    return df

def get_dataset_path() -> str:
    """
    Get the path to the downloaded dataset
    
    Returns:
        Path to the dataset directory
    """
    loader = LendingClubDataLoader()
    return loader.download_dataset()

if __name__ == "__main__":
    # Test the data loader
    print("Testing Lending Club Data Loader")
    print("="*40)
    
    try:
        # Load data
        df = load_lending_club_data()
        
        # Print info
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        if 'loan_status' in df.columns:
            print("\nLoan status distribution:")
            print(df['loan_status'].value_counts())
        
        print("\nData loading test successful!")
        
    except Exception as e:
        print(f"Error: {e}") 