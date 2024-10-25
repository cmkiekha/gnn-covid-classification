# tests/enhanced/quick_test_preprocessing.py

import sys
import os
from pathlib import Path

# Add the project root directory to Python path correctly
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import torch
from src.utils.enhanced.preprocessing import DataProcessor

def run_quick_test():
    """
    Quick test to verify DataProcessor functionality.
    """
    print("\n=== Quick Test: DataProcessor ===")
    
    # Create test data directory if it doesn't exist
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Create test data
    print("Creating test data...")
    test_data = pd.DataFrame({
        'feature1': np.random.randn(10),
        'feature2': np.random.randn(10)
    })
    test_file = data_dir / "quick_test_data.csv"
    test_data.to_csv(test_file, index=False)
    print("✓ Test data created")
    
    # Test processor
    print("\nTesting DataProcessor...")
    processor = DataProcessor()
    
    try:
        # Verify the processor has the required method
        if not hasattr(processor, 'load_and_process_data'):
            print("✗ Error: DataProcessor missing 'load_and_process_data' method")
            print(f"Available methods: {dir(processor)}")
            raise AttributeError("DataProcessor missing required method")
            
        # Test data loading and processing
        dataset, tensor_data, data, scaler, n_features = processor.load_and_process_data(str(test_file))
        
        # Print detailed results
        print("\nTest Results:")
        print(f"✓ Data loaded successfully")
        print(f"  - Original shape: {test_data.shape}")
        print(f"  - Processed shape: {data.shape}")
        print(f"  - Tensor shape: {tensor_data.shape}")
        print(f"  - Number of features: {n_features}")
        print(f"  - Scaler type: {type(scaler).__name__}")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        print("\nDebug Information:")
        print(f"  - Current directory: {os.getcwd()}")
        print(f"  - Project root: {project_root}")
        print(f"  - Test file path: {test_file}")
        
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()
            print("\n✓ Test file cleaned up")
            
    print("\n=== Test Complete ===\n")

if __name__ == "__main__":
    # Verify imports
    print("Verifying imports...")
    print(f"DataProcessor location: {DataProcessor.__module__}")
    
    # Run tests
    run_quick_test()