# tests/enhanced/test_preprocessing.py

import sys
import os
import pandas as pd
import numpy as np
import unittest
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.enhanced.preprocessing import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor"""
    
    def setUp(self):
        """Set up test environment"""
        # Create test data directory if it doesn't exist
        self.data_dir = project_root / 'data'
        self.data_dir.mkdir(exist_ok=True)
        
        # Create test data
        self.test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        
        # Save test data
        self.test_file = self.data_dir / 'test_data.csv'
        self.test_data.to_csv(self.test_file, index=False)
        
        # Initialize processor
        self.processor = DataProcessor()
    
    def tearDown(self):
        """Clean up test environment"""
        if self.test_file.exists():
            self.test_file.unlink()
    
    def test_data_loading(self):
        """Test data loading functionality"""
        try:
            dataset, tensor_data, data, scaler, n_features = (
                self.processor.load_and_process_data(str(self.test_file))
            )
            
            print("Testing data loading...")
            self.assertEqual(data.shape, self.test_data.shape)
            self.assertEqual(n_features, 2)
            print("✓ Data loading test passed")
            
        except Exception as e:
            print(f"✗ Data loading test failed: {str(e)}")
            raise
    
    def test_scaling(self):
        """Test data scaling functionality"""
        try:
            print("Testing data scaling...")
            scaled_data = self.processor.scale_data(self.test_data.values)
            
            # Check scaled data properties
            self.assertTrue(abs(scaled_data.mean()) < 1e-10)  # Close to 0
            self.assertEqual(scaled_data.shape, self.test_data.shape)
            print("✓ Scaling test passed")
            
        except Exception as e:
            print(f"✗ Scaling test failed: {str(e)}")
            raise
    
    def test_inverse_transform(self):
        """Test inverse transformation"""
        try:
            print("Testing inverse transform...")
            # Scale data
            scaled_data = self.processor.scale_data(self.test_data.values)
            
            # Inverse transform
            reconstructed_data = self.processor.inverse_transform(scaled_data)
            
            # Check reconstruction
            np.testing.assert_array_almost_equal(
                self.test_data.values, 
                reconstructed_data,
                decimal=10
            )
            print("✓ Inverse transform test passed")
            
        except Exception as e:
            print(f"✗ Inverse transform test failed: {str(e)}")
            raise

def run_tests():
    """Run all tests with detailed output"""
    print("\n=== Running DataProcessor Tests ===\n")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataProcessor)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)