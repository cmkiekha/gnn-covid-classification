# tests/enhanced/run_all_tests.py

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.enhanced.preprocessing import DataProcessor
import numpy as np
import pandas as pd
import torch

class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor implementation"""

    def setUp(self):
        """Set up test environment"""
        self.data_dir = project_root / 'data'
        self.data_dir.mkdir(exist_ok=True)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        
        # Save sample data
        self.test_file = self.data_dir / 'test_data.csv'
        self.sample_data.to_csv(self.test_file, index=False)
        
        # Initialize processor
        self.processor = DataProcessor(scaler_type='robust')

    def tearDown(self):
        """Clean up test environment"""
        if self.test_file.exists():
            self.test_file.unlink()

    def test_data_loading(self):
        """Test data loading functionality"""
        dataset, tensor_data, data, scaler, n_features = (
            self.processor.load_and_process_data(str(self.test_file))
        )
        
        self.assertEqual(data.shape, self.sample_data.shape)
        self.assertEqual(n_features, 3)
        self.assertTrue(isinstance(tensor_data, torch.Tensor))
        self.assertEqual(tensor_data.shape[1], 3)

    def test_scaling(self):
        """Test data scaling"""
        # Test scaling with new data
        scaled_data = self.processor.scale_data(self.sample_data.values)
        self.assertEqual(scaled_data.shape, self.sample_data.shape)
        
        # Check scaling properties
        mean_close_to_zero = np.abs(scaled_data.mean()) < 1.0
        self.assertTrue(mean_close_to_zero)

    def test_inverse_transform(self):
        """Test inverse transformation"""
        # Scale data
        scaled_data = self.processor.scale_data(self.sample_data.values)
        
        # Inverse transform
        reconstructed_data = self.processor.inverse_transform(scaled_data)
        
        # Check reconstruction
        np.testing.assert_array_almost_equal(
            self.sample_data.values, 
            reconstructed_data,
            decimal=10
        )

def run_test_suite():
    """Run all tests with detailed output"""
    print("\n=== Running Full Test Suite ===\n")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataProcessor)
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_test_suite()
    sys.exit(0 if success else 1)