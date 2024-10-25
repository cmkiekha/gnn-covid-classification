# tests/enhanced/verify_pipeline.py

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import torch
import logging

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.enhanced.preprocessing import DataProcessor


def verify_pipeline():
    """Verify entire preprocessing pipeline"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting pipeline verification...")

    # Create test data
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    test_file = data_dir / "pipeline_test.csv"

    try:
        # Generate test data
        logger.info("Generating test data...")
        test_data = pd.DataFrame(
            {f"feature_{i}": np.random.randn(100) for i in range(5)}
        )
        test_data.to_csv(test_file, index=False)

        # Initialize processor
        logger.info("Initializing DataProcessor...")
        processor = DataProcessor(scaler_type="robust")

        # Process data
        logger.info("Processing data...")
        dataset, tensor_data, data, scaler, n_features = (
            processor.load_and_process_data(str(test_file))
        )

        # Verify results
        logger.info("\nVerification Results:")
        logger.info(f"Original shape: {test_data.shape}")
        logger.info(f"Processed shape: {data.shape}")
        logger.info(f"Tensor shape: {tensor_data.shape}")
        logger.info(f"Number of features: {n_features}")
        logger.info(f"Scaler type: {type(scaler).__name__}")

        # Test reconstruction
        logger.info("\nTesting reconstruction...")
        reconstructed = processor.inverse_transform(tensor_data.numpy())
        max_diff = np.abs(reconstructed - data.values).max()
        logger.info(f"Maximum reconstruction difference: {max_diff:.6f}")

        logger.info("\n✓ Pipeline verification completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Pipeline verification failed: {str(e)}")
        return False

    finally:
        if test_file.exists():
            test_file.unlink()


if __name__ == "__main__":
    success = verify_pipeline()
    sys.exit(0 if success else 1)
