# src/train.py

import yaml
import logging
import torch
from pathlib import Path
from datetime import datetime

# Local imports
from src.models.data_augmentation_v2.GAN import WGANGPTrainer
from src.utils.enhanced.preprocessing import DataProcessor
from src.validation.metrics_tracking import MetricsTracker

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        with open('config/enhanced_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize components
        logger.info("Initializing components...")
        processor = DataProcessor()
        
        # Load and process data
        logger.info("Loading and processing data...")
        data = processor.load_and_process_data(config['data']['input_path'])
        
        # Initialize trainer
        trainer = WGANGPTrainer(config)
        
        # Train model
        logger.info("Starting training...")
        results = trainer.train_with_kfold(data)
        
        # Save results
        logger.info("Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path('results/enhanced/models')
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(results, save_path / f'training_results_{timestamp}.pkl')
        
        logger.info("Training complete!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()