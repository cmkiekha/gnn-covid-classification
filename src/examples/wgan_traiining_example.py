# src/examples/wgan_traIining_example.py
from pathlib import Path
from datetime import datetime
import logging
import os
import yaml
import torch


from src.models.wgan.trainer import WGANGPTrainer
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.visualization import VisualizationTools

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_config(config: dict) -> bool:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Returns:
        bool: True if configuration is valid
    """
    required_keys = ["training", "model", "evaluation"]
    return all(key in config for key in required_keys)


def train_wgan(config_path: str = "config/wgan_config.yaml"):
    """
    Example of WGAN-GP training with evaluation and visualization.

    Args:
        config_path: Path to configuration file

    Returns:
        tuple: (trainer, evaluation_metrics)
    """
    # Load configuration
    if not os.path.exists(config_path):
        logger.error("Configuration file not found at %s", config_path)
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate configuration
    if not validate_config(config):
        logger.error("Invalid configuration: missing required parameters")
        raise ValueError("Invalid configuration")

    # Initialize components
    trainer = WGANGPTrainer(config)
    evaluator = EvaluationMetrics()
    visualizer = VisualizationTools()

    logger.info("Starting training...")

    # Training loop with evaluation
    for epoch in range(config["training"]["epochs"]):
        # Train one epoch
        train_metrics = trainer.train_epoch()

        # Periodic evaluation
        if (epoch + 1) % config["evaluation"]["eval_frequency"] == 0:
            logger.info("Evaluating epoch %d", epoch + 1)

            # Generate samples
            generated_data = trainer.generate_samples(100)

            # Compute evaluation metrics
            eval_metrics = evaluator.evaluate_all(trainer.original_data, generated_data)

            # Visualize progress
            visualizer.plot_training_progress(train_metrics)
            visualizer.plot_quality_metrics(eval_metrics)

    logger.info("Training completed successfully")
    return trainer, eval_metrics


def main():
    """Main execution function."""
    try:
        trainer, metrics = train_wgan()
        logger.info("Training completed successfully")
        return trainer, metrics
    except Exception as e:
        logger.error("Training failed: %s", str(e))
        raise


if __name__ == "__main__":
    main()
