# src/evaluation/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from sklearn.manifold import TSNE
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class VisualizationTools:
    """
    Visualization tools for WGAN-GP evaluation.

    Features:
    - Training progress visualization
    - Quality metrics plotting
    - KS statistics analysis
    - Distribution comparisons
    """

    def __init__(self):
        """Initialize visualization settings"""
        plt.style.use("seaborn")
        self.default_figsize = (15, 10)

    def plot_training_progress(self, train_metrics: Dict[str, List[float]]):
        """
        Plot training progress metrics.

        Args:
            train_metrics: Dictionary containing training metrics
                Required keys: 'generator_loss', 'critic_loss'
                Optional keys: 'gradient_penalty', 'wasserstein_distance'
        """
        if not isinstance(train_metrics, dict):
            logger.error("Training metrics must be a dictionary")
            return

        fig, axes = plt.subplots(2, 2, figsize=self.default_figsize)

        # Generator loss
        if "generator_loss" in train_metrics:
            axes[0, 0].plot(train_metrics["generator_loss"], label="Generator")
            axes[0, 0].set_title("Generator Loss")
            axes[0, 0].set_xlabel("Iteration")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()

        # Critic loss
        if "critic_loss" in train_metrics:
            axes[0, 1].plot(train_metrics["critic_loss"], label="Critic")
            axes[0, 1].set_title("Critic Loss")
            axes[0, 1].set_xlabel("Iteration")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].legend()

        # Additional metrics
        if "gradient_penalty" in train_metrics:
            axes[1, 0].plot(train_metrics["gradient_penalty"], label="Gradient Penalty")
            axes[1, 0].set_title("Gradient Penalty")
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("Penalty")
            axes[1, 0].legend()

        if "wasserstein_distance" in train_metrics:
            axes[1, 1].plot(train_metrics["wasserstein_distance"], label="W-Distance")
            axes[1, 1].set_title("Wasserstein Distance")
            axes[1, 1].set_xlabel("Iteration")
            axes[1, 1].set_ylabel("Distance")
            axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def plot_quality_metrics(self, metrics_dict: Dict):
        """
        Plot comprehensive quality metrics.

        Args:
            metrics_dict: Dictionary containing quality metrics
                Possible keys: 'feature_correlations', 'density_overlap',
                             'statistical', 'nearest_neighbor_ratio'
        """
        if not metrics_dict:
            logger.warning("No metrics provided for plotting")
            return

        fig = plt.figure(figsize=self.default_figsize)

        # Feature correlations
        ax1 = fig.add_subplot(221)
        if "feature_correlations" in metrics_dict:
            corr_data = metrics_dict["feature_correlations"]
            if isinstance(corr_data, (float, int)):
                ax1.bar(["Correlation Score"], [corr_data])
            else:
                sns.heatmap(corr_data, ax=ax1, cmap="coolwarm", center=0)
            ax1.set_title("Feature Correlations")

        # Density overlap
        ax2 = fig.add_subplot(222)
        if "density_overlap" in metrics_dict:
            ax2.bar(["Density Overlap"], [metrics_dict["density_overlap"]])
            ax2.set_title("Density Overlap Score")
            ax2.set_ylim(0, 1)

        # Statistical metrics
        ax3 = fig.add_subplot(223)
        if "statistical" in metrics_dict:
            stats_data = pd.Series(metrics_dict["statistical"])
            stats_data.plot(kind="bar", ax=ax3)
            ax3.set_title("Statistical Metrics")
            ax3.tick_params(axis="x", rotation=45)

        # NN ratio
        ax4 = fig.add_subplot(224)
        if "nearest_neighbor_ratio" in metrics_dict:
            ax4.bar(["NN Ratio"], [metrics_dict["nearest_neighbor_ratio"]])
            ax4.set_title("Nearest Neighbor Ratio")

        plt.tight_layout()
        plt.show()

    def plot_ks_analysis(self, original: pd.DataFrame, generated: pd.DataFrame):
        """
        Plot detailed KS analysis.

        Args:
            original: Original data DataFrame
            generated: Generated data DataFrame
        """
        if not isinstance(original, pd.DataFrame) or not isinstance(
            generated, pd.DataFrame
        ):
            logger.error("Both inputs must be pandas DataFrames")
            return

        if len(original) == 0 or len(generated) == 0:
            logger.error("Empty data provided")
            return

        if not all(col in generated.columns for col in original.columns):
            logger.error("Column mismatch between original and generated data")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # KS statistics distribution
        ks_stats = []
        p_values = []
        for col in original.columns:
            stat, pvalue = stats.ks_2samp(original[col], generated[col])
            ks_stats.append(stat)
            p_values.append(pvalue)

        # Plot 1: KS Statistics Distribution
        sns.histplot(ks_stats, kde=True, ax=axes[0, 0])
        axes[0, 0].set_title("Distribution of KS Statistics")
        axes[0, 0].set_xlabel("KS Statistic")
        axes[0, 0].set_ylabel("Count")

        # Plot 2: QQ plots for top features
        top_features = original.var().nlargest(3).index
        for feature in top_features:
            stats.probplot(original[feature], dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title(f"Q-Q Plot: {feature}")

        # Plot 3: Feature-wise KS statistics
        feature_ks = pd.Series(ks_stats, index=original.columns)
        feature_ks.nlargest(10).plot(kind="bar", ax=axes[1, 0])
        axes[1, 0].set_title("Top 10 Features by KS Statistic")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Plot 4: P-value distribution
        sns.histplot(p_values, kde=True, ax=axes[1, 1])
        axes[1, 1].set_title("Distribution of P-values")
        axes[1, 1].set_xlabel("P-value")
        axes[1, 1].set_ylabel("Count")

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        logger.info("KS Analysis Summary:")
        logger.info("Mean KS statistic: %.4f", np.mean(ks_stats))
        logger.info("Mean p-value: %.4f", np.mean(p_values))
        logger.info("Features with p-value > 0.05: %d", sum(np.array(p_values) > 0.05))
