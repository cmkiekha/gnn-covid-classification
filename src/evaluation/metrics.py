# src/evaluation/metrics.py
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import ks_2samp
from typing import Dict, List, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import gaussian_kde


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for WGAN-GP generated samples.
    """

    def evaluate_all(
        self, original_data: np.ndarray, generated_data: np.ndarray
    ) -> Dict:
        """
        Comprehensive evaluation of generated samples.

        Args:
            original_data: Original control samples
            generated_data: Generated synthetic samples

        Returns:
            Dictionary containing all evaluation metrics
        """
        return {
            "basic_metrics": {
                "ks_stats": self.compute_ks_statistics(original_data, generated_data),
                "mmd_score": self.compute_mmd(original_data, generated_data),
                "correlation": self.compute_correlation_preservation(
                    original_data, generated_data
                ),
            },
            "statistical": self.compute_statistical_metrics(
                original_data, generated_data
            ),
            "quality": self.compute_quality_metrics(original_data, generated_data),
        }

    @staticmethod
    def compute_ks_statistics(
        original: np.ndarray,
        generated: np.ndarray
    ) -> pd.DataFrame:
        ks_results = {"Column": [], "KS Statistic": [], "P-Value": []}

        for column in original.columns:
            statistic, pvalue = ks_2samp(original[column], generated[column])
            ks_results["Column"].append(column)
            ks_results["KS Statistic"].append(statistic)
            ks_results["P-Value"].append(pvalue)

        ks_df = pd.DataFrame(ks_results)
        return ks_df

    @staticmethod
    def compute_mmd(x: np.ndarray, y: np.ndarray, kernel="rbf") -> float:
        """Previous implementation remains"""
        pass  # Your existing implementation

    @staticmethod
    def compute_correlation_preservation(
        original: pd.DataFrame, generated: pd.DataFrame
    ) -> float:
        """Previous implementation remains"""
        pass  # Your existing implementation

    def compute_statistical_metrics(
        self, original_data: np.ndarray, generated_data: np.ndarray
    ) -> Dict:
        """
        Compute comprehensive statistical metrics.

        Args:
            original_data: Original samples
            generated_data: Generated samples

        Returns:
            Dictionary of statistical metrics
        """
        return {
            "mean_difference": np.mean(
                np.abs(np.mean(original_data, axis=0) - np.mean(generated_data, axis=0))
            ),
            "std_difference": np.mean(
                np.abs(np.std(original_data, axis=0) - np.std(generated_data, axis=0))
            ),
            "skewness_difference": np.mean(
                np.abs(
                    stats.skew(original_data, axis=0)
                    - stats.skew(generated_data, axis=0)
                )
            ),
            "kurtosis_difference": np.mean(
                np.abs(
                    stats.kurtosis(original_data, axis=0)
                    - stats.kurtosis(generated_data, axis=0)
                )
            ),
        }

    # Add to src/evaluation/metrics.py

    def compute_wasserstein_distance(
        self, original: np.ndarray, generated: np.ndarray
    ) -> float:
        """Compute Wasserstein distance."""
        return stats.wasserstein_distance(original.flatten(), generated.flatten())

    def compute_feature_importance(
        self, original: np.ndarray, generated: np.ndarray
    ) -> Dict:
        """Compute feature importance scores."""
        # Prepare binary classification dataset
        X = np.vstack([original, generated])
        y = np.array([1] * len(original) + [0] * len(generated))

        # Train classifier
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X, y)

        return {
            "feature_importance": clf.feature_importances_,
            "top_features": np.argsort(clf.feature_importances_)[-10:],
        }

    def compute_quality_metrics(
        self, original_data: np.ndarray, generated_data: np.ndarray
    ) -> Dict:
        """
        Compute generation quality metrics.

        Args:
            original_data: Original samples
            generated_data: Generated samples

        Returns:
            Dictionary of quality metrics
        """
        return {
            "feature_correlations": self.compute_correlation_preservation(
                pd.DataFrame(original_data), pd.DataFrame(generated_data)
            ),
            "nearest_neighbor_ratio": self._compute_nearest_neighbor_ratio(
                original_data, generated_data
            ),
            "density_overlap": self._compute_density_overlap(
                original_data, generated_data
            ),
        }

    def _compute_nearest_neighbor_ratio(
        self, original: np.ndarray,
        generated: np.ndarray,
        k: int = 5
    ) -> float:
        """
        Compute nearest neighbor distance ratio.

        Args:
            original: Original samples
            generated: Generated samples
            k: Number of neighbors

        Returns:
            Nearest neighbor distance ratio
        """
        nbrs = NearestNeighbors(n_neighbors=k).fit(original)
        orig_distances = nbrs.kneighbors(original)[0].mean()
        gen_distances = nbrs.kneighbors(generated)[0].mean()
        return float(gen_distances / orig_distances)

    def _compute_density_overlap(
        self, original: np.ndarray,
        generated: np.ndarray
    ) -> float:
        """
        Compute density overlap between distributions.

        Args:
            original: Original samples
            generated: Generated samples

        Returns:
            Density overlap score
        """
        # Compute KDE for both distributions
        kde_orig = gaussian_kde(original.T)
        kde_gen = gaussian_kde(generated.T)

        # Evaluate densities
        x_eval = np.linspace(
            min(original.min(), generated.min()),
            max(original.max(), generated.max()),
            1000,
        )

        density_orig = kde_orig(x_eval)
        density_gen = kde_gen(x_eval)

        # Compute overlap
        return float(
            np.minimum(density_orig, density_gen).sum()
            / np.maximum(density_orig, density_gen).sum()
        )
