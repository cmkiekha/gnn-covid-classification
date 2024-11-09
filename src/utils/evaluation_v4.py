import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.manifold import TSNE
from typing import Optional, Dict, List, Any


class SyntheticDataEvaluator:
    """
    A comprehensive suite for evaluating synthetic data quality.

    This class provides methods for statistical comparisons and visualizations
    between original and synthetic datasets, including functionality to recenter
    generated data to better match the original data distribution.

    Attributes:
        original_data (pd.DataFrame): Stores a copy of the original data for comparison.
        synthetic_data (pd.DataFrame): Stores a copy of the synthetic data for comparison.
        feature_columns (List[str]): List of feature columns to evaluate.
        ks_threshold (float): Threshold for the Kolmogorov-Smirnov test to determine
                              distribution similarity between original and synthetic data.
    """

    def __init__(self):
        """
        Initializes the SyntheticDataEvaluator with default settings.
        """
        self.original_data: Optional[pd.DataFrame] = None
        self.synthetic_data: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        self.ks_threshold = 0.05

    @staticmethod
    def recenter_data(
        generated_samples: np.ndarray, original_data: np.ndarray, epsilon: float = 1e-10
    ) -> np.ndarray:
        """
        Recenter synthetic data to align with the mean and standard deviation of the original data.

        Parameters:
            generated_samples (np.ndarray): The synthetic data to be recentered.
            original_data (np.ndarray): The original data used for reference in recentering.
            epsilon (float): Small constant to avoid division by zero in standard deviation calculation.

        Returns:
            np.ndarray: The recentered synthetic data.

        Raises:
            ValueError: If the feature dimensions of generated and original data do not match,
                        or if the recentering process produces non-finite values.
        """
        if generated_samples.shape[1] != original_data.shape[1]:
            raise ValueError(
                "Feature dimension mismatch between generated and original data."
            )

        gen_mean, gen_std = (
            np.mean(generated_samples, axis=0),
            np.std(generated_samples, axis=0) + epsilon,
        )
        orig_mean, orig_std = (
            np.mean(original_data, axis=0),
            np.std(original_data, axis=0) + epsilon,
        )

        standardized = (generated_samples - gen_mean) / gen_std
        recentered = (standardized * orig_std) + orig_mean

        if not np.isfinite(recentered).all():
            raise ValueError("Recentering produced invalid values.")

        return recentered

    def evaluate(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        recenter: bool = True,
    ) -> Dict[str, Any]:
        """
        Conducts a comprehensive evaluation of synthetic data quality, including statistical
        comparisons and recentering of synthetic data if specified.

        Parameters:
            original_data (pd.DataFrame): The original dataset for comparison.
            synthetic_data (pd.DataFrame): The synthetic dataset to evaluate.
            recenter (bool): Whether to recenter the synthetic data to match the original data distribution.

        Returns:
            Dict[str, Any]: A dictionary with statistical comparison, distribution comparison,
                            and data summary results.
        """
        self.original_data = original_data.copy()
        self.synthetic_data = synthetic_data.copy()

        # Define feature columns for evaluation
        self.feature_columns = [
            col
            for col in synthetic_data.columns
            if col not in ["fold", "data_type", "sample_id"]
        ]

        if recenter:
            recentered_data = self.recenter_data(
                synthetic_data[self.feature_columns].values,
                original_data[self.feature_columns].values,
            )
            self.synthetic_data[self.feature_columns] = recentered_data

        results = {
            "statistical_comparison": self.compare_statistics(),
            "distribution_comparison": self.compare_distributions(),
            "data_summary": self.generate_summary(),
        }

        self.plot_evaluation(results)
        return results

    def compare_statistics(self) -> pd.DataFrame:
        """
        Compares statistical properties (mean and standard deviation) of original
        and synthetic data for each feature.

        Returns:
            pd.DataFrame: A DataFrame summarizing mean and standard deviation differences for each feature.
        """
        stats_comparison = []
        for feature in self.feature_columns:
            orig_mean, synth_mean = (
                self.original_data[feature].mean(),
                self.synthetic_data[feature].mean(),
            )
            orig_std, synth_std = (
                self.original_data[feature].std(),
                self.synthetic_data[feature].std(),
            )

            comparison = {
                "Feature": feature,
                "Original_Mean": orig_mean,
                "Synthetic_Mean": synth_mean,
                "Mean_Difference": abs(orig_mean - synth_mean),
                "Original_Std": orig_std,
                "Synthetic_Std": synth_std,
                "Std_Difference": abs(orig_std - synth_std),
            }
            stats_comparison.append(comparison)

        print("Statistical comparison complete.")
        return pd.DataFrame(stats_comparison)

    def compare_distributions(self) -> pd.DataFrame:
        """
        Compares feature distributions between original and synthetic data using
        the Kolmogorov-Smirnov test.

        Returns:
            pd.DataFrame: A DataFrame summarizing KS statistics, p-values, and distribution match results.
        """
        ks_results = []
        for feature in self.feature_columns:
            ks_stat, p_value = ks_2samp(self.original_data[feature], self.synthetic_data[feature])
            ks_results.append({
                "Feature": feature,
                "KS_Statistic": ks_stat,
                "P_Value": p_value,
                "Distribution_Match": "Similar" if p_value > self.ks_threshold else "Different",
            })
        return pd.DataFrame(ks_results).sort_values("KS_Statistic", ascending=False)


    def plot_evaluation(self, results: Dict[str, Any]) -> None:
        """
        Creates visualizations for the evaluation results, including KS statistics
        and feature-wise distribution comparisons.

        Parameters:
            results (Dict[str, Any]): The evaluation results to visualize.
        """
        sns.set_style("whitegrid")
        # self.plot_ks_statistics(results["distribution_comparison"])
        # self.plot_feature_distributions()

    def plot_ks_statistics(self, ks_results: pd.DataFrame) -> None:
        """
        Plots the KS statistics for each feature, highlighting distribution matches.

        Parameters:
            ks_results (pd.DataFrame): The DataFrame containing KS test results for each feature.
        """
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=ks_results, x="Feature", y="KS_Statistic", hue="Distribution_Match"
        )
        plt.xticks(rotation=45)
        plt.title("KS Statistics by Feature")
        plt.tight_layout()
        plt.show()

    def plot_feature_distributions(self) -> None:
        for feature in self.feature_columns:
            plt.figure(figsize=(10, 4))
            sns.histplot(self.original_data[feature], label="Original", kde=True, stat="density")
            sns.histplot(self.synthetic_data[feature], label="Synthetic", kde=True, stat="density", color="orange")
            plt.title(f"Distribution Comparison: {feature}")
            plt.legend()
            plt.tight_layout()
            plt.show()


    def generate_summary(self) -> Dict[str, Any]:
        """
        Generates a summary of the synthetic and original datasets.

        Returns:
            Dict[str, Any]: A dictionary with sample counts, feature count, and evaluation timestamp.
        """
        return {
            "n_original_samples": len(self.original_data),
            "n_synthetic_samples": len(self.synthetic_data),
            "n_features": len(self.feature_columns),
            "timestamp": pd.Timestamp.now().isoformat(),
        }


# Example usage
if __name__ == "__main__":
    print("Synthetic Data Evaluation Module Loaded.")
