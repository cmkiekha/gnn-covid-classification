import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
from typing import List


def recenter_data(
    generated_samples: np.ndarray, original_data: np.ndarray, epsilon: float = 1e-10
) -> np.ndarray:
    """
    Adjusts the generated samples to match the statistical properties (mean and standard deviation)
    of the original data. This recentering ensures that the synthetic data mimics the real data's
    distribution more closely.

    Args:
        generated_samples (numpy.array): The synthetic data generated by the model.
        original_data (numpy.array): The real data used as the reference.

    Returns:
        numpy.array: Recentered synthetic data.
    """
    # Validate inputs
    if generated_samples.shape[1] != original_data.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {generated_samples.shape[1]} vs {original_data.shape[1]}"
        )

    if not np.isfinite(generated_samples).all() or not np.isfinite(original_data).all():
        raise ValueError("Input arrays contain inf or nan values")

    # Calculate statistics
    gen_mean = np.mean(generated_samples, axis=0)
    gen_std = np.std(generated_samples, axis=0) + epsilon
    orig_mean = np.mean(original_data, axis=0)
    orig_std = np.std(original_data, axis=0) + epsilon

    # Standardize and rescale
    standardized = (generated_samples - gen_mean) / gen_std
    recentered = (standardized * orig_std) + orig_mean

    # Verify recentering
    if not np.isfinite(recentered).all():
        raise ValueError("Recentering produced invalid values")

    return recentered


def compare_statistics(df1, df2):
    """
    Compares the statistical properties between original and synthesized
    Calculates differences in means and variances for each feature, providing
    a comprehensive comparison of the statistical properties of both datasets.

    Args:
        df1 (pd.DataFrame): First dataframe (typically original data).
        df2 (pd.DataFrame): Second dataframe (typically synthetic data).

    Returns:
        pd.DataFrame: DataFrame containing the mean and variance differences for each column.

    Example:
    >>> stats_comparison = compare_statistics(original_df, synthetic_df)
    >>> print(stats_comparison)
    """
    comparison_dict = {
        "Column": [],
        "Mean Difference": [],
        "Variance Difference": [],
        "Relative_Mean_Diff_%": [],
        "Relative_Var_Diff_%": [],
    }

    for column in df1.columns:
        original_mean = df1[column].mean()
        synthetic_mean = df2[column].mean()
        original_var = df1[column].var()
        synthetic_var = df2[column].var()

        mean_diff = original_mean - synthetic_mean
        var_diff = original_var - synthetic_var

        # Calculate relative differences as percentages
        rel_mean_diff = (
            (mean_diff / original_mean) * 100 if original_mean != 0 else np.inf
        )
        rel_var_diff = (var_diff / original_var) * 100 if original_var != 0 else np.inf

        comparison_dict["Feature"].append(column)
        comparison_dict["Mean_Difference"].append(mean_diff)
        comparison_dict["Variance_Difference"].append(var_diff)
        comparison_dict["Relative_Mean_Diff_%"].append(rel_mean_diff)
        comparison_dict["Relative_Var_Diff_%"].append(rel_var_diff)

    return pd.DataFrame(comparison_dict)


def compare_distributions(df1, df2):
    """
    Performs the Kolmogorov-Smirnov test for each column in the dataframes to compare
    their distributions.

    Args:
        df1 (pd.DataFrame): First dataframe (original data).
        df2 (pd.DataFrame): Second dataframe (synthetic data).

    Returns:
        pd.DataFrame: Results of the KS test, including the statistic and p-value for each column.

    Example:
    >>> ks_results = compare_distributions(original_df, synthetic_df)
    >>> significant_differences = ks_results[ks_results['P-Value'] < 0.05]
    """
    ks_results = {
        "Column": [],
        "KS Statistic": [],
        "P-Value": [],
        "Distribution_Match": [],
    }

    for column in df1.columns:
        statistic, pvalue = ks_2samp(df1[column], df2[column])
        ks_results["Column"].append(column)
        ks_results["KS Statistic"].append(statistic)
        ks_results["P-Value"].append(pvalue)
        ks_results["Distribution_Match"].append(
            "Similar" if pvalue > 0.05 else "Different"
        )

    results_df = pd.DataFrame(ks_results)
    results_df = results_df.sort_values("KS_Statistic", ascending=False)
    return results_df


def generate_tsne(
    df1: pd.DataFrame, df2: pd.DataFrame, perplexity: int = 30, n_iter: int = 1000
) -> None:
    """
    Generates a t-SNE visualization to compare the feature space of original and synthetic data.
    Creates a 2D visualization that helps assess how well the synthetic data
    captures the structure of the original data distribution.

    Args:
        df1 (pd.DataFrame): DataFrame containing original data.
        df2 (pd.DataFrame): DataFrame containing synthetic data.

    Returns:
        None: Displays a scatter plot of the t-SNE results.
    """
    # Prepare the data for t-SNE
    df1["type"] = "Original"
    df2["type"] = "Synthetic"
    combined_df = pd.concat([df1, df2])

    data_for_tsne = combined_df.drop("type", axis=1)
    labels = combined_df["type"]

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(data_for_tsne)

    tsne_df = pd.DataFrame(
        {"TSNE-1": tsne_results[:, 0], "TSNE-2": tsne_results[:, 1], "Type": labels}
    )

    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        x="TSNE-1",
        y="TSNE-2",
        hue="Type",
        palette=sns.color_palette("hsv", 2),
        data=tsne_df,
        legend="full",
        alpha=0.7,
    ).set_title("t-SNE Visualization of Original vs. Synthetic Data")
    plt.show()


# def recenter_data(generated_samples, original_data):
#     """
#     Recenters the generated data to match the mean and standard deviation of the original data.

#     Args:
#         generated_samples (numpy.ndarray): Generated data to be recentered.
#         original_data (numpy.ndarray): Original data used as the reference for recentering.

#     Returns:
#         numpy.ndarray: Recentered generated data.
#     """
#     generated_samples_mean = generated_samples.mean()
#     generated_samples_std = generated_samples.std()
#     original_data_mean = original_data.mean()
#     original_data_std = original_data.std()

#     generated_samples_centered = (generated_samples - generated_samples_mean) / generated_samples_std
#     return generated_samples_centered * original_data_std + original_data_mean


def plot_feature_distributions(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    features: List[str] = None,
    max_features: int = 10,
) -> None:
    """
    Create side-by-side distribution plots comparing original and synthetic data.

    Args:
        original_df (pd.DataFrame): Original dataset
        synthetic_df (pd.DataFrame): Synthetic dataset
        features (List[str]): List of features to plot. If None, uses all features
        max_features (int): Maximum number of features to plot

    Returns:
        None: Displays the plots directly
    """
    if features is None:
        features = original_df.columns[:max_features]
    else:
        features = features[:max_features]

    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        sns.kdeplot(
            data=original_df[feature], ax=axes[idx], label="Original", color="blue"
        )
        sns.kdeplot(
            data=synthetic_df[feature], ax=axes[idx], label="Synthetic", color="red"
        )
        axes[idx].set_title(f"Distribution of {feature}")
        axes[idx].legend()

    # Remove empty subplots
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


# Function to get original data for testing
def get_test_data(data: pd.DataFrame, test_fold: int = None) -> pd.DataFrame:
    """
    Extract original test data.

    Args:
        data: Combined DataFrame containing both original and synthetic data
        test_fold: Optional fold number to get specific fold's data

    Returns:
        DataFrame containing only original test data
    """
    # Get original data
    original_data = data[data["data_type"] == "original"].copy()

    if test_fold is not None:
        # Get data from specific fold
        test_data = original_data[original_data["fold"] == test_fold]
    else:
        # Get all original data
        test_data = original_data

    print(f"Selected {len(test_data)} original samples for testing")

    return test_data
