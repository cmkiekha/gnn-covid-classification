import pandas as pd
from scipy.stats import ks_2samp
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def compare_statistics(df1, df2):
    comparison_dict = {"Column": [], "Mean Difference": [], "Variance Difference": []}

    # Ensure both DataFrames have the same columns
    common_columns = df1.columns.intersection(df2.columns)

    for column in common_columns:
        mean_diff = df1[column].mean() - df2[column].mean()
        var_diff = df1[column].var() - df2[column].var()

        comparison_dict["Column"].append(column)
        comparison_dict["Mean Difference"].append(mean_diff)
        comparison_dict["Variance Difference"].append(var_diff)

    comparison_df = pd.DataFrame(comparison_dict)
    return comparison_df


def compare_distributions(df1, df2):
    ks_results = {"Column": [], "KS Statistic": [], "P-Value": []}

    for column in df1.columns:
        statistic, pvalue = ks_2samp(df1[column], df2[column])
        ks_results["Column"].append(column)
        ks_results["KS Statistic"].append(statistic)
        ks_results["P-Value"].append(pvalue)

    ks_df = pd.DataFrame(ks_results)
    return ks_df


def plot_ks_statistics_over_epochs(epochs, real_samples, fake_samples):
    ks_stats_per_epoch = []

    for epoch in tqdm(range(epochs), desc="Training WGAN-GP"):
        # Convert samples to numpy format if they are PyTorch tensors
        real_samples_np = real_samples.cpu().detach().numpy() if hasattr(real_samples, 'cpu') else real_samples
        fake_samples_np = fake_samples.cpu().detach().numpy() if hasattr(fake_samples, 'cpu') else fake_samples

        ks_stat = compare_distributions(real_samples_np, fake_samples_np)["KS Statistic"].mean()
        ks_stats_per_epoch.append(ks_stat)

    # Plot KS statistics over epochs
    plt.plot(range(1, epochs + 1), ks_stats_per_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Mean KS Statistic")
    plt.title("KS Statistic Over Epochs")
    plt.show()


def generate_tsne(df1, df2):
    df1["type"] = "Original"
    df2["type"] = "Synthetic"
    combined_df = pd.concat([df1, df2])

    data_for_tsne = combined_df.drop("type", axis=1)
    labels = combined_df["type"]

    tsne = TSNE(n_components=2, random_state=42)
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
    )

    plt.title("t-SNE Visualization of Original vs. Synthetic Data")
    plt.show()


def recenter_data(generated_samples, original_data):
    generated_samples_mean = generated_samples.mean()
    generated_samples_std = generated_samples.std()

    original_data_mean = original_data.mean()
    original_data_std = original_data.std()

    generated_samples_centered = (
        generated_samples - generated_samples_mean
    ) / generated_samples_std
    generated_samples = (
        generated_samples_centered * original_data_std + original_data_mean
    )

    return generated_samples
