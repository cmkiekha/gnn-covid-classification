import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src_v2.config import DEVICE, CV_N_SPLITS, RESULT_DIR
from src.utils.evaluation import (
    compare_statistics,
    compare_distributions,
    generate_tsne,
)
from src_v2.config import (
    DEVICE,
    RANDOM_STATE,
    SAVE_INFO,
    LEARNING_RATE,
    CV_N_SPLITS,
    BATCH_SIZE,
    DEV_EPOCHS,
    RESULT_DIR,
)


def evaluate_wgan_samples(saved_splits_path, num_folds=CV_N_SPLITS):
    ks_stats_list = []

    for fold in range(1, num_folds + 1):
        fold_path = saved_splits_path / f"fold{fold}"

        # Load fold data
        try:
            tr_data = torch.load(fold_path / "tr_data.pt")
            te_data = torch.load(fold_path / "te_data.pt")
            synthetic_data = torch.load(fold_path / "synthetic_data.pt")
        except FileNotFoundError:
            print(f"Error: Missing data files in {fold_path}")
            continue

        # Convert data to DataFrames for evaluation
        te_df, synthetic_df = pd.DataFrame(te_data), pd.DataFrame(synthetic_data)

        # Evaluate distributions and generate plots
        stats_df = compare_statistics(te_df, synthetic_df)
        distr_df = compare_distributions(te_df, synthetic_df)
        generate_tsne(te_df, synthetic_df)

        # Plot and save KS statistics
        ks_stats = distr_df["KS Statistic"]
        plt.hist(ks_stats)
        plt.title(f"KS Statistic Distribution for Fold {fold}")
        plt.xlabel("KS Statistic")
        plt.ylabel("Frequency")
        plt.savefig(fold_path / "ks_stats.pdf")
        plt.close()

        ks_stats_list.append(ks_stats.mean())

    print("Average KS Statistic across folds:", np.mean(ks_stats_list))


# After training
plt.plot(range(1, epochs + 1), ks_stats_per_epoch)
plt.xlabel("Epoch")
plt.ylabel("Mean KS Statistic")
plt.title("KS Statistic Over Epochs")
plt.show()


if __name__ == "__main__":
    evaluate_wgan_samples(RESULT_DIR)
