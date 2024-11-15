import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import matplotlib.pyplot as plt
import seaborn as sns

import src.config as config

from src.models.data_augmentation.WGAN import *
from src.utils.evaluation import *

dataset_path = config.DATA_PATH

(
    dataset,
    tensor_data,
    original_data_unscaled,
    leftout_original_data,
    scaler,
    original_dim,
) = process(dataset_path)

generated_data = train_and_generate_k_fold(
    dataset,
    tensor_data,
    original_data_unscaled,
    leftout_original_data,
    scaler,
    original_dim,
    config.DEV_EPOCHS,
)

generated_data = pd.DataFrame(generated_data, columns=original_data_unscaled.columns)
generated_data_unscaled = pd.DataFrame(
    scaler.inverse_transform(generated_data), columns=generated_data.columns
)
ks_stats = compare_distributions(original_data_unscaled, generated_data_unscaled)

print(f"\nGenerated data:\n\n")
print(generated_data_unscaled.head())

print(f"\nOriginal data:\n\n")
print(original_data_unscaled.head())

plt.figure(figsize=(6, 6))
sns.histplot(ks_stats["KS Statistic"], kde=True)
plt.title("Distribution of KS Statistics for Original vs. Synthetic Data")
plt.xlabel("KS Statistic")
plt.ylabel("Frequency")
plt.show()

# Save the train/test DataFrames
original_data_unscaled.to_csv("data/aug_results/control_original_train_unscaled.csv")
leftout_original_data.to_csv("data/aug_results/control_original_test_unscaled.csv")
generated_data_unscaled.to_csv("data/aug_results/control_generated_unscaled.csv")

# Save the Robust Scaler to reuse
# TODO: The scaler should be fit on the Train of both Case and Control.
joblib.dump(scaler, "data/aug_results/robustScalerScaledOnControlONLY.pkl")
