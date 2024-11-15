# augment.py

# Import necessary modules

from src.models.data_augmentation.GAN import train_and_generate
import preprocessing
import evaluation
import torch
import numpy as np
import pandas as pd


def process_and_augment_data(dataset_path):
    """
    Process and augment data using the GAN model.

    Args:
        dataset_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: The recentered augmented data.
    """
    # Process the dataset
    dataset, tensor_data, scaled_data, scaler, original_dim = preprocessing.process(dataset_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate synthetic data using GAN
    generated_samples = train_and_generate(dataset_path, batch_size=32, epochs=100, device=device)
    augmented_df = pd.DataFrame(generated_samples, columns=scaled_data.columns)

    # Recenter augmented data using evaluation functions
    augmented_data_recentered = evaluation.recenter_data(np.array(augmented_df), np.array(scaled_data))
    print("Augmentation process completed successfully.")
    return augmented_data_recentered

def main():
    dataset_path = "/Users/carolkiekhaefer10-2023/Documents/COVID-19_CKOA/4-13-24/data/data_combined_controls.csv"
    
    # Process and augment data using GAN
    augmented_data_recentered = process_and_augment_data(dataset_path)
    print(augmented_data_recentered.head())  # Example to show some data output

if __name__ == "__main__":
    main()

## BELOW IS ORIGINAL
# from src.models.data_augmentation.VAE import *
# from src.models.data_augmentation.WAE import *
# from src.models.data_augmentation.GAN import *

# dataset_path = input("Enter the dataset path (hit enter to keep default): ")

# if not len(dataset_path):
#     dataset_path = "/Users/carolkiekhaefer10-2023/Documents/COVID-19_CKOA/4-13-24/data/data_combined_controls.csv"

# print(f"\nLoading dataset from path: {dataset_path}")

# model_type = input(
#     "Select the data generator model \n\n1. VAE\n2. WAE\n3. GAN\n\nSelection: "
# )

# if model_type not in ("1", "2", "3"):
#     print("\n\nYou selected an invalid option. Terminating script.\n\n")
#     exit(0)

# dataset, tensor_data, scaled_data, scaler, original_dim = process(dataset_path)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if model_type == "1":
#     print("\n\nYou selected VAE. Training initiated.\n\n")

#     vae = train_vae(dataset, original_dim)
#     augmented_df = generate_vae(vae, scaled_data.columns, scaler)

# elif model_type == "2":
#     print("\n\nYou selected WAE. Training initiated.\n\n")

#     wae = train_wae(dataset, original_dim)
#     augmented_df = generate_wae(wae, scaled_data.columns, scaler)

# elif model_type == "3":
#     print("\n\nYou selected GAN. Training initiated.\n\n")

#     generated_samples = train_and_generate(dataset_path, batch_size=32, epochs=100)
