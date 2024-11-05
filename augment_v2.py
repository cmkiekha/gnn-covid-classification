# augment_v2.py
from src.models.data_augmentation.GAN_v2 import train_and_generate
import config

def main():
    """Main function to run the GAN-based data augmentation pipeline."""
    
    # Get dataset path
    dataset_path = input("Enter the dataset path (hit enter to keep default): ")
    if not dataset_path:
        dataset_path = "/Users/carolkiekhaefer10-2023/Documents/COVID-19_CKOA/4-13-24/data/data_combined_controls.csv"

    print(f"\nLoading dataset from path: {dataset_path}")
    print("\nInitiating WGAN-GP training for COVID-19 control augmentation...\n")
    
    # Train GAN and generate synthetic samples
    generated_samples = train_and_generate(
        dataset_path,
        batch_size=config.BATCH_SIZE,
        epochs=config.DEV_EPOCHS,
        device=config.DEVICE
    )
    
    print("\nData augmentation completed successfully.")
    return generated_samples

if __name__ == "__main__":
    main()