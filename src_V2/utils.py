import pandas as pd


def load_raw_data(filepath, to_tensor=False):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)

    if to_tensor:  # Should we convert the dataframe to a tensor?
        df = torch.from_numpy(df.values)

    return df


def df_to_dataloader(data_df, batch_size=32, y_name="Group"):
    # Function transforms a DataFrame into a DataLoader
    target = data_df.pop(y_name)  # Assuming 'Group' is in data_df
    dataset = TensorDataset(
        torch.tensor(data_df.values, dtype=torch.float32),
        torch.tensor(target.values, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
