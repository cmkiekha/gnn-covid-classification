import pandas as pd
from scipy.stats import ks_2samp
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def compare_statistics(df1, df2):
    comparison_dict = {
        'Column': [],
        'Mean Difference': [],
        'Variance Difference': []
    }
    
    for column in df1.columns:
        mean_diff = df1[column].mean() - df2[column].mean()
        var_diff = df1[column].var() - df2[column].var()
        
        comparison_dict['Column'].append(column)
        comparison_dict['Mean Difference'].append(mean_diff)
        comparison_dict['Variance Difference'].append(var_diff)
        
    comparison_df = pd.DataFrame(comparison_dict)
    return comparison_df

# Kolmogorov-Smirnov test
def compare_distributions(df1, df2):
    ks_results = {
        'Column': [],
        'KS Statistic': [],
        'P-Value': []
    }
    
    for column in df1.columns:
        statistic, pvalue = ks_2samp(df1[column], df2[column])
        ks_results['Column'].append(column)
        ks_results['KS Statistic'].append(statistic)
        ks_results['P-Value'].append(pvalue)
        
    ks_df = pd.DataFrame(ks_results)
    return ks_df

def generate_tsne(df1, df2):
    # Combine the dataframes with a label column to distinguish them
    df1['type'] = 'Original'
    df2['type'] = 'Synthetic'
    combined_df = pd.concat([df1, df2])
    
    # Exclude the label column for t-SNE
    data_for_tsne = combined_df.drop('type', axis=1)
    labels = combined_df['type']
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(data_for_tsne)
    
    tsne_df = pd.DataFrame({
        'TSNE-1': tsne_results[:, 0],
        'TSNE-2': tsne_results[:, 1],
        'Type': labels
    })
    
    # Plotting
    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        x='TSNE-1', y='TSNE-2',
        hue='Type',
        palette=sns.color_palette("hsv", 2),
        data=tsne_df,
        legend="full",
        alpha=0.7
    )

    plt.title('t-SNE Visualization of Original vs. Synthetic Data')
    plt.show()

def recenter_data(generated_samples, original_data):
    
    generated_samples_mean = generated_samples.mean()
    generated_samples_std = generated_samples.std()

    original_data_mean = original_data.mean()
    original_data_std = original_data.std()
    
    generated_samples_centered = (generated_samples - generated_samples_mean ) / generated_samples_std
    generated_samples = generated_samples_centered * original_data_std + original_data_mean

    return generated_samples

