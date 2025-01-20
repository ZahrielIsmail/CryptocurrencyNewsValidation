import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Load the Excel file
file_path = 'Model Performance and Evaluation.xlsx'
sheet_names = ['Ethereum_V2_Metrics', 'Cardano_V2_Metrics', 'Shib_V2_Metrics']

# Read the specific sheets
df_ethereum = pd.read_excel(file_path, sheet_name='Ethereum_V2_Metrics')
df_cardano = pd.read_excel(file_path, sheet_name='Cardano_V2_Metrics')
df_shib = pd.read_excel(file_path, sheet_name='Shib_V2_Metrics')

def plot_metric(df, metric, title):
    plt.figure(figsize=(14, 8))
    plt.bar(df['Model Name'], df[metric], color=['blue', 'green', 'red'])
    plt.xlabel('Model Name')
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

metrics = ['MSE', 'RMSE', 'MAE', 'R-squared']
for metric in metrics:
    plot_metric(df_ethereum, metric, f'Ethereum - {metric}')
    plot_metric(df_cardano, metric, f'Cardano - {metric}')
    plot_metric(df_shib, metric, f'Shib - {metric}')

# Function to plot metrics for a single cryptocurrency with distinct colors
def plot_metrics_for_crypto(df, crypto_name):
    metrics = ['MSE', 'RMSE', 'MAE', 'R-squared']
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid for 4 metrics
    colors = cm.tab20(np.linspace(0, 1, len(df['Model Name'])))

    for i, metric in enumerate(metrics):
        row, col = divmod(i, 2)
        axs[row, col].barh(df['Model Name'], df[metric], color=colors)
        axs[row, col].set_ylabel('Model Name')
        axs[row, col].set_xlabel(metric)
        axs[row, col].set_title(f'{crypto_name} - {metric}')

    plt.tight_layout()
    plt.savefig(f'{crypto_name}_performance_metrics.png')
    plt.show()

# Plot metrics for each cryptocurrency
plot_metrics_for_crypto(df_ethereum, 'Ethereum')
plot_metrics_for_crypto(df_cardano, 'Cardano')
plot_metrics_for_crypto(df_shib, 'Shib')