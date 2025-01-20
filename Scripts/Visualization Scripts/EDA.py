import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'Sentiment_Combined_V2.xlsx'
df = pd.read_excel(file_path)

# Drop the specified columns
df = df.drop(columns=['Website', 'Content', 'Header', 'SubHeader'])

# Rename the columns by adding 'Vader_' prefix
df = df.rename(columns={
    'Sentiment_Score_Content': 'Vader_Sentiment_Score_Content',
    'Sentiment_Score_Header': 'Vader_Sentiment_Score_Header',
    'Sentiment_Score_SubHeader': 'Vader_Sentiment_Score_SubHeader'
})

# Convert 'Dates' column to datetime
df['Dates'] = pd.to_datetime(df['Dates'])

# Drop the 'Theme' column for correlation heatmap
corr_df = df.drop(columns=['Theme'])

# Example of line plot for time series analysis
plt.figure(figsize=(12, 6))
sns.lineplot(x='Dates', y='Finbert_Sentiment_Score_Content', data=df)
plt.title('Time Series of Finbert Sentiment Scores')
plt.xticks(rotation=45)
plt.show()

# Example of line plot for time series analysis
plt.figure(figsize=(12, 6))
sns.lineplot(x='Dates', y='Vader_Sentiment_Score_Content', data=df)
plt.title('Time Series of Vader Sentiment Scores')
plt.xticks(rotation=45)
plt.show()


# Melt the DataFrame to make it suitable for a grouped bar plot
melted_df = df.melt(id_vars=['Theme'],
                    value_vars=['Finbert_Sentiment_Score_Content', 'Vader_Sentiment_Score_Content'],
                    var_name='Sentiment_Type', value_name='Sentiment_Score')

# Plot a bar chart
plt.figure(figsize=(14, 8))
sns.barplot(x='Theme', y='Sentiment_Score', hue='Sentiment_Type', data=melted_df, ci=None)
plt.title('Comparison of Sentiment Scores by Theme')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(14, 8))
sns.boxplot(x='Theme', y='Sentiment_Score_Content_Difference', data=df)
plt.title('Boxplot of Content Sentiment Score Differences by Theme')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(14, 8))
sns.boxplot(x='Theme', y='Sentiment_Score_Header_Difference', data=df)
plt.title('Boxplot of Header Sentiment Score Differences by Theme')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(14, 8))
sns.boxplot(x='Theme', y='Sentiment_Score_SubHeader_Difference', data=df)
plt.title('Boxplot of SubHeader Sentiment Score Differences by Theme')
plt.xticks(rotation=90)
plt.show()

"""Price EDA"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'Cryptocurrency_Price_V2.xlsx'
df = pd.read_excel(file_path)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Plotting Volume over time by Cryptocurrency
plt.figure(figsize=(14, 8))
sns.lineplot(x='Date', y='Volume', hue='Cryptocurrency', data=df)
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Trading Volume Over Time by Cryptocurrency')
plt.xticks(rotation=45)
plt.legend(title='Cryptocurrency', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'Cryptocurrency_Price_V2.xlsx'
df = pd.read_excel(file_path)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filter the dataset for Bitcoin only (if needed)
bitcoin_df = df[df['Cryptocurrency'] == 'Cardano']

# Create a plot with dual y-axes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot Movement on the primary y-axis
ax1.set_xlabel('Date')
ax1.set_ylabel('Movement', color='tab:blue')
ax1.plot(bitcoin_df['Date'], bitcoin_df['Movement'], color='tab:blue', label='Movement')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a secondary y-axis to plot Volume
ax2 = ax1.twinx()
ax2.set_ylabel('Volume', color='tab:orange')
ax2.plot(bitcoin_df['Date'], bitcoin_df['Volume'], color='tab:orange', label='Volume')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Add titles and show the plot
plt.title('Cardano Movement and Volume Over Time')
fig.tight_layout()  # Adjust layout to prevent clipping
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'Cryptocurrency_Price_V2.xlsx'
df = pd.read_excel(file_path)

# Drop the 'Cryptocurrency' column
df = df.drop(columns=['Cryptocurrency'])

# Convert 'Date' column to datetime, and if needed, set it as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Cryptocurrency Data (excluding Cryptocurrency column)')
plt.show()