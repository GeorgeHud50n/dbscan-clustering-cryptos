import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Fetch and preprocess cryptocurrency data
def fetch_cryptocurrency_data():
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false"
    response = requests.get(url)
    data = pd.DataFrame(response.json())
    return data

def preprocess_data(df):
    df = df[['id', 'name', 'symbol', 'current_price', 'market_cap', 'total_volume', 'price_change_percentage_24h', 'circulating_supply', 'total_supply', 'ath', 'atl']].copy()
    df['price_change_percentage_7d_in_currency'] = df['price_change_percentage_24h']
    df['price_change_percentage_30d_in_currency'] = df['price_change_percentage_24h']
    df['log_returns'] = np.log(1 + df['price_change_percentage_24h'] / 100)
    return df


crypto_data = fetch_cryptocurrency_data()
crypto_data = preprocess_data(crypto_data)

# Perform DBSCAN clustering
def perform_dbscan_clustering(df, eps=3, min_samples=2):
    features = ['market_cap', 'total_volume', 'circulating_supply', 'price_change_percentage_7d_in_currency', 'price_change_percentage_30d_in_currency']
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(x)
    return clusters

crypto_data['dbscan_cluster'] = perform_dbscan_clustering(crypto_data)

# Find undervalued cryptocurrencies within each DBSCAN cluster
def find_undervalued_cryptos_dbscan(df):
    undervalued_cryptos = pd.DataFrame()
    for cluster in df['dbscan_cluster'].unique():
        if cluster != -1:  # Ignore noise points
            cluster_data = df[df['dbscan_cluster'] == cluster]
            avg_log_returns = cluster_data['log_returns'].mean()
            undervalued = cluster_data[cluster_data['log_returns'] < avg_log_returns]
            undervalued_cryptos = undervalued_cryptos.append(undervalued)
    return undervalued_cryptos

undervalued_cryptos_dbscan = find_undervalued_cryptos_dbscan(crypto_data)

# Create a scatter plot to visualize the data
def plot_scatter(df, undervalued_df):
    fig, ax = plt.subplots(figsize=(15, 10))

    for cluster in df['dbscan_cluster'].unique():
        cluster_data = df[df['dbscan_cluster'] == cluster]
        undervalued_data = undervalued_df[undervalued_df['dbscan_cluster'] == cluster]

        ax.scatter(cluster_data['market_cap'], cluster_data['log_returns'], label=f'Cluster {cluster}', alpha=0.8)
        ax.scatter(undervalued_data['market_cap'], undervalued_data['log_returns'], marker='x', color='red', s=100, label=f'Undervalued (Cluster {cluster})')

        for index, row in undervalued_data.iterrows():
            ax.text(row['market_cap'], row['log_returns'], row['symbol'], fontsize=10, ha='left', va='bottom')

    ax.set_xscale('log')
    ax.set_xlabel('Market Cap')
    ax.set_ylabel('Log Returns')
    ax.set_title('Scatter Plot of Log Returns vs. Market Cap (DBSCAN Clustering)')
    ax.legend()
    plt.show()

plot_scatter(crypto_data, undervalued_cryptos_dbscan)
