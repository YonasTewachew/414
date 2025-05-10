import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


df = pd.read_csv('2022-23 Football Player Stats.csv', encoding='utf-8')

if 'Pos' in df.columns:
    df = df[~df['Pos'].str.contains('GK')]
if '90s' in df.columns:
    df = df[df['90s'] >= 5]
elif 'Minutes_Played' in df.columns:
    df = df[df['Minutes_Played'] >= 450]

features = {
    'Passes_per90': ('Passes_Completed',),
    'LongPass_per90': ('Passes_Cmp_Long',),
    'Tackles_per90': ('Tackles',),
    'Interceptions_per90': ('Interceptions',),
    'Dribbles_per90': ('Succ_Take',),
    'Shots_per90': ('Shots',),
    'TouchesAttThird_per90': ('Final_Third',)
}

if '90s' in df.columns:
    df = df[df['90s'] >= 2]
elif 'Minutes_Played' in df.columns:
    df = df[df['Minutes_Played'] >= 180]

for new_col, (raw_col,) in features.items():
    if raw_col in df.columns and '90s' in df.columns:
        df[new_col] = df[raw_col] / df['90s']
    elif raw_col in df.columns and 'Minutes_Played' in df.columns:
        df[new_col] = df[raw_col] / (df['Minutes_Played'] / 90)

df = df.dropna(subset=list(features.keys()))


X = df[list(features.keys())].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans_tmp = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans_tmp.fit(X_scaled)
    wcss.append(kmeans_tmp.inertia_)
    sil = silhouette_score(X_scaled, kmeans_tmp.labels_)
    sil_scores.append(sil)

optimal_k = 6
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=1)
df['Cluster'] = kmeans.fit_predict(X_scaled)
features = ['Passes_per90', 'LongPass_per90', 'Tackles_per90', 'Interceptions_per90',
            'Dribbles_per90', 'Shots_per90', 'TouchesAttThird_per90']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]


plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', s=70, alpha=0.8)
plt.title('PCA Plot of Player Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

cluster_means = df.groupby('Cluster')[features].mean().round(2)
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Cluster Averages for Key Stats')
plt.xlabel('Features')
plt.ylabel('Cluster')
plt.tight_layout()
plt.show()

centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)
centroid_df = pd.DataFrame(centroids, columns=features)
centroid_df.index.name = 'Cluster'
print("Cluster centroids (average per-90 stats):")
print(centroid_df.round(2))

print("\nExample players in each cluster:")
for c in range(optimal_k):
    cluster_players = df[df['Cluster'] == c]
    if 'Minutes_Played' in cluster_players.columns:
        top_players = cluster_players.sort_values('Minutes_Played', ascending=False).head(3)
    else:
        top_players = cluster_players.head(3)
    player_names = top_players['Player'].tolist()
    print(f"Cluster {c}: {player_names}")
