import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
import kagglehub
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Download latest version
path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")

print("Path to dataset files:", path)

df = pd.read_csv("C:\\Users\\Anindita\\.cache\\kagglehub\\datasets\\maharshipandya\\-spotify-tracks-dataset\\versions\\1\\dataset.csv")
df_sampled = df.sample(n=500,random_state=5)

features = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','track_genre']
df_features = df_sampled[features]
df_encoded = pd.get_dummies(df_features,columns=['key','track_genre'],drop_first=False)

scalar = MinMaxScaler()

df_scaled = scalar.fit_transform(df_encoded)
df_scaled = pd.DataFrame(df_scaled)

optimal_k = 3  
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df_sampled['cluster'] = kmeans.fit_predict(df_scaled)

#print(df_sampled[['track_name','cluster']].head(20))

cluster_summary = df_sampled.groupby('cluster')[[
    'danceability', 'energy', 'valence', 'tempo', 
    'acousticness', 'instrumentalness', 'loudness'
]].mean()

print(cluster_summary)

cluster_moods = {
    0: "Sad / Acoustic",
    1: "Happy / Upbeat",
    2: "Electronic / Dance"
}

df_sampled['mood_label'] = df_sampled['cluster'].map(cluster_moods)

final_features = ['artists','album_name','track_name','popularity','cluster','mood_label','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','track_genre']

df_final = df_sampled[final_features]

print(df_final.columns)

df_final.to_csv('spotify_mood_clusters.csv', index=False)
cluster_summary.to_csv('cluster_summary.csv')


