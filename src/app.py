import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/workspace/K-means-Project/data/df_data.csv')

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

km_6 = KMeans(n_clusters=6, random_state=42)
km_6.fit(df_scaled)

df_inv = scaler.inverse_transform(df_scaled)
df_inv = pd.DataFrame(df_inv,columns=['ind','Latitude','Longitude','MedInc'])
df_inv['Cluster'] = km_6.labels_

df_inv['Cluster'] = pd.Categorical(df_inv.Cluster)

model_inertia = km_6.inertia_