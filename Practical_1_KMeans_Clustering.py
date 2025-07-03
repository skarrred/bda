# Practical 1: K-Means Clustering (Simulated Data in Python)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Simulate student data
np.random.seed(123)
data = {
    'English': np.random.randint(50, 100, 30),
    'Math': np.random.randint(50, 100, 30),
    'Science': np.random.randint(50, 100, 30)
}
df = pd.DataFrame(data)

# Elbow method
wss = []
for k in range(1, 16):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmeans.fit(df)
    wss.append(kmeans.inertia_)

plt.plot(range(1, 16), wss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Within Sum of Squares')
plt.title('Elbow Method for KMeans')
plt.show()

# KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)
df['Cluster'] = kmeans.fit_predict(df)

# Plot clusters
plt.scatter(df['English'], df['Math'], c=df['Cluster'], cmap='viridis')
plt.title('Student Cluster (English vs Math)')
plt.xlabel('English')
plt.ylabel('Math')
plt.show()