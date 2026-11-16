import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

x, _ = make_blobs(
    n_samples=100,
    centers=4,
    cluster_std=0.8,
    random_state=42
)

df = pd.DataFrame(x, columns=['Annual', 'Spending'])

kmeans = KMeans(
    n_clusters=2,
    init='random',
    n_init=10,
    max_iter=10,
    random_state=42
)

kmeans.fit(x)

labels = kmeans.labels_

centroids = kmeans.cluster_centers_

print("Labels:")
print(labels)

print("\nCentroids:")
print(centroids)

plt.scatter(
    x[:, 0],
    x[:, 1],
    c=labels,
    cmap='viridis'
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color='red',
    marker='X',
    s=200,
    label='Centroids'
)

plt.xlabel('Annual')
plt.ylabel('Spending')
plt.title('K-Means Clustering')
plt.legend()
plt.show()