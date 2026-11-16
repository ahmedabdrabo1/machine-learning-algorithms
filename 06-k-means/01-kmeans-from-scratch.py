import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs

x, _ = make_blobs(
    n_samples=100,
    centers=4,
    cluster_std=0.8,
    random_state=42
)

df = pd.DataFrame(x, columns=['Annual', 'Spending'])

np.random.seed(42)

def KMeans(x, k, iters=10):

    centroids = x[np.random.choice(len(x), k, replace=False)]

    for _ in range(iters):

        distances = []

        for c in centroids:
            dis = np.sum((x - c)**2, axis=1)
            dis = np.sqrt(dis)
            distances.append(dis)

        distances = np.array(distances)

        labels = np.argmin(distances, axis=0)

        old_centroids = centroids.copy()

        new_centroids = []

        for j in range(k):
            points = x[labels == j]

            if len(points) > 0:
                new_centroids.append(np.mean(points, axis=0))
            else:
                new_centroids.append(centroids[j])

        centroids = np.array(new_centroids)

        if np.allclose(old_centroids, centroids):
            break

    return labels, centroids


labels, centroids = KMeans(x, 3)

print(labels)
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