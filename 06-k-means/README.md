# K-Means Clustering

## What is K-Means?

K-Means is an **Unsupervised Machine Learning algorithm** used to group data points into clusters based on their similarity.

It divides a dataset into K clusters, where each cluster is represented by a center point called a **Centroid**.

The main goal is to group similar data points together while keeping different groups separated.

## How Does K-Means Work?

K-Means follows these steps:

1. **Choose K:** Specify the number of clusters.
2. **Initialize Centroids:** Select K initial centroids from the dataset.
3. **Calculate Distances:** Calculate the distance between each data point and all centroids using Euclidean Distance.
4. **Assign Clusters:** Assign each data point to its nearest centroid.
5. **Update Centroids:** Calculate the mean of all data points in each cluster to update its centroid.
6. **Repeat:** Continue assigning points and updating centroids until the centroids stop changing or the maximum number of iterations is reached.

## Euclidean Distance

K-Means uses Euclidean Distance to measure the distance between a data point and a centroid.

For two points in a 2D space:

$$
d(p,c)=\sqrt{(p_1-c_1)^2+(p_2-c_2)^2}
$$

Where:

- `p` represents a data point.
- `c` represents a centroid.
- `d` represents the distance between them.

## Objective Function (Inertia)

K-Means tries to minimize the **Within-Cluster Sum of Squares (WCSS)**, also known as **Inertia**.

Inertia measures the sum of squared distances between each data point and its assigned cluster centroid.

$$
\text{Inertia} =
\sum_{j=1}^{K}
\sum_{x_i \in C_j}
\|x_i-\mu_j\|^2
$$

Where:

- `K` is the number of clusters.
- `C_j` represents the data points belonging to cluster `j`.
- `x_i` represents a data point.
- `μ_j` represents the centroid of cluster `j`.

**Lower Inertia** means that data points are closer to their assigned centroids.

## When Should We Use K-Means?

K-Means is useful for several real-world applications:

- **Customer Segmentation:** Group customers based on purchasing behavior and interests.
- **Market Analysis:** Identify groups of customers with similar characteristics.
- **Image Compression:** Reduce the number of colors used in an image.
- **Document Clustering:** Group similar documents based on their content.
- **Anomaly Detection:** Identify unusual data points that are far from their assigned cluster centers.

## Advantages

- Simple and easy to understand.
- Fast and computationally efficient for many datasets.
- Easy to implement.
- Works well with compact and well-separated clusters.
- Scales relatively well to large datasets.

## Disadvantages

- Requires choosing the number of clusters (K) before training.
- Sensitive to initial centroid positions.
- Sensitive to outliers.
- Different initializations may produce different results.
- Does not perform well with irregularly shaped or overlapping clusters.
- Features with larger numerical ranges can dominate distance calculations.

## Choosing the Number of Clusters (Elbow Method)

The **Elbow Method** is a technique used to find a suitable number of clusters (K).

Steps:

1. Run K-Means with different values of K.
2. Calculate Inertia for each value of K.
3. Plot the number of clusters (K) against Inertia.
4. Look for the point where the decrease in Inertia starts slowing down.

This point is called the **Elbow Point**.

It provides a useful estimate of the number of clusters, although it does not always produce a clear answer.

## Feature Scaling

Since K-Means relies on distance calculations, feature scaling is important when features have different numerical ranges.

Common scaling techniques:

- **StandardScaler:** Transforms features to have a mean of 0 and a standard deviation of 1.
- **MinMaxScaler:** Scales feature values to a specified range, usually between 0 and 1.

Scaling prevents features with larger numerical ranges from dominating the distance calculations.

## Implementation

This project includes two implementations of K-Means:

### 1. K-Means From Scratch

Implemented using NumPy without relying on Scikit-Learn's K-Means algorithm.

Concepts covered:

- Random Centroid Initialization.
- Euclidean Distance Calculation.
- Assigning Data Points to Clusters.
- Updating Centroids using Mean.
- Iterative Optimization.
- Convergence Checking.

### 2. K-Means Using Scikit-Learn

Implemented using `sklearn.cluster.KMeans`.

Concepts covered:

- Model Initialization using `KMeans()`.
- Training the model using `fit()`.
- Accessing cluster labels using `labels_`.
- Extracting centroids using `cluster_centers_`.
- Visualizing clusters using Matplotlib.

## Libraries Used

- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

## Key Takeaways

- K-Means is an Unsupervised Learning algorithm.
- It groups similar data points into K clusters.
- Each cluster has a centroid representing its center.
- Euclidean Distance is used to assign points to their nearest centroids.
- The algorithm updates centroids using the mean of assigned data points.
- K-Means minimizes Inertia to create compact clusters.
- Choosing K and applying Feature Scaling are important for better clustering results.
