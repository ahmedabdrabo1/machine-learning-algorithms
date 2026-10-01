# K-Nearest Neighbors (KNN)

## Overview

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm that makes predictions based on the closest training samples.

KNN can be used for:

* Classification
* Regression

Unlike algorithms such as Linear Regression, KNN does not learn a mathematical model with parameters such as weights and bias. Instead, it uses the training data when making predictions.

---

## 1. How KNN Works

For a new data point, KNN:

1. Calculates the distance between the new point and every training point.
2. Sorts the training points by distance.
3. Selects the `K` nearest neighbors.
4. Uses those neighbors to make the prediction.

For classification, KNN uses majority voting.

For regression, KNN uses the average of the neighbors' target values.

---

## 2. Euclidean Distance

The most common distance used with KNN is Euclidean distance.

For two points:

```text
p1 = [x1, x2, ..., xn]
p2 = [y1, y2, ..., yn]
```

the Euclidean distance is:

```text
distance = sqrt(
    (x1 - y1)^2 +
    (x2 - y2)^2 +
    ...
    (xn - yn)^2
)
```

For example:

```text
p1 = [1, 2]
p2 = [4, 6]
```

The distance is:

```text
sqrt((1 - 4)^2 + (2 - 6)^2)
= sqrt(9 + 16)
= 5
```

---

## 3. KNN Classification

In classification, the neighbors vote for the predicted class.

Example:

```text
K = 3

Neighbor 1 → Class 0
Neighbor 2 → Class 0
Neighbor 3 → Class 1
```

The votes are:

```text
Class 0 → 2 votes
Class 1 → 1 vote
```

Therefore:

```text
Prediction → Class 0
```

The implementation from scratch is available in:

```text
01-knn-scratch.py
```

---

## 4. KNN with Scikit-Learn

Scikit-learn provides:

```python
from sklearn.neighbors import KNeighborsClassifier
```

Example:

```python
model = KNeighborsClassifier(n_neighbors=3)

model.fit(x_train, y_train)

prediction = model.predict(new_point)
```

`n_neighbors` determines the value of `K`.

---

## 5. Choosing K

`K` determines how many neighbors are used.

### Small K

A small value such as:

```text
K = 1
```

makes the model highly dependent on the closest points.

Advantages:

* Very local decisions

Disadvantage:

* More sensitive to noise

### Large K

A larger value considers more training samples.

Advantages:

* Less sensitive to individual noisy points

Disadvantage:

* Local patterns can become less important

The appropriate value of `K` depends on the dataset and should be evaluated rather than assumed.

---

## 6. Feature Scaling

Feature scaling is important for KNN because KNN relies on distance.

Consider:

```text
Age    Salary
20     30000
25     50000
30     80000
```

Salary has much larger numerical values than Age.

Without scaling, the Salary difference can dominate the distance calculation.

---

## 7. Min-Max Scaling

Min-Max Scaling transforms a feature into a specified range, commonly `[0, 1]`.

The formula is:

```text
x_scaled = (x - min) / (max - min)
```

Example:

```text
Age = [20, 25, 30]
```

Minimum:

```text
20
```

Maximum:

```text
30
```

After scaling:

```text
20 → 0
25 → 0.5
30 → 1
```

Scaling is performed independently for each feature.

---

## 8. Scikit-Learn Feature Scaling

Scikit-learn provides:

```python
from sklearn.preprocessing import MinMaxScaler
```

Example:

```python
scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(x_train)
```

For new or test data:

```python
new_point_scaled = scaler.transform(new_point)
```

It is important to use the same scaler learned from the training data.

### `fit()`

Learns the required scaling information from the data.

### `transform()`

Applies the learned scaling.

### `fit_transform()`

Performs both operations.

---

## 9. KNN Regression

KNN can also be used for regression.

Scikit-learn provides:

```python
from sklearn.neighbors import KNeighborsRegressor
```

Example:

```text
X:

1
2
3
4
5

Y:

10
20
30
40
50
```

For:

```text
new point = 2.5
K = 2
```

The nearest neighbors are:

```text
2 → 20
3 → 30
```

The prediction is:

```text
(20 + 30) / 2 = 25
```

---

## 10. Weighted KNN

By default, KNN uses uniform weights.

```python
KNeighborsRegressor(
    n_neighbors=2
)
```

Every neighbor has the same influence.

KNN can also give more influence to closer neighbors:

```python
KNeighborsRegressor(
    n_neighbors=2,
    weights="distance"
)
```

With distance weighting, closer neighbors have greater influence.

Example:

```text
Point = 2.1

Neighbor:
2 → 20
Distance = 0.1

Neighbor:
3 → 30
Distance = 0.9
```

The neighbor at `2` has a much larger influence because it is closer.

In the experiment:

```text
Uniform prediction  → 25
Weighted prediction → 21
```

---

## 11. KNN From Scratch

The scratch implementation in:

```text
01-knn-scratch.py
```

implements the main KNN classification steps manually:

```text
Calculate distance
        ↓
Calculate all distances
        ↓
Sort distances
        ↓
Select K neighbors
        ↓
Count votes
        ↓
Return prediction
```

This implementation was created to understand the algorithm before using the scikit-learn implementation.

---

## 12. Project Files

```text
02-knn/
│
├── 01-knn-scratch.py
├── 02-knn-sklearn.py
├── 03-feature-scaling-scratch.py
├── 04-feature-scaling-sklearn.py
├── 05-knn-with-scaling.py
├── 06-knn-regression.py
└── README.md
```

### File descriptions

| File                            | Purpose                                |
| ------------------------------- | -------------------------------------- |
| `01-knn-scratch.py`             | KNN Classification from scratch        |
| `02-knn-sklearn.py`             | KNN Classification with scikit-learn   |
| `03-feature-scaling-scratch.py` | Min-Max Scaling from scratch           |
| `04-feature-scaling-sklearn.py` | Min-Max Scaling with scikit-learn      |
| `05-knn-with-scaling.py`        | Applying scaling to KNN                |
| `06-knn-regression.py`          | KNN Regression and weighted regression |

---

## 13. Advantages

* Simple concept
* Easy to implement
* Can be used for classification and regression
* No explicit training model is required
* Works naturally with multi-dimensional feature spaces

---

## 14. Disadvantages

* Prediction can become expensive with large datasets because distances must be considered.
* Sensitive to feature scaling.
* Sensitive to the choice of `K`.
* Can be affected by irrelevant features and noisy data.
* Performance can decrease as the number of dimensions increases.

---

## 15. Key Takeaways

```text
KNN is a supervised learning algorithm.

Classification:
→ Majority voting

Regression:
→ Average of neighbor values

K:
→ Number of neighbors used

Distance:
→ Determines which points are neighbors

Feature Scaling:
→ Important because KNN uses distance

Min-Max Scaling:
→ Maps features to a comparable range

Weights:
→ uniform = equal influence
→ distance = closer neighbors have more influence
```

The goal of this project is to understand KNN from the inside by implementing important parts manually and then comparing them with scikit-learn implementations.
