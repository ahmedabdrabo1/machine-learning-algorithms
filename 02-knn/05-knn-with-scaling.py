import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


x_train = np.array([
    [20, 30000],
    [22, 35000],
    [25, 40000],
    [28, 45000],
    [35, 70000],
    [40, 85000],
    [42, 90000],
    [45, 100000]
])

y_train = np.array([
    "Class 0",
    "Class 0",
    "Class 0",
    "Class 0",
    "Class 1",
    "Class 1",
    "Class 1",
    "Class 1"
])

new_point = np.array([
    [30, 50000]
])


# KNN without scaling

model = KNeighborsClassifier(n_neighbors=3)

model.fit(x_train, y_train)

prediction = model.predict(new_point)

print("Prediction without scaling:")
print(prediction)


# Feature Scaling

scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(x_train)

new_point_scaled = scaler.transform(new_point)


print("\nOriginal data:")
print(x_train)

print("\nScaled data:")
print(x_train_scaled)

print("\nScaled new point:")
print(new_point_scaled)


# KNN with scaling

model_scaled = KNeighborsClassifier(n_neighbors=3)

model_scaled.fit(x_train_scaled, y_train)

prediction_scaled = model_scaled.predict(new_point_scaled)

print("\nPrediction with scaling:")
print(prediction_scaled)