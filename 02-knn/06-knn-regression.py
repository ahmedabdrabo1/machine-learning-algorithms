import numpy as np

from sklearn.neighbors import KNeighborsRegressor


x_train = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y_train = np.array([
    10,
    20,
    30,
    40,
    50
])


new_point = np.array([
    [2.1]
])


k = 2

model = KNeighborsRegressor(n_neighbors=k)

model.fit(x_train, y_train)

prediction = model.predict(new_point)

print("Prediction:", prediction)


weighted_model = KNeighborsRegressor(
    n_neighbors=2,
    weights="distance"
)

weighted_model.fit(x_train, y_train)

weighted_prediction = weighted_model.predict(new_point)

print("Weighted prediction:", weighted_prediction)