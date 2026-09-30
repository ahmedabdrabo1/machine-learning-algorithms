from sklearn.neighbors import KNeighborsClassifier
import numpy as np


x_train = np.array([
    [1, 2, 3],
    [3, 4, 5],
    [10, 11, 12],
    [12, 14, 15]
])

y_train = np.array([
    "Class 0",
    "Class 0",
    "Class 1",
    "Class 1"
])

newPoint = np.array([[2, 2, 2]])

k = 3

model = KNeighborsClassifier(n_neighbors=k)

model.fit(x_train, y_train)

prediction = model.predict(newPoint)

print("Prediction:", prediction)