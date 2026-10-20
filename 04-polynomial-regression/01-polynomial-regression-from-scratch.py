
import numpy as np

X = np.array([1, 2, 3, 4], dtype=float)
y = np.array([6, 11, 18, 27], dtype=float)

print(X.shape)
print(y.shape)

X_poly = np.column_stack([X, X**2])

print(X_poly.shape)

w = np.ones(X_poly.shape[1])
b = 1.0
lr = 0.01

print(w.shape)

bestMse = []

for i in range(100000):
  # Prediction
  y_pred = (X_poly @ w) + b

  # Error
  error = y - y_pred

  # Mean Squared Error
  mse = (1 / len(X)) * np.sum(error**2)

  # Gradients
  wd = -(2 / len(X)) * (X_poly.T @ error)
  bd = -(2 / len(X)) * np.sum(error)

  # Update weights and bias
  w = w - lr * wd
  b = b - lr * bd

  bestMse.append({
    "mse" : mse,
    "w": w,
    "b": b
  })


print(bestMse[-1])
