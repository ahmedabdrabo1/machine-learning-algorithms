
import numpy as np

np.random.seed(42)

X = np.random.uniform(-3, 3, size=(100, 1))
y = (X[:, 0] > 0).astype(float)

w = np.zeros(X.shape[1])
b = 0.0

lr = 0.01
epochs = 1000


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y, y_pred):
  loss = y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
  return -np.mean(loss)



z = X @ w + b
y_pred = sigmoid(z).ravel()

initial_loss = binary_cross_entropy(y, y_pred)

history = []

for i in range(epochs):

    # Forward Pass
    z = X @ w + b
    y_pred = sigmoid(z).ravel()

    # Calculate Error
    error = y_pred - y

    # Calculate Gradients
    dw = (X.T @ error) / len(y)
    db = np.mean(error)

    # Update Parameters
    w = w - lr * dw
    b = b - lr * db

    # Evaluate Updated Model
    z = X @ w + b
    y_pred = sigmoid(z).ravel()

    loss = binary_cross_entropy(y, y_pred)

    # Save Training History
    history.append({
        "iteration": i + 1,
        "loss": float(loss),
        "w": w.copy(),
        "b": float(b)
    })


z = X @ w + b
y_pred = sigmoid(z).ravel()

y_pred_class = (y_pred >= 0.5).astype(float)

accuracy = np.mean(y_pred_class == y)


print("Final Loss:", history[-1]["loss"])
print("Final Weights:", w)
print("Final Bias:", b)

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)

