# X = [1, 2, 3]
# y = [2, 4, 6]

# w = 0
# b = 0

# learning_rate = 0.1
# epochs = 1000

# def predict(x):
#     return w * x + b


# def calculate_mse(X, y):
#     total_error = 0

#     for i in range(len(X)):
#         prediction = predict(X[i])
#         error = y[i] - prediction
#         total_error += error ** 2

#     return total_error / len(X)


# for epoch in range(epochs):

#     dw = 0
#     db = 0

#     for i in range(len(X)):
#         prediction = predict(X[i])
#         error = prediction - y[i]

#         dw += X[i] * error
#         db += error

#     dw = (2 / len(X)) * dw
#     db = (2 / len(X)) * db

#     w = w - learning_rate * dw
#     b = b - learning_rate * db


# print("w:", w)
# print("b:", b)
# print("Prediction for x = 4:", predict(4))
# print("MSE:", calculate_mse(X, y))


import numpy as np


class LinearRegression:

    def __init__(self, lr=0.1, epochs=1000):
        self.epochs = epochs
        self.lr = lr
        self.weights = 0
        self.bias = 0
        self.loss_history = []

    def fit(self, X, Y):

        for _ in range(self.epochs):

            # Predictions
            y_prediction = self.predict(X)

            # Error
            error = y_prediction - Y

            # Gradients
            dw = (2 / len(X)) * np.dot(X, error)

            db = (2 / len(X)) * np.sum(error)

            # Update parameters
            self.weights = self.weights - self.lr * dw

            self.bias = self.bias - self.lr * db

            # Calculate MSE
            mse = np.mean((Y - self.predict(X)) ** 2)

            # Store loss
            self.loss_history.append(mse)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

X = np.array([1, 2, 3, 4, 5])
Y = np.array([50, 55, 65, 70, 80])

model = LinearRegression(lr=0.01, epochs=5000)

model.fit(X, Y)

print("Weight:", model.weights)
print("Bias:", model.bias)

print("First 5 losses:")

for loss in model.loss_history[:5]:
    print(loss)

print("Final loss:", model.loss_history[-1])
