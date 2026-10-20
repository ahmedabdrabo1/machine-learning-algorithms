
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

np.random.seed(42)

X = np.random.uniform(-3, 3, size=(100, 1))
y = (X[:, 0] > 0).astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)


initial_probabilities = np.full(len(y), 0.5)

initial_loss = log_loss(y, initial_probabilities)

print("\nInitial Loss:", initial_loss)

model = LogisticRegression()


model.fit(X, y)

# Predict classes: 0 or 1
y_pred = model.predict(X)

# Predict probability of class 1
y_prob = model.predict_proba(X)[:, 1]

accuracy = accuracy_score(y, y_pred)

final_loss = log_loss(y, y_prob)

conf_matrix = confusion_matrix(y, y_pred)

w = model.coef_
b = model.intercept_

print("\nFinal Loss:", final_loss)

print("Weights:", w)
print("Bias:", b)

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)

print("\nConfusion Matrix:")
print(conf_matrix)

