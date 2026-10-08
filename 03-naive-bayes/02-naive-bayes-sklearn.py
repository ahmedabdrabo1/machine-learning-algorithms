import numpy as np
from sklearn.naive_bayes import BernoulliNB


# Features:
# [FREE, WIN]

X = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 0],
    [0, 0]
])


# 1 = Spam
# 0 = Not Spam

y = np.array([
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0
])


# Create model

model = BernoulliNB()


# Train model

model.fit(X, y)


# New email: FREE = Yes, WIN = Yes

new_email = np.array([[1, 1]])


# Prediction

prediction = model.predict(new_email)

print("Prediction:", prediction)


# Class probabilities

probabilities = model.predict_proba(new_email)

print("Probabilities:", probabilities)


# Classes

print("Classes:", model.classes_)