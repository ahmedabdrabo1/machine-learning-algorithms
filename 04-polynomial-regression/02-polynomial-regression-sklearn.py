import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.array([1, 2, 3, 4], dtype=float).reshape(-1, 1)
y = np.array([6, 11, 18, 27], dtype=float)

print(X)
print(y)

poly = PolynomialFeatures(degree=2, include_bias=False)

X = poly.fit_transform(X)

print(X)

model = LinearRegression()

model.fit(X,y)

y_pred = model.predict(X)

print("Actual values:", y)
print("Predicted values:", y_pred)

mse = mean_squared_error(y, y_pred)
print("MSE:", mse)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

new_X = np.array([[5.0]])
new_X_poly = poly.transform(new_X)

new_y_pred = model.predict(new_X_poly)
print("Prediction for x = 5:", new_y_pred[0])