# Polynomial Regression

## Overview

Polynomial Regression is a supervised machine learning algorithm used to model nonlinear relationships between input features and a target variable.

It extends Linear Regression by transforming the original features into polynomial features, allowing the model to fit curved relationships while remaining linear in its coefficients.

## Dataset

We use a small dataset to understand how Polynomial Regression works.

| x | y |
|---:|---:|
| 1 | 6 |
| 2 | 11 |
| 3 | 18 |
| 4 | 27 |

The relationship between the input and target values follows:

\[
y = 3 + 2x + x^2
\]

## Mathematical Foundation

### 1. Polynomial Model

For a polynomial of degree 2, the prediction equation is:

\[
\hat{y} = b_0 + b_1x + b_2x^2
\]

Where:

- \(b_0\): Intercept.
- \(b_1\): Coefficient of \(x\).
- \(b_2\): Coefficient of \(x^2\).
- \(x\): Input feature.
- \(\hat{y}\): Predicted value.

### 2. Feature Transformation

Polynomial Regression transforms each input \(x\) into polynomial features:

\[
x \rightarrow [x, x^2]
\]

The transformed dataset becomes:

| x | x² | y |
|---:|---:|---:|
| 1 | 1 | 6 |
| 2 | 4 | 11 |
| 3 | 9 | 18 |
| 4 | 16 | 27 |

The model can then be written as:

\[
\hat{y} = b_0 + b_1x + b_2x^2
\]

Although the relationship between \(x\) and \(y\) can be curved, the model is still linear in its coefficients.

### 3. Mean Squared Error (MSE)

Mean Squared Error measures the average squared difference between actual and predicted values.

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

A lower MSE indicates that the predictions are closer to the actual values.

For this dataset, the model can learn the exact relationship, so the theoretical minimum MSE is zero.

## Implementation

### 1. From Scratch

File: `01-polynomial-regression-from-scratch.py`

The model is implemented using NumPy and Gradient Descent.

Main steps:

1. Prepare the input and target arrays.
2. Generate polynomial features \(x\) and \(x^2\).
3. Initialize the weights and bias.
4. Calculate predictions.
5. Calculate the prediction errors and MSE.
6. Calculate the gradients.
7. Update the weights and bias using Gradient Descent.
8. Repeat the process until the model converges.

The prediction function is:

\[
\hat{y} = X_{\text{poly}}w + b
\]

The weight gradient is:

\[
\frac{\partial MSE}{\partial w}
= -\frac{2}{n}X_{\text{poly}}^T(y-\hat{y})
\]

The bias gradient is:

\[
\frac{\partial MSE}{\partial b}
= -\frac{2}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)
\]

The parameters are updated using:

\[
w := w-\alpha\frac{\partial MSE}{\partial w}
\]

\[
b := b-\alpha\frac{\partial MSE}{\partial b}
\]

Where \(\alpha\) is the learning rate.

#### Learned Parameters

The implementation converged approximately to:

- \(b_0 = 3\)
- \(b_1 = 2\)
- \(b_2 = 1\)

The final MSE was approximately \(1.31 \times 10^{-26}\), which is effectively zero for this example.

### 2. Using Scikit-learn

File: `02-polynomial-regression-sklearn.py`

Libraries used:

- `PolynomialFeatures`: Generates polynomial features.
- `LinearRegression`: Fits a linear regression model to the transformed features.
- `mean_squared_error`: Evaluates prediction errors.
- `NumPy`: Handles numerical arrays.

Main steps:

1. Reshape the input data into a two-dimensional array.
2. Generate polynomial features with degree 2.
3. Train the `LinearRegression` model.
4. Generate predictions.
5. Evaluate the model using MSE.
6. Inspect the learned intercept and coefficients.
7. Predict values for new inputs after applying the same polynomial transformation.

The Scikit-learn implementation should learn approximately the same coefficients as the From Scratch implementation.

## Example Prediction

For \(x = 5\):

\[
\hat{y} = 3 + 2(5) + 5^2
\]

\[
\hat{y} = 3 + 10 + 25 = 38
\]

Therefore, the predicted value is:

\[
\boxed{\hat{y}=38}
\]

## From Scratch vs. Scikit-learn

| Aspect | From Scratch | Scikit-learn |
|---|---|---|
| Feature generation | NumPy | `PolynomialFeatures` |
| Parameter optimization | Gradient Descent | Built-in Linear Regression solver |
| Parameter updates | Implemented manually | Handled by the library |
| Evaluation | NumPy calculations | `mean_squared_error` |
| Main purpose | Understand the mathematics | Apply the algorithm efficiently |

## Limitations

- High polynomial degrees can lead to overfitting.
- Polynomial features can become very large as the degree increases.
- Features with large values may cause numerical instability.
- A low training error does not guarantee good performance on unseen data.
- Real-world applications should use suitable validation techniques to select the polynomial degree.

## Key Takeaways

- Polynomial Regression models nonlinear relationships by adding polynomial features.
- The model remains linear in its coefficients.
- Feature transformation is essential when using a linear estimator for polynomial relationships.
- Gradient Descent can be used to learn the parameters from scratch.
- Scikit-learn simplifies implementation and evaluation.
- Model complexity should be selected carefully to avoid overfitting.
