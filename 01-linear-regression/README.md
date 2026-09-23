# Linear Regression

## What is Linear Regression?

Linear Regression is a supervised learning algorithm used to predict a continuous numerical value.

It tries to find a linear relationship between the input features and the target.

### Example

```text
Hours Studied → Exam Score
```

The model can learn that students who study more hours tend to have higher scores.

---

## Mathematical Formula

The basic equation is:

```text
ŷ = wx + b
```

Where:

* `x` = input feature
* `ŷ` = predicted value
* `w` = weight / slope
* `b` = bias / intercept

---

## Mean Squared Error (MSE)

MSE measures how far the predictions are from the actual values.

```text
MSE = (1/n) Σ(y - ŷ)²
```

A lower MSE means the predictions are closer to the actual values.

---

## Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the loss.

The model repeatedly:

```text
1. Makes predictions
2. Calculates the error
3. Calculates gradients
4. Updates weights and bias
5. Repeats
```

The learning rate controls the size of each update.

```text
w = w - learning_rate × dw

b = b - learning_rate × db
```

---

## From Scratch

The Linear Regression algorithm was implemented from scratch using NumPy.

The implementation includes:

* Weight
* Bias
* Prediction
* MSE
* Gradient calculation
* Gradient Descent
* Learning Rate
* Epochs
* Loss History
* `fit()`
* `predict()`

The implementation was tested on sample data and successfully learned the relationship between the input and target.

---

## Scikit-learn

Linear Regression was also implemented using Scikit-learn.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X, y)

prediction = model.predict(X)
```

Important attributes:

```text
model.coef_       → weight / coefficient
model.intercept_  → bias / intercept
```

For Scikit-learn, the feature matrix `X` is represented as:

```text
(samples, features)
```

For example:

```python
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
```

has the shape:

```text
(5, 1)
```

which means:

```text
5 samples
1 feature
```

---

## Scratch vs Scikit-learn

| From Scratch                 | Scikit-learn           |
| ---------------------------- | ---------------------- |
| Implemented manually         | Library implementation |
| Calculate gradients manually | Handled by the library |
| Update weights manually      | Handled by the library |
| `fit()`                      | `fit()`                |
| `predict()`                  | `predict()`            |
| `weights`                    | `coef_`                |
| `bias`                       | `intercept_`           |

The scratch implementation was useful for understanding how Linear Regression works internally, while Scikit-learn provides a convenient implementation for practical use.

---

## What I Learned

* What Regression means
* How Linear Regression works
* The equation `ŷ = wx + b`
* Weight and Bias
* Prediction and Error
* Mean Squared Error
* Gradient Descent
* Learning Rate
* Epochs
* Loss History
* NumPy implementation
* Scikit-learn implementation
* Feature and sample shapes
* Using `reshape(-1, 1)` to prepare a single-feature dataset
* The basic Machine Learning workflow:

```text
Data
 ↓
Model
 ↓
Fit
 ↓
Predict
 ↓
Evaluate
```
