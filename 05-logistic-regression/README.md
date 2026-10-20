# Logistic Regression

## Overview

Logistic Regression is a supervised machine learning algorithm used for classification tasks. It estimates the probability that an input belongs to a particular class.

Despite its name, Logistic Regression is primarily used for classification rather than predicting continuous numerical values.

In binary classification, the model predicts the probability of belonging to Class 1, then uses a decision threshold to assign a class label.

## Dataset

This project uses a synthetic dataset generated with NumPy.

```python
np.random.seed(42)

X = np.random.uniform(-3, 3, size=(100, 1))
y = (X[:, 0] > 0).astype(int)
```

The dataset contains 100 samples and one feature.

The target labels follow this rule:

- If `x > 0`, then `y = 1`.
- Otherwise, `y = 0`.

This synthetic dataset provides a simple example for understanding binary classification.

## Mathematical Foundation

### 1. Linear Score

The model first calculates a linear score:

$$
z = Xw + b
$$

Where:

- $X$: Input features.
- $w$: Model weights.
- $b$: Bias.
- $z$: Linear score before applying the Sigmoid function.

### 2. Sigmoid Function

The Sigmoid function converts the linear score into a probability between 0 and 1.

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The predicted probability is:

$$
p = \sigma(z)
$$

For example, when $z = 0$:

$$
p = \frac{1}{1 + e^0} = 0.5
$$

The model therefore predicts a 50% probability of belonging to Class 1.

### 3. Decision Threshold

The predicted probability is converted into a class label using a threshold.

With a threshold of 0.5:

- If $p \geq 0.5$, predict Class 1.
- If $p < 0.5$, predict Class 0.

The threshold can be adjusted depending on the requirements of the classification problem.

### 4. Binary Cross-Entropy (BCE)

Binary Cross-Entropy measures the difference between actual labels and predicted probabilities.

For one training example:

$$
L = -[y\ln(p) + (1-y)\ln(1-p)]
$$

Where:

- $y$: Actual binary label.
- $p$: Predicted probability of Class 1.
- $L$: Loss for one example.

For $n$ training examples, the mean BCE loss is:

$$
J(w,b) = -\frac{1}{n}\sum_{i=1}^{n}
\left[
y_i\ln(p_i) + (1-y_i)\ln(1-p_i)
\right]
$$

A lower loss indicates better predicted probabilities on the evaluated dataset.

### 5. Gradient Descent

Gradient Descent updates the model parameters to reduce the loss.

For Logistic Regression with a Sigmoid output and Binary Cross-Entropy loss, the gradient with respect to the linear score is:

$$
\frac{\partial L}{\partial z} = p-y
$$

Using the Chain Rule, the gradients for one example are:

$$
\frac{\partial L}{\partial w} = (p-y)x
$$

$$
\frac{\partial L}{\partial b} = p-y
$$

For a dataset with $n$ examples, the average gradients are:

$$
\frac{\partial J}{\partial w} = \frac{1}{n}X^T(p-y)
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}(p_i-y_i)
$$

The parameters are updated using:

$$
w := w-\alpha\frac{\partial J}{\partial w}
$$

$$
b := b-\alpha\frac{\partial J}{\partial b}
$$

Where $\alpha$ is the learning rate.

The training process repeats the forward pass, gradient calculation, and parameter updates over multiple iterations.

## Numerical Example

Consider the following two training examples:

| x | y |
|---:|---:|
| -1 | 0 |
| 1 | 1 |

Initialize the parameters:

$$
w = 0,\qquad b = 0
$$

Assume a learning rate of:

$$
\alpha = 0.1
$$

### Step 1: Calculate the Predictions

The initial linear scores are:

$$
z = wx+b = 0
$$

Therefore, the predicted probabilities are:

$$
p = \sigma(0) = 0.5
$$

The predictions are $[0.5, 0.5]$.

### Step 2: Calculate the Initial Loss

Both samples initially have a predicted probability of 0.5. The mean BCE loss is:

$$
J = -\ln(0.5) \approx 0.693147
$$

### Step 3: Calculate the Gradients

The prediction errors are:

$$
p-y = [0.5, -0.5]
$$

The average weight gradient is:

$$
\frac{\partial J}{\partial w}
= \frac{(0.5)(-1)+(-0.5)(1)}{2}
= -0.5
$$

The average bias gradient is:

$$
\frac{\partial J}{\partial b}
= \frac{0.5+(-0.5)}{2}
= 0
$$

### Step 4: Update the Parameters

Update the weight:

$$
w_{\text{new}} = 0-0.1(-0.5)=0.05
$$

Update the bias:

$$
b_{\text{new}} = 0-0.1(0)=0
$$

After one update:

$$
w = 0.05,\qquad b = 0
$$

The new mean BCE loss is approximately 0.6685, which is lower than the initial loss of 0.6931.

This example demonstrates how the gradients guide the parameter updates.

## Implementation

### 1. From Scratch

File: `01-logistic-regression-from-scratch.py`

This implementation uses NumPy to build Logistic Regression without relying on a machine learning estimator.

Main steps:

1. Generate the synthetic dataset.
2. Initialize the weights and bias.
3. Implement the Sigmoid function.
4. Calculate predicted probabilities.
5. Calculate Binary Cross-Entropy.
6. Calculate the weight and bias gradients.
7. Update the parameters using Gradient Descent.
8. Repeat the training process.
9. Evaluate the model using Accuracy.

This implementation helps explain the mathematical operations involved in training Logistic Regression.

### 2. Using Scikit-learn

File: `02-logistic-regression-sklearn.py`

This implementation uses Scikit-learn's `LogisticRegression` estimator.

Main steps:

1. Generate the same synthetic dataset.
2. Create the Logistic Regression model.
3. Train the model using `model.fit(X, y)`.
4. Predict class labels using `model.predict(X)`.
5. Predict class probabilities using `model.predict_proba(X)`.
6. Evaluate the model using Accuracy, Log Loss, and a Confusion Matrix.
7. Inspect the learned coefficients and intercept.

Scikit-learn handles parameter optimization internally, reducing the amount of code required.

## Evaluation Metrics

### 1. Accuracy

Accuracy measures the proportion of correctly classified examples.

$$
\text{Accuracy} =
\frac{\text{Number of Correct Predictions}}
{\text{Total Number of Predictions}}
$$

### 2. Log Loss

Log Loss evaluates predicted probabilities using the Binary Cross-Entropy formula.

Lower Log Loss generally indicates better probability predictions on the evaluated dataset.

### 3. Confusion Matrix

A Confusion Matrix compares actual labels with predicted labels.

For binary classification, it contains:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

These values help explain the types of classification errors made by the model.

## From Scratch vs. Scikit-learn

| Aspect | From Scratch | Scikit-learn |
|---|---|---|
| Sigmoid function | Implemented with NumPy | Handled internally |
| Loss evaluation | Custom BCE function | `log_loss` for evaluation |
| Gradient calculation | Implemented manually | Handled internally |
| Parameter optimization | Manual Gradient Descent | Built-in optimization algorithm |
| Predictions | NumPy operations | `predict()` and `predict_proba()` |
| Evaluation | NumPy calculations | Scikit-learn metrics |
| Main purpose | Understand the mathematics | Apply the algorithm efficiently |

The learned parameters and loss values may differ because Scikit-learn applies regularization by default, while the From Scratch implementation uses the specified learning rate and number of iterations.

## Limitations

- The dataset is synthetic and follows a simple classification rule.
- Evaluating the model on training data does not measure its performance on unseen data.
- Logistic Regression learns a linear decision boundary in the original feature space.
- The learning rate and number of iterations affect the From Scratch implementation.
- Scikit-learn's default regularization affects the learned coefficients.
- Real-world datasets may require feature engineering, train/test splitting, and additional evaluation metrics.

## Key Takeaways

- Logistic Regression is a supervised classification algorithm.
- The Sigmoid function converts linear scores into probabilities.
- Binary Cross-Entropy measures errors in predicted probabilities.
- The Chain Rule is used to derive gradients for the model parameters.
- Gradient Descent updates the weights and bias to reduce the loss.
- NumPy can be used to implement Logistic Regression from scratch.
- Scikit-learn provides a convenient implementation with built-in optimization and evaluation tools.

## Project Structure

```text
05-logistic-regression/
├── 01-logistic-regression-from-scratch.py
├── 02-logistic-regression-sklearn.py
└── README.md
```

## Requirements

- Python 3
- NumPy
- Scikit-learn

Install the required libraries:

```bash
pip install numpy scikit-learn
```

## Learning Objective

The objective of this project is to understand the mathematics of Logistic Regression, derive gradients using the Chain Rule, implement Gradient Descent with NumPy, and compare the manual implementation with Scikit-learn.