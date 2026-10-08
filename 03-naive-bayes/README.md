# Naive Bayes

Naive Bayes is a supervised machine learning algorithm primarily used for classification tasks. It is based on Bayes' Theorem and assumes that features are conditionally independent given the class.

## 1. Bayes' Theorem

Bayes' Theorem is the mathematical foundation of the Naive Bayes algorithm:

`P(A | B) = P(B | A) × P(A) / P(B)`

Where:

* `P(A | B)` — Posterior probability
* `P(B | A)` — Likelihood
* `P(A)` — Prior probability
* `P(B)` — Evidence

In classification, Naive Bayes calculates a probability score for each possible class and selects the class with the highest score.

## 2. Naive Independence Assumption

The term **"Naive"** refers to the assumption that features are conditionally independent given the class.

For an email containing the features `FREE` and `WIN`:

`P(FREE, WIN | Spam) ≈ P(FREE | Spam) × P(WIN | Spam)`

This assumption simplifies the probability calculations and makes the algorithm computationally efficient.

## 3. Manual Calculation Example

Consider a dataset containing **8 emails**:

* `3` Spam emails
* `5` Not Spam emails

For a new email containing both `FREE` and `WIN`:

### Class Priors

`P(Spam) = 3 / 8 = 0.375`

`P(NotSpam) = 5 / 8 = 0.625`

### Feature Likelihoods

For Spam:

`P(FREE | Spam) = 2 / 3`

`P(WIN | Spam) = 2 / 3`

For Not Spam:

`P(FREE | NotSpam) = 1 / 5`

`P(WIN | NotSpam) = 1 / 5`

### Class Scores

Spam:

`P(Spam) × P(FREE | Spam) × P(WIN | Spam)`

`= 0.375 × (2/3) × (2/3)`

`≈ 0.1667`

Not Spam:

`P(NotSpam) × P(FREE | NotSpam) × P(WIN | NotSpam)`

`= 0.625 × (1/5) × (1/5)`

`= 0.025`

### Final Comparison

`Spam Score = 0.1667`

`Not Spam Score = 0.0250`

Since:

`0.1667 > 0.0250`

the predicted class is:

**Spam**

After normalization:

`P(Spam | FREE, WIN) ≈ 86.96%`

`P(NotSpam | FREE, WIN) ≈ 13.04%`

## 4. Laplace Smoothing

A zero frequency for a feature can result in a zero probability. Since Naive Bayes multiplies feature probabilities, a single zero probability can make the entire class score zero.

Laplace smoothing addresses this problem:

`P(feature | class) = (count + 1) / (total + k)`

Where:

* `count` — Number of occurrences of the feature value
* `total` — Total observations in the class
* `k` — Number of possible feature values

For example, if `WIN = Yes` occurs `0` times among `5` Not Spam emails:

Without smoothing:

`P(WIN | NotSpam) = 0 / 5 = 0`

With Laplace smoothing for a binary feature:

`P(WIN | NotSpam) = (0 + 1) / (5 + 2) = 1/7 ≈ 0.1429`

This prevents the class probability from becoming zero.

## 5. Implementations

### From Scratch

**File:** `01-naive-bayes-from-scratch.py`

This implementation demonstrates the fundamental Naive Bayes calculations manually, including:

* Class prior probabilities
* Feature likelihoods
* Naive independence assumption
* Laplace smoothing
* Class score calculation
* Probability normalization
* Final prediction

The example uses binary email features (`FREE` and `WIN`) to classify an email as **Spam** or **Not Spam**.

### Scikit-learn

**File:** `02-naive-bayes-sklearn.py`

This implementation uses Scikit-learn's `BernoulliNB`, which is appropriate for binary features.

It demonstrates:

* Model initialization
* Model training
* Prediction
* Class probability estimation
* Class prior inspection

## 6. Project Structure

```text
03-naive-bayes/
├── 01-naive-bayes-from-scratch.py
├── 02-naive-bayes-sklearn.py
└── README.md
```

## 7. Key Concepts

* Bayes' Theorem
* Prior Probability
* Likelihood
* Posterior Probability
* Conditional Independence
* Laplace Smoothing
* Bernoulli Naive Bayes
* Classification
