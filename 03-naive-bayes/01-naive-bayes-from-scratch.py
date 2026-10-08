import numpy as np


# 1 = Yes
# 0 = No

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


# Calculate class priors

spam_count = np.sum(y == 1)
not_spam_count = np.sum(y == 0)

total = len(y)

p_spam = spam_count / total
p_not_spam = not_spam_count / total


# Calculate feature probability using Laplace smoothing

def probability(feature, class_value):
    class_samples = X[y == class_value]

    count = np.sum(class_samples[:, feature] == 1)
    total = len(class_samples)

    k = 2

    return (count + 1) / (total + k)


# Feature probabilities

p_free_spam = probability(0, 1)
p_win_spam = probability(1, 1)

p_free_not_spam = probability(0, 0)
p_win_not_spam = probability(1, 0)


# Calculate class scores

spam_score = (
    p_spam
    * p_free_spam
    * p_win_spam
)

not_spam_score = (
    p_not_spam
    * p_free_not_spam
    * p_win_not_spam
)


# Normalize scores

total_score = spam_score + not_spam_score

spam_probability = spam_score / total_score
not_spam_probability = not_spam_score / total_score


# Prediction

if spam_probability > not_spam_probability:
    prediction = "Spam"
else:
    prediction = "Not Spam"


print("P(Spam):", p_spam)
print("P(Not Spam):", p_not_spam)

print()

print("P(FREE | Spam):", p_free_spam)
print("P(WIN | Spam):", p_win_spam)

print("P(FREE | Not Spam):", p_free_not_spam)
print("P(WIN | Not Spam):", p_win_not_spam)

print()

print("Spam score:", spam_score)
print("Not Spam score:", not_spam_score)

print()

print("Spam probability:", spam_probability)
print("Not Spam probability:", not_spam_probability)

print()

print("Prediction:", prediction)