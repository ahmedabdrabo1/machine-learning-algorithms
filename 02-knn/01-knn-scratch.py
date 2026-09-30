def distance(p1, p2):
    total = 0

    for i in range(len(p1)):
        total += (p1[i] - p2[i]) ** 2

    return total ** 0.5


def get_distances(x_test, x_train, y_train):
    distances = []

    for i in range(len(x_train)):
        dist = distance(x_train[i], x_test)
        distances.append([dist, y_train[i]])

    return distances


def get_neighbors(distances, k):
    return distances[:k]


def predict(neighbors):
    votes = {}

    for neighbor in neighbors:
        label = neighbor[1]

        if label not in votes:
            votes[label] = 1
        else:
            votes[label] += 1

    prediction = max(votes, key=votes.get)

    return prediction


def knn(x_test, x_train, y_train, k):
    distances = get_distances(x_test, x_train, y_train)

    distances.sort()

    neighbors = get_neighbors(distances, k)

    return predict(neighbors)


x_train = [
    [1, 2, 3],
    [3, 4, 5],
    [10, 11, 12],
    [12, 14, 15]
]

y_train = [
    "Class 0",
    "Class 0",
    "Class 1",
    "Class 1"
]

newPoint = [2, 2, 2]

prediction = knn(newPoint, x_train, y_train, 3)

print("Prediction:", prediction)