import numpy as np


class KNN:
    def __init__(self, k=3):
        # number of nearest neighbors
        self.k = k

        # training data will be stored here
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # KNN does not actually train
        # it just stores the training data
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        # calculate Euclidean distance between two points
        distance = np.sqrt(np.sum((x1 - x2) ** 2))
        return distance

    def predict_one(self, x):
        distances = []

        # calculate distance between x and every training sample
        for i in range(len(self.X_train)):
            distance = self.euclidean_distance(x, self.X_train[i])
            distances.append((distance, self.y_train[i]))

        # sort by distance from smallest to largest
        distances.sort(key=lambda item: item[0])

        # take labels of the k nearest neighbors
        k_nearest_labels = []
        for i in range(self.k):
            k_nearest_labels.append(distances[i][1])

        # majority vote
        prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)

        return prediction

    def predict(self, X):
        predictions = []

        # predict one sample at a time
        for x in X:
            pred = self.predict_one(x)
            predictions.append(pred)

        return np.array(predictions)