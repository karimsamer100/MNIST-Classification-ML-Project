import numpy as np


class KNN:
    def __init__(self, k=3):
        # k must be a positive integer
        if k <= 0:
            raise ValueError("k must be greater than 0")

        self.k = k

        # training data will be stored here
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # basic check to make sure X and y match
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        # k should not be bigger than number of training samples
        if self.k > len(X):
            raise ValueError("k cannot be greater than number of training samples")

        # KNN does not actually train
        # it just stores the training data
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        # calculate Euclidean distance between two points
        distance = np.sqrt(np.sum((x1 - x2) ** 2))
        return distance

    def predict_one(self, x):
        # cannot predict before storing training data
        if self.X_train is None or self.y_train is None:
            raise ValueError("You must call fit before predict")

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
        # cannot predict before storing training data
        if self.X_train is None or self.y_train is None:
            raise ValueError("You must call fit before predict")

        predictions = []

        # predict one sample at a time
        for x in X:
            pred = self.predict_one(x)
            predictions.append(pred)

        return np.array(predictions)
    
#=========================================== Fasel wa nowasel :) ================================================================

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        # learning rate for gradient descent
        self.learning_rate = learning_rate

        # number of training iterations
        self.num_iterations = num_iterations

        # weights and bias will be initialized later in fit
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        # sigmoid converts values to range between 0 and 1
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # number of samples and features
        num_samples, num_features = X.shape

        # initialize weights with zeros
        self.weights = np.zeros(num_features)
        self.bias = 0

        # gradient descent loop
        for _ in range(self.num_iterations):
            # linear combination
            z = np.dot(X, self.weights) + self.bias

            # apply sigmoid
            y_predicted = self.sigmoid(z)

            # compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict_proba(self, X):
        # compute probabilities using learned weights
        z = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(z)

        return y_predicted

    def predict(self, X):
        # convert probabilities to class labels
        y_predicted = self.predict_proba(X)
        y_predicted_labels = np.where(y_predicted >= 0.5, 1, 0)

        return y_predicted_labels