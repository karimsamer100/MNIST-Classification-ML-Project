import numpy as np


class KNN:
    def __init__(self, k=3):
        if k <= 0:
            raise ValueError("k must be greater than 0")

        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        if self.k > len(X):
            raise ValueError("k cannot be greater than number of training samples")

        # KNN just stores the training data
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict_one(self, x):
        if self.X_train is None or self.y_train is None:
            raise ValueError("You must call fit before predict")

        distances = []

        for i in range(len(self.X_train)):
            distance = self.euclidean_distance(x, self.X_train[i])
            distances.append((distance, self.y_train[i]))

        distances.sort(key=lambda item: item[0])

        k_nearest_labels = []
        for i in range(self.k):
            k_nearest_labels.append(distances[i][1])

        # majority vote for multi-class
        values, counts = np.unique(k_nearest_labels, return_counts=True)
        prediction = values[np.argmax(counts)]

        return prediction

    def predict(self, X):
        if self.X_train is None or self.y_train is None:
            raise ValueError("You must call fit before predict")

        predictions = []

        for x in X:
            pred = self.predict_one(x)
            predictions.append(pred)

        return np.array(predictions)


# =========================================================


class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.variance = None
        self.priors = None

    def fit(self, X, y):
        self.classes = np.unique(y)

        num_classes = len(self.classes)
        num_features = X.shape[1]

        self.mean = np.zeros((num_classes, num_features))
        self.variance = np.zeros((num_classes, num_features))
        self.priors = np.zeros(num_classes)

        for index, current_class in enumerate(self.classes):
            X_class = X[y == current_class]

            self.mean[index, :] = np.mean(X_class, axis=0)
            self.variance[index, :] = np.var(X_class, axis=0)
            self.priors[index] = X_class.shape[0] / X.shape[0]

    def gaussian_probability(self, class_index, x):
        epsilon = 1e-9

        mean = self.mean[class_index]
        variance = self.variance[class_index] + epsilon

        numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)

        return numerator / denominator

    def predict_one(self, x):
        class_scores = []

        for class_index, current_class in enumerate(self.classes):
            log_prior = np.log(self.priors[class_index])

            probabilities = self.gaussian_probability(class_index, x)

            log_likelihood = np.sum(np.log(probabilities + 1e-9))

            class_score = log_prior + log_likelihood
            class_scores.append(class_score)

        predicted_class = self.classes[np.argmax(class_scores)]
        return predicted_class

    def predict(self, X):
        predictions = []

        for x in X:
            prediction = self.predict_one(x)
            predictions.append(prediction)

        return np.array(predictions)


# =========================================================

class MulticlassLogisticRegression:
    def __init__(self, learning_rate=0.1, num_iterations=1000, lambda_reg=0.0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_reg = lambda_reg  # L2 regularization strength

        self.weights = None
        self.bias = None
        self.classes = None

    def softmax(self, z):
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_values = np.exp(z_shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def one_hot_encode(self, y):
        num_samples = len(y)
        num_classes = len(self.classes)

        y_one_hot = np.zeros((num_samples, num_classes))

        for i, label in enumerate(y):
            class_index = np.where(self.classes == label)[0][0]
            y_one_hot[i, class_index] = 1

        return y_one_hot

    def fit(self, X, y):
        num_samples, num_features = X.shape

        self.classes = np.unique(y)
        num_classes = len(self.classes)

        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros((1, num_classes))

        y_one_hot = self.one_hot_encode(y)

        for _ in range(self.num_iterations):
            scores = np.dot(X, self.weights) + self.bias
            probabilities = self.softmax(scores)

            # gradients (with L2 regularization)
            dw = (1 / num_samples) * np.dot(X.T, (probabilities - y_one_hot))
            dw += (self.lambda_reg / num_samples) * self.weights  

            db = (1 / num_samples) * np.sum(probabilities - y_one_hot, axis=0, keepdims=True)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        scores = np.dot(X, self.weights) + self.bias
        return self.softmax(scores)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes[predicted_indices]