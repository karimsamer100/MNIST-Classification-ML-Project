import numpy as np


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
    

# =========================================================

class NearestCentroidClassifier:
    def __init__(self):
        self.classes = None
        self.centroids = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        num_features = X.shape[1]

        self.centroids = np.zeros((num_classes, num_features))

        for idx, cls in enumerate(self.classes):
            X_class = X[y == cls]
            self.centroids[idx] = np.mean(X_class, axis=0)

    def predict_one(self, x):
        distances = np.sum((self.centroids - x) ** 2, axis=1)
        closest_index = np.argmin(distances)
        return self.classes[closest_index]

    def predict(self, X):
        predictions = []

        for x in X:
            predictions.append(self.predict_one(x))

        return np.array(predictions)