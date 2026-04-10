from data_module import (
    load_mnist_csv,
    filter_binary_classes,
    normalize_pixels,
    split_train_validation
)

import numpy as np
from models_module import KNN, LogisticRegression, GaussianNaiveBayes

# load original training data
X_train_full, y_train_full = load_mnist_csv("MNIST-data/mnist_train.csv")

# keep only digits 0 and 1
X_train_full, y_train_full = filter_binary_classes(X_train_full, y_train_full)

# normalize
X_train_full = normalize_pixels(X_train_full)

# split into train and validation
X_train, X_val, y_train, y_val = split_train_validation(X_train_full, y_train_full)

print("Train set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Train labels shape:", y_train.shape)
print("Validation labels shape:", y_val.shape)

print("Train unique labels:", set(y_train))
print("Validation unique labels:", set(y_val))
print("Train min pixel:", X_train_full.min())
print("Train max pixel:", X_train_full.max())

# load original test data
X_test, y_test = load_mnist_csv("MNIST-data/mnist_test.csv")

# keep only digits 0 and 1
X_test, y_test = filter_binary_classes(X_test, y_test)

# normalize
X_test = normalize_pixels(X_test)

print("Test set shape:", X_test.shape)
print("Test labels shape:", y_test.shape)
print("Test unique labels:", set(y_test))
print("Test min pixel:", X_test.min())
print("Test max pixel:", X_test.max())

# test KNN fit only
knn = KNN(k=3)
knn.fit(X_train, y_train)

print("\nKNN test:")
print("k value:", knn.k)
print("Stored train shape:", knn.X_train.shape)
print("Stored labels shape:", knn.y_train.shape)

# test Euclidean distance
distance = knn.euclidean_distance(X_train[0], X_train[1])

print("\nDistance test:")
print("Distance between first two training samples:", distance)

# test predict_one on a single validation sample
single_prediction = knn.predict_one(X_val[0])

print("\nSingle prediction test:")
print("Predicted label:", single_prediction)
print("Actual label:", y_val[0])

# test predict on a small validation subset
small_predictions = knn.predict(X_val[:5])

print("\nMultiple predictions test:")
print("Predictions:", small_predictions)
print("Actual labels:", y_val[:5])

# random subsets for stronger testing
rng = np.random.default_rng(42)

train_indices = rng.choice(len(X_train), size=2000, replace=False)
val_indices = rng.choice(len(X_val), size=300, replace=False)
test_indices = rng.choice(len(X_test), size=300, replace=False)

X_train_subset = X_train[train_indices]
y_train_subset = y_train[train_indices]

X_val_subset = X_val[val_indices]
y_val_subset = y_val[val_indices]

X_test_subset = X_test[test_indices]
y_test_subset = y_test[test_indices]

print("\nRandom subset class counts:")
print("Train subset class counts:", np.bincount(y_train_subset))
print("Validation subset class counts:", np.bincount(y_val_subset))
print("Test subset class counts:", np.bincount(y_test_subset))


#========================

# test Gaussian Naive Bayes initialization
gnb = GaussianNaiveBayes()

print("\nGaussian Naive Bayes test:")
print("Classes:", gnb.classes)
print("Mean:", gnb.mean)
print("Variance:", gnb.variance)
print("Priors:", gnb.priors)

# test Gaussian Naive Bayes fit
gnb.fit(X_train_subset, y_train_subset)

print("\nGaussian Naive Bayes fit test:")
print("Classes:", gnb.classes)
print("Mean shape:", gnb.mean.shape)
print("Variance shape:", gnb.variance.shape)
print("Priors:", gnb.priors)
# test Gaussian probability on one sample
gaussian_probs = gnb.gaussian_probability(0, X_val_subset[0])

print("\nGaussian probability test:")
print("Shape:", gaussian_probs.shape)
print("First 10 values:", gaussian_probs[:10])

# test predict_one for Gaussian Naive Bayes
gnb_single_prediction = gnb.predict_one(X_val_subset[0])

print("\nGaussian Naive Bayes single prediction test:")
print("Predicted label:", gnb_single_prediction)
print("Actual label:", y_val_subset[0])

# test Gaussian Naive Bayes on multiple samples
gnb_predictions = gnb.predict(X_val_subset[:20])

print("\nGaussian Naive Bayes multiple predictions test:")
print("Predictions:", gnb_predictions)
print("Actual labels:", y_val_subset[:20])

gnb_accuracy = np.mean(gnb_predictions == y_val_subset[:20])
print("Accuracy on first 20 validation samples:", gnb_accuracy)

# bigger Gaussian Naive Bayes test on random validation subset
gnb = GaussianNaiveBayes()
gnb.fit(X_train_subset, y_train_subset)

gnb_val_predictions = gnb.predict(X_val_subset)

print("\nGaussian Naive Bayes bigger random subset test:")
print("Predictions shape:", gnb_val_predictions.shape)
print("First 20 predictions:", gnb_val_predictions[:20])
print("First 20 actual labels:", y_val_subset[:20])

gnb_val_accuracy = np.mean(gnb_val_predictions == y_val_subset)
print("Validation accuracy:", gnb_val_accuracy)

# random test subset check
gnb_test_predictions = gnb.predict(X_test_subset)
gnb_test_accuracy = np.mean(gnb_test_predictions == y_test_subset)

print("\nGaussian Naive Bayes random test subset check:")
print("Test accuracy:", gnb_test_accuracy)
print("First 20 test predictions:", gnb_test_predictions[:20])
print("First 20 actual test labels:", y_test_subset[:20])