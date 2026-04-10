from data_module import (
    load_mnist_csv,
    filter_binary_classes,
    normalize_pixels,
    split_train_validation
)
from models_module import KNN
import numpy as np
from models_module import KNN, LogisticRegression

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

# bigger random subset test
knn = KNN(k=3)
knn.fit(X_train_subset, y_train_subset)

predictions = knn.predict(X_val_subset)

print("\nBigger random subset test:")
print("Predictions shape:", predictions.shape)
print("First 20 predictions:", predictions[:20])
print("First 20 actual labels:", y_val_subset[:20])

accuracy = np.mean(predictions == y_val_subset)
print("Validation accuracy with k=3:", accuracy)

# try more than one k value
print("\nTesting different k values on random validation subset:")

k_values = [1, 3, 5, 7, 9]

for k in k_values:
    knn = KNN(k=k)
    knn.fit(X_train_subset, y_train_subset)

    predictions = knn.predict(X_val_subset)
    accuracy = np.mean(predictions == y_val_subset)

    print(f"k = {k}, validation accuracy = {accuracy}")

# final random test subset check
knn = KNN(k=3)
knn.fit(X_train_subset, y_train_subset)

test_predictions = knn.predict(X_test_subset)
test_accuracy = np.mean(test_predictions == y_test_subset)

print("\nRandom test subset check:")
print("Test accuracy with k=3:", test_accuracy)
print("First 20 test predictions:", test_predictions[:20])
print("First 20 actual test labels:", y_test_subset[:20])
#=======================================================

# test Logistic Regression initialization
log_reg = LogisticRegression(learning_rate=0.01, num_iterations=1000)

print("\nLogistic Regression test:")
print("Learning rate:", log_reg.learning_rate)
print("Number of iterations:", log_reg.num_iterations)

# test sigmoid function
sigmoid_result = log_reg.sigmoid(0)
print("Sigmoid(0):", sigmoid_result)


# train Logistic Regression on random subset
log_reg = LogisticRegression(learning_rate=0.1, num_iterations=1000)
log_reg.fit(X_train_subset, y_train_subset)

print("\nLogistic Regression training test:")
print("Weights shape:", log_reg.weights.shape)
print("Bias value:", log_reg.bias)

# predict on validation subset
log_reg_predictions = log_reg.predict(X_val_subset)

print("Predictions shape:", log_reg_predictions.shape)
print("First 20 predictions:", log_reg_predictions[:20])
print("First 20 actual labels:", y_val_subset[:20])

log_reg_accuracy = np.mean(log_reg_predictions == y_val_subset)
print("Validation accuracy:", log_reg_accuracy)


# test Logistic Regression on random test subset
log_reg_test_predictions = log_reg.predict(X_test_subset)
log_reg_test_accuracy = np.mean(log_reg_test_predictions == y_test_subset)

print("\nLogistic Regression random test subset check:")
print("Test accuracy:", log_reg_test_accuracy)
print("First 20 test predictions:", log_reg_test_predictions[:20])
print("First 20 actual test labels:", y_test_subset[:20])

# try more than one learning rate and number of iterations
print("\nTesting different Logistic Regression settings:")

settings = [
    (0.01, 500),
    (0.01, 1000),
    (0.1, 500),
    (0.1, 1000)
]

for learning_rate, num_iterations in settings:
    log_reg = LogisticRegression(
        learning_rate=learning_rate,
        num_iterations=num_iterations
    )

    log_reg.fit(X_train_subset, y_train_subset)
    predictions = log_reg.predict(X_val_subset)
    accuracy = np.mean(predictions == y_val_subset)

    print(
        f"learning_rate = {learning_rate}, "
        f"num_iterations = {num_iterations}, "
        f"validation accuracy = {accuracy}"
    )