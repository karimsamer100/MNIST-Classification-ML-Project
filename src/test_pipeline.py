from data_module import (
    load_mnist_csv,
    filter_binary_classes,
    normalize_pixels,
    split_train_validation
)
from models_module import KNN

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