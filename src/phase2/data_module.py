import pandas as pd
import numpy as np


def load_mnist_csv(file_path):
    # MNIST CSV has no header
    data = pd.read_csv(file_path, header=None)

    # first column is the label
    y = data.iloc[:, 0].to_numpy(dtype=np.int64)

    # remaining 784 columns are pixel values
    X = data.iloc[:, 1:].to_numpy(dtype=np.float64)

    return X, y


def normalize_pixels(X):
    # scale pixels from [0, 255] to [0, 1]
    return X / 255.0


def split_train_validation(X, y, val_size=0.2, random_state=42):
    # basic checks
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")

    if not (0 < val_size < 1):
        raise ValueError("val_size must be between 0 and 1")

    rng = np.random.default_rng(random_state)

    train_indices = []
    val_indices = []

    # stratified split for all classes
    unique_classes = np.unique(y)

    for current_class in unique_classes:
        class_indices = np.where(y == current_class)[0]
        shuffled_class_indices = rng.permutation(class_indices)

        num_val_class = int(len(shuffled_class_indices) * val_size)

        if num_val_class == 0 and len(shuffled_class_indices) > 1:
            num_val_class = 1

        class_val_indices = shuffled_class_indices[:num_val_class]
        class_train_indices = shuffled_class_indices[num_val_class:]

        val_indices.extend(class_val_indices)
        train_indices.extend(class_train_indices)

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    train_indices = rng.permutation(train_indices)
    val_indices = rng.permutation(val_indices)

    X_train = X[train_indices]
    X_val = X[val_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]

    return X_train, X_val, y_train, y_val


def get_class_distribution(y):
    classes, counts = np.unique(y, return_counts=True)

    distribution = {}
    for cls, count in zip(classes, counts):
        distribution[int(cls)] = int(count)

    return distribution