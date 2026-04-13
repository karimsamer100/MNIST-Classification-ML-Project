import pandas as pd
import numpy as np

def load_mnist_csv(file_path):
    # no header in this dataset  so we tell pandas explicitly
    data = pd.read_csv(file_path, header=None)

    # first column is the label (digit)
    y = data.iloc[:, 0].to_numpy(dtype=np.int64)

    # remaining columns are pixel values (28x28 flattened → 784)
    X = data.iloc[:, 1:].to_numpy(dtype=np.float64)

    return X, y

def filter_binary_classes(X, y, class_1=0, class_2=1):
    # keep only rows that belong to class_1 or class_2
    mask = (y == class_1) | (y == class_2)

    X_filtered = X[mask]
    y_filtered = y[mask]

    # convert labels to binary (0 and 1)
    y_filtered = np.where(y_filtered == class_1, 0, 1)

    return X_filtered, y_filtered

def normalize_pixels(X):
    # pixel values in MNIST are from 0 to 255
    # dividing by 255 makes them between 0 and 1
    X_normalized = X / 255.0

    return X_normalized




def split_train_validation(X, y, val_size=0.2, random_state=42):
    # basic checks
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")

    if not (0 < val_size < 1):
        raise ValueError("val_size must be between 0 and 1")

    # random generator for reproducibility
    rng = np.random.default_rng(random_state)

    train_indices = []  #which rows go to training
    val_indices = []    #which rows go to validation

    # stratified split: split each class separately
    unique_classes = np.unique(y)

    for current_class in unique_classes:
        class_indices = np.where(y == current_class)[0] #find class indices

        # shuffle indices of this class
        shuffled_class_indices = rng.permutation(class_indices)

        # number of validation samples from this class
        num_val_class = int(len(shuffled_class_indices) * val_size)

        # make sure validation is not empty if class has enough samples
        if num_val_class == 0 and len(shuffled_class_indices) > 1:
            num_val_class = 1

        class_val_indices = shuffled_class_indices[:num_val_class]
        class_train_indices = shuffled_class_indices[num_val_class:]

        val_indices.extend(class_val_indices)
        train_indices.extend(class_train_indices)

    # convert to numpy arrays
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    # final shuffle so samples are mixed
    train_indices = rng.permutation(train_indices)
    val_indices = rng.permutation(val_indices)

    X_train = X[train_indices]
    X_val = X[val_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]

    return X_train, X_val, y_train, y_val