import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
    # split the available training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_val, y_train, y_val