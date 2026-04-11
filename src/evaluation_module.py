import numpy as np


def accuracy_score(y_true, y_pred):
    # make sure both arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # count correct predictions
    correct_predictions = np.sum(y_true == y_pred)

    # accuracy = correct / total
    accuracy = correct_predictions / len(y_true)

    return accuracy

def confusion_matrix_binary(y_true, y_pred):
    # make sure both arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[tn, fp],
                     [fn, tp]])

def precision_score_binary(y_true, y_pred):
    conf_matrix = confusion_matrix_binary(y_true, y_pred)

    tp = conf_matrix[1, 1]
    fp = conf_matrix[0, 1]

    # avoid division by zero
    if tp + fp == 0:
        return 0.0

    precision = tp / (tp + fp)
    return precision


def recall_score_binary(y_true, y_pred):
    conf_matrix = confusion_matrix_binary(y_true, y_pred)

    tp = conf_matrix[1, 1]
    fn = conf_matrix[1, 0]

    # avoid division by zero
    if tp + fn == 0:
        return 0.0

    recall = tp / (tp + fn)
    return recall


def f1_score_binary(y_true, y_pred):
    precision = precision_score_binary(y_true, y_pred)
    recall = recall_score_binary(y_true, y_pred)

    # avoid division by zero
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix_binary(y_true, y_pred)

    print("\nConfusion Matrix:")
    print("          Predicted")
    print("          0     1")
    print(f"Actual 0  {cm[0,0]:<5} {cm[0,1]}")
    print(f"Actual 1  {cm[1,0]:<5} {cm[1,1]}")
    