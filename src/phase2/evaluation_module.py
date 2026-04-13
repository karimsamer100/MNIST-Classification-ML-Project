import numpy as np


def accuracy_score(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true)


def confusion_matrix_multiclass(y_true, y_pred, num_classes=None):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if num_classes is None:
        num_classes = int(max(np.max(y_true), np.max(y_pred))) + 1

    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1

    return cm


def precision_recall_f1_multiclass(y_true, y_pred, num_classes=None, average="macro"):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    cm = confusion_matrix_multiclass(y_true, y_pred, num_classes)

    precisions = []
    recalls = []
    f1_scores = []

    for class_index in range(cm.shape[0]):
        tp = cm[class_index, class_index]
        fp = np.sum(cm[:, class_index]) - tp
        fn = np.sum(cm[class_index, :]) - tp

        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)

    if average == "macro":
        return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

    elif average == "weighted":
        class_counts = np.bincount(y_true.astype(int), minlength=cm.shape[0])
        total_count = np.sum(class_counts)

        weighted_precision = np.sum(precisions * class_counts) / total_count
        weighted_recall = np.sum(recalls * class_counts) / total_count
        weighted_f1 = np.sum(f1_scores * class_counts) / total_count

        return weighted_precision, weighted_recall, weighted_f1

    else:
        raise ValueError("average must be either 'macro' or 'weighted'")


def classification_report_multiclass(y_true, y_pred, num_classes=None):
    if num_classes is None:
        num_classes = int(max(np.max(y_true), np.max(y_pred))) + 1

    cm = confusion_matrix_multiclass(y_true, y_pred, num_classes)

    print("\nClassification Report:")
    print("Class | Precision | Recall | F1-score")

    for class_index in range(cm.shape[0]):
        tp = cm[class_index, class_index]
        fp = np.sum(cm[:, class_index]) - tp
        fn = np.sum(cm[class_index, :]) - tp

        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        print(f"{class_index:^5} | {precision:^9.4f} | {recall:^6.4f} | {f1:^8.4f}")


def print_confusion_matrix(y_true, y_pred, num_classes=None):
    cm = confusion_matrix_multiclass(y_true, y_pred, num_classes)

    print("\nConfusion Matrix:")
    print(cm)