import numpy as np

from data_module import (
    load_mnist_csv,
    filter_binary_classes,
    normalize_pixels,
    split_train_validation,
)

from models_module import KNN, LogisticRegression, GaussianNaiveBayes
from evaluation_module import (
    accuracy_score,
    precision_score_binary,
    recall_score_binary,
    f1_score_binary,
)


def stratified_subset(X, y, size, random_state=42):
    """
    Take a balanced subset from binary classes.
    Useful if you want faster experiments.
    """
    rng = np.random.default_rng(random_state)

    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("This helper expects binary classification only.")

    samples_per_class = size // 2

    selected_indices = []

    for current_class in classes:
        class_indices = np.where(y == current_class)[0]

        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"Not enough samples in class {current_class} "
                f"to take {samples_per_class} samples."
            )

        chosen = rng.choice(class_indices, size=samples_per_class, replace=False)
        selected_indices.extend(chosen)

    selected_indices = np.array(selected_indices)
    selected_indices = rng.permutation(selected_indices)

    return X[selected_indices], y[selected_indices]


def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    """
    Train model, then compare train vs validation performance.
    This helps us see if there may be overfitting.
    """
    model.fit(X_train, y_train)

    # predictions on training data
    y_train_pred = model.predict(X_train)

    # predictions on validation data
    y_val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    train_prec = precision_score_binary(y_train, y_train_pred)
    val_prec = precision_score_binary(y_val, y_val_pred)

    train_rec = recall_score_binary(y_train, y_train_pred)
    val_rec = recall_score_binary(y_val, y_val_pred)

    train_f1 = f1_score_binary(y_train, y_train_pred)
    val_f1 = f1_score_binary(y_val, y_val_pred)

    gap = train_acc - val_acc

    print("=" * 70)
    print(f"{model_name}")
    print("=" * 70)
    print(f"Train Accuracy   : {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Train Precision  : {train_prec:.4f}")
    print(f"Validation Precision: {val_prec:.4f}")
    print(f"Train Recall     : {train_rec:.4f}")
    print(f"Validation Recall: {val_rec:.4f}")
    print(f"Train F1         : {train_f1:.4f}")
    print(f"Validation F1    : {val_f1:.4f}")
    print(f"Accuracy Gap     : {gap:.4f}")

    # simple interpretation
    if gap > 0.05:
        print("Possible Overfitting: training is noticeably better than validation.")
    elif train_acc < 0.85 and val_acc < 0.85:
        print("Possible Underfitting: both training and validation are relatively low.")
    else:
        print("Generalization looks reasonable.")

    print()

    return {
        "model": model_name,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "train_precision": train_prec,
        "val_precision": val_prec,
        "train_recall": train_rec,
        "val_recall": val_rec,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "gap": gap,
    }


def main():
    # =========================
    # 1) Load training CSV
    # =========================
    X, y = load_mnist_csv("MNIST-data/mnist_train.csv")

    # =========================
    # 2) Binary filter
    # =========================
    X, y = filter_binary_classes(X, y, class_1=0, class_2=1)

    # =========================
    # 3) Normalize
    # =========================
    X = normalize_pixels(X)

    # =========================
    # 4) Train / validation split
    # =========================
    X_train, X_val, y_train, y_val = split_train_validation(
        X, y, val_size=0.2, random_state=42
    )

    # =========================
    # 5) Optional subset for speed
    #    You can remove this if you want full data
    # =========================
    X_train_small, y_train_small = stratified_subset(
        X_train, y_train, size=2000, random_state=42
    )

    X_val_small, y_val_small = stratified_subset(
        X_val, y_val, size=300, random_state=42
    )

    print("=" * 70)
    print("Overfitting Check on Binary MNIST (0 vs 1)")
    print("=" * 70)
    print(f"Train subset shape: {X_train_small.shape}")
    print(f"Validation subset shape: {X_val_small.shape}")
    print()

    results = []

    # =========================
    # 6) KNN
    # =========================
    knn = KNN(k=3)
    results.append(
        evaluate_model(
            knn,
            X_train_small,
            y_train_small,
            X_val_small,
            y_val_small,
            "KNN (k=3)"
        )
    )

    # =========================
    # 7) Logistic Regression
    # =========================
    logistic = LogisticRegression(learning_rate=0.1, num_iterations=1000)
    results.append(
        evaluate_model(
            logistic,
            X_train_small,
            y_train_small,
            X_val_small,
            y_val_small,
            "Logistic Regression"
        )
    )

    # =========================
    # 8) Gaussian Naive Bayes
    # =========================
    gnb = GaussianNaiveBayes()
    results.append(
        evaluate_model(
            gnb,
            X_train_small,
            y_train_small,
            X_val_small,
            y_val_small,
            "Gaussian Naive Bayes"
        )
    )

    # =========================
    # 9) Final compact summary
    # =========================
    print("=" * 70)
    print("Final Summary")
    print("=" * 70)

    for r in results:
        print(
            f"{r['model']:<25} "
            f"Train Acc: {r['train_accuracy']:.4f} | "
            f"Val Acc: {r['val_accuracy']:.4f} | "
            f"Gap: {r['gap']:.4f}"
        )


if __name__ == "__main__":
    main()