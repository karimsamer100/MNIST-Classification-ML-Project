import numpy as np

from data_module import load_mnist_csv, normalize_pixels
from models_module import MulticlassLogisticRegression
from evaluation_module import accuracy_score, precision_recall_f1_multiclass


RANDOM_STATE = 42
TRAIN_PATH = "MNIST-data/mnist_train.csv"
NUM_CLASSES = 10

K_FOLDS = 3

# keep it reasonable so it does not take forever
MAX_SAMPLES = 15000

PARAM_GRID = [
    {"learning_rate": 0.05, "lambda_reg": 0.0},
    {"learning_rate": 0.1, "lambda_reg": 0.0},
    {"learning_rate": 0.1, "lambda_reg": 0.001},
    {"learning_rate": 0.1, "lambda_reg": 0.01},
]

ITERATIONS = 300


def evaluate_model(model, X, y):
    preds = model.predict(X)
    precision, recall, f1 = precision_recall_f1_multiclass(
        y, preds, num_classes=NUM_CLASSES
    )
    acc = accuracy_score(y, preds)
    return acc, precision, recall, f1


def stratified_sample(X, y, max_samples, random_state=42):
    rng = np.random.default_rng(random_state)

    selected_indices = []
    classes = np.unique(y)

    samples_per_class = max_samples // len(classes)

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(
            cls_indices,
            size=min(samples_per_class, len(cls_indices)),
            replace=False
        )
        selected_indices.extend(chosen)

    selected_indices = rng.permutation(np.array(selected_indices))
    return X[selected_indices], y[selected_indices]


def make_folds(y, k_folds=3, random_state=42):
    rng = np.random.default_rng(random_state)

    folds = [[] for _ in range(k_folds)]

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        cls_indices = rng.permutation(cls_indices)

        for i, index in enumerate(cls_indices):
            folds[i % k_folds].append(index)

    return [np.array(fold) for fold in folds]


def main():
    X, y = load_mnist_csv(TRAIN_PATH)
    X = normalize_pixels(X)

    X, y = stratified_sample(X, y, MAX_SAMPLES, RANDOM_STATE)

    folds = make_folds(y, K_FOLDS, RANDOM_STATE)

    print("\nHyperparameter Tuning with 3-Fold Cross-Validation")
    print("=" * 90)
    print(f"{'LR':<10} {'Lambda':<10} {'Mean Acc':>10} {'Mean Precision':>16} {'Mean Recall':>14} {'Mean F1':>10}")
    print("-" * 90)

    best_params = None
    best_mean_f1 = -1

    for params in PARAM_GRID:
        fold_scores = []

        for fold_idx in range(K_FOLDS):
            val_indices = folds[fold_idx]
            train_indices = np.concatenate(
                [folds[i] for i in range(K_FOLDS) if i != fold_idx]
            )

            X_train = X[train_indices]
            y_train = y[train_indices]
            X_val = X[val_indices]
            y_val = y[val_indices]

            model = MulticlassLogisticRegression(
                learning_rate=params["learning_rate"],
                num_iterations=ITERATIONS,
                lambda_reg=params["lambda_reg"],
            )

            model.fit(X_train, y_train)
            acc, precision, recall, f1 = evaluate_model(model, X_val, y_val)

            fold_scores.append([acc, precision, recall, f1])

        fold_scores = np.array(fold_scores)
        mean_scores = np.mean(fold_scores, axis=0)

        print(
            f"{params['learning_rate']:<10} "
            f"{params['lambda_reg']:<10} "
            f"{mean_scores[0]:>10.4f} "
            f"{mean_scores[1]:>16.4f} "
            f"{mean_scores[2]:>14.4f} "
            f"{mean_scores[3]:>10.4f}"
        )

        if mean_scores[3] > best_mean_f1:
            best_mean_f1 = mean_scores[3]
            best_params = params

    print("\nBest parameters based on mean cross-validation Macro F1:")
    print(best_params)
    print(f"Best mean CV F1 = {best_mean_f1:.4f}")


if __name__ == "__main__":
    main()