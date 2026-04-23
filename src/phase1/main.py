import numpy as np

from data_module import (
    load_mnist_csv,
    filter_binary_classes,
    normalize_pixels,
    split_train_validation,
)
from evaluation_module import (
    accuracy_score,
    confusion_matrix_binary,
    f1_score_binary,
    precision_score_binary,
    recall_score_binary,
)
from features_module import apply_pca
from models_module import GaussianNaiveBayes, KNN, LogisticRegression


# =========================
# CONFIGURATION
# =========================

CLASS_1 = 0
CLASS_2 = 1
RANDOM_STATE = 42

TRAIN_PATH = "MNIST-data/mnist_train.csv"
TEST_PATH = "MNIST-data/mnist_test.csv"

VALIDATION_SIZE = 0.2
PCA_COMPONENTS = 50


# =========================
# HELPERS
# =========================

def make_models():
    return {
        "KNN (k=3)": KNN(k=3),
        "Logistic Regression": LogisticRegression(
            learning_rate=0.1,
            num_iterations=1000,
        ),
        "Gaussian Naive Bayes": GaussianNaiveBayes(),
    }


def evaluate_model(model, X_train, y_train, X_eval, y_eval):
    model.fit(X_train, y_train)
    predictions = model.predict(X_eval)

    return {
        "accuracy": accuracy_score(y_eval, predictions),
        "precision": precision_score_binary(y_eval, predictions),
        "recall": recall_score_binary(y_eval, predictions),
        "f1": f1_score_binary(y_eval, predictions),
        "confusion_matrix": confusion_matrix_binary(y_eval, predictions),
    }


def evaluate_all_models(X_train, y_train, X_eval, y_eval):
    results = {}
    for model_name, model in make_models().items():
        results[model_name] = evaluate_model(model, X_train, y_train, X_eval, y_eval)
    return results


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_dataset_summary(name, X, y):
    counts = np.bincount(y)
    print(f"{name:<12} shape={str(X.shape):<14} class_counts={counts.tolist()}")


def print_metric_table(title, results):
    print_section(title)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 72)

    for model_name, metrics in results.items():
        print(
            f"{model_name:<25} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f}"
        )


def print_confusion_matrices(title, results):
    print_section(title)
    print(f"Rows are actual labels, columns are predicted labels. 0={CLASS_1}, 1={CLASS_2}")

    for model_name, metrics in results.items():
        cm = metrics["confusion_matrix"]
        print(f"\n{model_name}")
        print("              Pred 0   Pred 1")
        print(f"Actual 0      {cm[0, 0]:>6}   {cm[0, 1]:>6}")
        print(f"Actual 1      {cm[1, 0]:>6}   {cm[1, 1]:>6}")


def print_accuracy_comparison(raw_validation, raw_test, pca_validation, pca_test):
    print_section("Accuracy Comparison Before and After PCA")
    print(
        f"{'Model':<25} "
        f"{'Raw Val':>9} {'Raw Test':>9} "
        f"{'PCA Val':>9} {'PCA Test':>9}"
    )
    print("-" * 68)

    for model_name in raw_validation:
        print(
            f"{model_name:<25} "
            f"{raw_validation[model_name]['accuracy']:>9.4f} "
            f"{raw_test[model_name]['accuracy']:>9.4f} "
            f"{pca_validation[model_name]['accuracy']:>9.4f} "
            f"{pca_test[model_name]['accuracy']:>9.4f}"
        )


def load_and_prepare_data():
    X_train_full, y_train_full = load_mnist_csv(TRAIN_PATH)
    X_train_full, y_train_full = filter_binary_classes(
        X_train_full,
        y_train_full,
        CLASS_1,
        CLASS_2,
    )
    X_train_full = normalize_pixels(X_train_full)

    X_train, X_val, y_train, y_val = split_train_validation(
        X_train_full,
        y_train_full,
        val_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    X_test, y_test = load_mnist_csv(TEST_PATH)
    X_test, y_test = filter_binary_classes(X_test, y_test, CLASS_1, CLASS_2)
    X_test = normalize_pixels(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


# =========================
# MAIN PIPELINE
# =========================

def main():
    print_section("Phase 1: Binary MNIST Classification")
    print(f"Task: classify digit {CLASS_1} vs digit {CLASS_2}")
    print(f"Encoded labels: 0 means original digit {CLASS_1}, 1 means original digit {CLASS_2}")
    print(f"Random state: {RANDOM_STATE}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()

    print_section("Prepared Full Dataset")
    print_dataset_summary("Train", X_train, y_train)
    print_dataset_summary("Validation", X_val, y_val)
    print_dataset_summary("Test", X_test, y_test)
    print(f"Pixel range after normalization: [{X_train.min():.1f}, {X_train.max():.1f}]")

    raw_validation_results = evaluate_all_models(
        X_train,
        y_train,
        X_val,
        y_val,
    )
    raw_test_results = evaluate_all_models(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    X_train_pca, X_val_pca, X_test_pca, pca_model = apply_pca(
        X_train,
        X_val,
        X_test,
        n_components=PCA_COMPONENTS,
    )

    print_section("PCA Summary")
    print(f"Components kept: {PCA_COMPONENTS}")
    print(f"Raw feature count: {X_train.shape[1]}")
    print(f"PCA feature count: {X_train_pca.shape[1]}")
    print(f"Total explained variance ratio: {np.sum(pca_model.explained_variance_ratio):.4f}")
    print("First 10 explained variance ratios:")
    print(np.round(pca_model.explained_variance_ratio[:10], 4))

    pca_validation_results = evaluate_all_models(
        X_train_pca,
        y_train,
        X_val_pca,
        y_val,
    )
    pca_test_results = evaluate_all_models(
        X_train_pca,
        y_train,
        X_test_pca,
        y_test,
    )

    print_metric_table("Validation Scores Before PCA", raw_validation_results)
    print_metric_table("Test Scores Before PCA", raw_test_results)
    print_metric_table("Validation Scores After PCA", pca_validation_results)
    print_metric_table("Test Scores After PCA", pca_test_results)

    print_accuracy_comparison(
        raw_validation_results,
        raw_test_results,
        pca_validation_results,
        pca_test_results,
    )

    print_confusion_matrices("Validation Confusion Matrices Before PCA", raw_validation_results)
    print_confusion_matrices("Test Confusion Matrices Before PCA", raw_test_results)
    print_confusion_matrices("Validation Confusion Matrices After PCA", pca_validation_results)
    print_confusion_matrices("Test Confusion Matrices After PCA", pca_test_results)


if __name__ == "__main__":
    main()