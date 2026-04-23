import numpy as np

from data_module import load_mnist_csv, normalize_pixels, split_train_validation
from evaluation_module import (
    accuracy_score,
    confusion_matrix_multiclass,
    precision_recall_f1_multiclass,
)
from features_module import apply_pca
from models_module import KNN, GaussianNaiveBayes, MulticlassLogisticRegression


# =========================
# CONFIGURATION
# =========================

RANDOM_STATE = 42

TRAIN_PATH = "MNIST-data/mnist_train.csv"
TEST_PATH = "MNIST-data/mnist_test.csv"

VALIDATION_SIZE = 0.2
PCA_COMPONENTS = 50

LOGISTIC_LEARNING_RATE = 0.1
LOGISTIC_ITERATIONS = 300
LOGISTIC_LAMBDA = 0.0

KNN_K = 2


# =========================
# HELPERS
# =========================

def make_models():
    return {
        f"KNN (k={KNN_K})": KNN(k=KNN_K),
        "Multiclass Logistic Regression": MulticlassLogisticRegression(
            learning_rate=LOGISTIC_LEARNING_RATE,
            num_iterations=LOGISTIC_ITERATIONS,
            lambda_reg=LOGISTIC_LAMBDA,
        ),
        "Gaussian Naive Bayes": GaussianNaiveBayes(),
    }


def evaluate_model(model, X_train, y_train, X_eval, y_eval):
    model.fit(X_train, y_train)
    predictions = model.predict(X_eval)

    precision, recall, f1 = precision_recall_f1_multiclass(
        y_eval,
        predictions,
        average="macro",
    )

    return {
        "accuracy": accuracy_score(y_eval, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix_multiclass(y_eval, predictions, num_classes=10),
    }


def evaluate_all_models(X_train, y_train, X_eval, y_eval):
    results = {}

    for model_name, model in make_models().items():
        results[model_name] = evaluate_model(
            model,
            X_train,
            y_train,
            X_eval,
            y_eval,
        )

    return results


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_dataset_summary(name, X, y):
    counts = np.bincount(y, minlength=10)
    print(f"{name:<12} shape={str(X.shape):<16} class_counts={counts.tolist()}")


def print_metric_table(title, results):
    print_section(title)
    print(f"{'Model':<32} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 78)

    for model_name, metrics in results.items():
        print(
            f"{model_name:<32} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f}"
        )


def print_confusion_matrices(title, results):
    print_section(title)
    print("Rows are actual labels, columns are predicted labels.")

    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        print(metrics["confusion_matrix"])


def print_f1_comparison(raw_validation, raw_test, pca_validation, pca_test):
    print_section("Macro F1 Comparison Before and After PCA")
    print(
        f"{'Model':<32} "
        f"{'Raw Val':>10} {'Raw Test':>10} "
        f"{'PCA Val':>10} {'PCA Test':>10}"
    )
    print("-" * 78)

    for model_name in raw_validation:
        print(
            f"{model_name:<32} "
            f"{raw_validation[model_name]['f1']:>10.4f} "
            f"{raw_test[model_name]['f1']:>10.4f} "
            f"{pca_validation[model_name]['f1']:>10.4f} "
            f"{pca_test[model_name]['f1']:>10.4f}"
        )


def load_and_prepare_data():
    X_train_full, y_train_full = load_mnist_csv(TRAIN_PATH)
    X_test, y_test = load_mnist_csv(TEST_PATH)

    X_train_full = normalize_pixels(X_train_full)
    X_test = normalize_pixels(X_test)

    X_train, X_val, y_train, y_val = split_train_validation(
        X_train_full,
        y_train_full,
        val_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# =========================
# MAIN PIPELINE
# =========================

def main():
    print_section("Phase 2: Multi-Class MNIST Classification")
    print("Task: classify all digits from 0 to 9")
    print(f"Random state: {RANDOM_STATE}")
    print(f"Validation size: {VALIDATION_SIZE}")
    print(f"PCA components: {PCA_COMPONENTS}")
    print(f"KNN k: {KNN_K}")
    

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()

    print_section("Prepared Full Dataset")
    print_dataset_summary("Train", X_train, y_train)
    print_dataset_summary("Validation", X_val, y_val)
    print_dataset_summary("Test", X_test, y_test)
    print(f"Pixel range after normalization: [{X_train.min():.1f}, {X_train.max():.1f}]")

    # -------------------------
    # RAW FEATURES
    # -------------------------
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

    # -------------------------
    # PCA FEATURES
    # -------------------------
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

    # -------------------------
    # PRINT RESULTS
    # -------------------------
    print_metric_table("Validation Scores Before PCA", raw_validation_results)
    print_metric_table("Test Scores Before PCA", raw_test_results)
    print_metric_table("Validation Scores After PCA", pca_validation_results)
    print_metric_table("Test Scores After PCA", pca_test_results)

    print_f1_comparison(
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