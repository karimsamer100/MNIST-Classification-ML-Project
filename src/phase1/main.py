import numpy as np

from data_module import (
    load_mnist_csv,
    filter_binary_classes,
    normalize_pixels,
    split_train_validation
)
from features_module import apply_pca
from models_module import KNN, LogisticRegression, GaussianNaiveBayes
from evaluation_module import (
    accuracy_score,
    precision_score_binary,
    recall_score_binary,
    f1_score_binary,
    print_confusion_matrix
)

# =========================
# HELPER FUNCTIONS
# =========================

def stratified_subset(X, y, size, random_state=42):
    rng = np.random.default_rng(random_state)

    indices = []
    classes = np.unique(y)
    per_class = size // len(classes)

    for c in classes:
        class_indices = np.where(y == c)[0]
        selected = rng.choice(class_indices, size=per_class, replace=False)
        indices.extend(selected)

    indices = np.array(indices)
    rng.shuffle(indices)

    return X[indices], y[indices]


def evaluate_model(model, X_train, y_train, X_eval, y_eval):
    model.fit(X_train, y_train)
    predictions = model.predict(X_eval)

    results = {
        "accuracy": accuracy_score(y_eval, predictions),
        "precision": precision_score_binary(y_eval, predictions),
        "recall": recall_score_binary(y_eval, predictions),
        "f1": f1_score_binary(y_eval, predictions),
        "predictions": predictions
    }

    return results


def print_results(title, results_dict):
    print(f"\n{title}")
    print("-" * len(title))

    for model_name, metrics in results_dict.items():
        print(f"\n{model_name}:")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1']:.4f}")


# =========================
# MAIN PIPELINE
# =========================

def main():

   
    CLASS_1 = 0
    CLASS_2 = 1

    # =========================
    # LOAD DATA
    # =========================

    X_train_full, y_train_full = load_mnist_csv("MNIST-data/mnist_train.csv")
    X_train_full, y_train_full = filter_binary_classes(X_train_full, y_train_full, CLASS_1, CLASS_2)
    X_train_full = normalize_pixels(X_train_full)

    X_train, X_val, y_train, y_val = split_train_validation(X_train_full, y_train_full)

    X_test, y_test = load_mnist_csv("MNIST-data/mnist_test.csv")
    X_test, y_test = filter_binary_classes(X_test, y_test, CLASS_1, CLASS_2)
    X_test = normalize_pixels(X_test)

    print(f"\nRunning classification for digits: {CLASS_1} vs {CLASS_2}")

    # =========================
    # STRATIFIED SUBSETS
    # =========================

    X_train_subset, y_train_subset = stratified_subset(X_train, y_train, 2000)
    X_val_subset, y_val_subset = stratified_subset(X_val, y_val, 300)
    X_test_subset, y_test_subset = stratified_subset(X_test, y_test, 300)

    print("\nStratified subsets created.")

    # =========================
    # RAW FEATURES
    # =========================

    raw_results = {}

    models = {
        "KNN": KNN(k=3),
        "Logistic Regression": LogisticRegression(learning_rate=0.1, num_iterations=1000),
        "Gaussian Naive Bayes": GaussianNaiveBayes()
    }

    for name, model in models.items():
        raw_results[name] = evaluate_model(
            model,
            X_train_subset, y_train_subset,
            X_val_subset, y_val_subset
        )

    print_results("Validation Results (Raw Features)", raw_results)

    # =========================
    # TEST RESULTS
    # =========================

    print("\nTest Results:")

    for name, model in models.items():
        test_res = evaluate_model(
            model,
            X_train_subset, y_train_subset,
            X_test_subset, y_test_subset
        )

        print(f"\n{name}:")
        print(f"Accuracy: {test_res['accuracy']:.4f}")

    # =========================
    # PCA
    # =========================

    X_train_pca, X_val_pca, X_test_pca, pca_model = apply_pca(
        X_train_subset,
        X_val_subset,
        X_test_subset,
        n_components=50
    )

    print("\nPCA Applied")
    print("Explained variance:", round(np.sum(pca_model.explained_variance_ratio), 4))

    # =========================
    # PCA RESULTS
    # =========================

    pca_results = {}

    for name, model in models.items():
        pca_results[name] = evaluate_model(
            model,
            X_train_pca, y_train_subset,
            X_val_pca, y_val_subset
        )

    print_results("Validation Results (PCA Features)", pca_results)

    # =========================
    # CONFUSION MATRIX
    # =========================

    print("\nConfusion Matrix Example (KNN - Raw):")
    print_confusion_matrix(y_val_subset, raw_results["KNN"]["predictions"])

    # =========================
    # FINAL SUMMARY
    # =========================

    print("\nFinal Comparison:")

    for name in raw_results:
        print(f"\n{name}:")
        print(f"Raw Accuracy: {raw_results[name]['accuracy']:.4f}")
        print(f"PCA Accuracy: {pca_results[name]['accuracy']:.4f}")


if __name__ == "__main__":
    main()