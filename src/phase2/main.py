import numpy as np

from data_module import load_mnist_csv, normalize_pixels, split_train_validation
from features_module import apply_pca
from models_module import KNN, GaussianNaiveBayes, MulticlassLogisticRegression
from evaluation_module import accuracy_score, precision_recall_f1_multiclass


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

    accuracy = accuracy_score(y_eval, predictions)
    precision, recall, f1 = precision_recall_f1_multiclass(
        y_eval, predictions, average="macro"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


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
    # load and normalize data
    X_train_full, y_train_full = load_mnist_csv("MNIST-data/mnist_train.csv")
    X_test, y_test = load_mnist_csv("MNIST-data/mnist_test.csv")

    X_train_full = normalize_pixels(X_train_full)
    X_test = normalize_pixels(X_test)

    # split training data into train and validation
    X_train, X_val, y_train, y_val = split_train_validation(X_train_full, y_train_full)

    print("\nRunning Phase 2 multi-class classification for digits 0 to 9")

    # smaller subsets for faster experiments
    X_train_subset, y_train_subset = stratified_subset(X_train, y_train, 2000)
    X_val_subset, y_val_subset = stratified_subset(X_val, y_val, 300)
    X_test_subset, y_test_subset = stratified_subset(X_test, y_test, 300)

    print("\nStratified subsets created.")

    # =========================
    # VALIDATION RESULTS (RAW)
    # =========================

    raw_models = {
        "KNN (k=1)": KNN(k=3),
        "Logistic Regression": MulticlassLogisticRegression(
            learning_rate=0.1,
            num_iterations=300,
            lambda_reg=0.0
        ),
        "Gaussian Naive Bayes": GaussianNaiveBayes()
    }

    raw_results = {}

    for name, model in raw_models.items():
        raw_results[name] = evaluate_model(
            model,
            X_train_subset, y_train_subset,
            X_val_subset, y_val_subset
        )

    print_results("Validation Results (Raw Features)", raw_results)

    # =========================
    # VALIDATION RESULTS (PCA)
    # =========================

    X_train_pca, X_val_pca, X_test_pca, pca_model = apply_pca(
        X_train_subset,
        X_val_subset,
        X_test_subset,
        n_components=50
    )

    print("\nPCA Applied")
    print("Explained variance:", round(np.sum(pca_model.explained_variance_ratio), 4))

    pca_models = {
        "KNN (k=1)": KNN(k=1),
        "Logistic Regression": MulticlassLogisticRegression(
            learning_rate=0.1,
            num_iterations=300,
            lambda_reg=0.0
        ),
        "Gaussian Naive Bayes": GaussianNaiveBayes()
    }

    pca_results = {}

    for name, model in pca_models.items():
        pca_results[name] = evaluate_model(
            model,
            X_train_pca, y_train_subset,
            X_val_pca, y_val_subset
        )

    print_results("Validation Results (PCA Features)", pca_results)

    # =========================
    # FINAL VALIDATION COMPARISON
    # =========================

    print("\nFinal Comparison (Validation Macro F1):")

    for name in raw_results:
        print(f"\n{name}:")
        print(f"Raw F1: {raw_results[name]['f1']:.4f}")
        print(f"PCA F1: {pca_results[name]['f1']:.4f}")

    # =========================
    # FINAL TEST EVALUATION
    # chosen final model: KNN (k=1) + PCA
    # =========================

    final_model = KNN(k=1)
    final_test_results = evaluate_model(
        final_model,
        X_train_pca, y_train_subset,
        X_test_pca, y_test_subset
    )

    print("\nFinal Test Results (Chosen Model: KNN (k=1) + PCA)")
    print("---------------------------------------------------")
    print(f"Accuracy : {final_test_results['accuracy']:.4f}")
    print(f"Precision: {final_test_results['precision']:.4f}")
    print(f"Recall   : {final_test_results['recall']:.4f}")
    print(f"F1 Score : {final_test_results['f1']:.4f}")


if __name__ == "__main__":
    main()