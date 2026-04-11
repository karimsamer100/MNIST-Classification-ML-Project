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
    f1_score_binary
)


def evaluate_model(model, X_train, y_train, X_eval, y_eval):
    # train the model
    model.fit(X_train, y_train)

    # make predictions
    predictions = model.predict(X_eval)

    # compute evaluation metrics
    results = {
        "accuracy": accuracy_score(y_eval, predictions),
        "precision": precision_score_binary(y_eval, predictions),
        "recall": recall_score_binary(y_eval, predictions),
        "f1": f1_score_binary(y_eval, predictions)
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


def main():
    # =========================
    # 1. LOAD AND PREPROCESS DATA
    # =========================

    # load original training data
    X_train_full, y_train_full = load_mnist_csv("MNIST-data/mnist_train.csv")

    # keep only digits 0 and 1
    X_train_full, y_train_full = filter_binary_classes(X_train_full, y_train_full)

    # normalize training data
    X_train_full = normalize_pixels(X_train_full)

    # split training data into train and validation
    X_train, X_val, y_train, y_val = split_train_validation(X_train_full, y_train_full)

    # load original test data
    X_test, y_test = load_mnist_csv("MNIST-data/mnist_test.csv")

    # keep only digits 0 and 1
    X_test, y_test = filter_binary_classes(X_test, y_test)

    # normalize test data
    X_test = normalize_pixels(X_test)

    print("Data loaded successfully.")
    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    # =========================
    # 2. CREATE RANDOM SUBSETS
    # =========================

    rng = np.random.default_rng(42)

    train_indices = rng.choice(len(X_train), size=2000, replace=False)
    val_indices = rng.choice(len(X_val), size=300, replace=False)
    test_indices = rng.choice(len(X_test), size=300, replace=False)

    X_train_subset = X_train[train_indices]
    y_train_subset = y_train[train_indices]

    X_val_subset = X_val[val_indices]
    y_val_subset = y_val[val_indices]

    X_test_subset = X_test[test_indices]
    y_test_subset = y_test[test_indices]

    print("\nRandom subsets created.")
    print("Train subset shape:", X_train_subset.shape)
    print("Validation subset shape:", X_val_subset.shape)
    print("Test subset shape:", X_test_subset.shape)

    # =========================
    # 3. EVALUATE MODELS ON RAW FEATURES
    # =========================

    raw_validation_results = {
        "KNN": evaluate_model(
            KNN(k=3),
            X_train_subset, y_train_subset,
            X_val_subset, y_val_subset
        ),
        "Logistic Regression": evaluate_model(
            LogisticRegression(learning_rate=0.1, num_iterations=1000),
            X_train_subset, y_train_subset,
            X_val_subset, y_val_subset
        ),
        "Gaussian Naive Bayes": evaluate_model(
            GaussianNaiveBayes(),
            X_train_subset, y_train_subset,
            X_val_subset, y_val_subset
        )
    }

    print_results("Validation Results on Raw Features", raw_validation_results)

    raw_test_results = {
        "KNN": evaluate_model(
            KNN(k=3),
            X_train_subset, y_train_subset,
            X_test_subset, y_test_subset
        ),
        "Logistic Regression": evaluate_model(
            LogisticRegression(learning_rate=0.1, num_iterations=1000),
            X_train_subset, y_train_subset,
            X_test_subset, y_test_subset
        ),
        "Gaussian Naive Bayes": evaluate_model(
            GaussianNaiveBayes(),
            X_train_subset, y_train_subset,
            X_test_subset, y_test_subset
        )
    }

    print_results("Test Results on Raw Features", raw_test_results)

    # =========================
    # 4. APPLY PCA
    # =========================

    X_train_pca, X_val_pca, X_test_pca, pca_model = apply_pca(
        X_train_subset,
        X_val_subset,
        X_test_subset,
        n_components=50
    )

    total_explained_variance = np.sum(pca_model.explained_variance_ratio)

    print("\nPCA applied successfully.")
    print("Original train shape:", X_train_subset.shape)
    print("PCA train shape:", X_train_pca.shape)
    print("Total explained variance ratio:", round(total_explained_variance, 4))

    # =========================
    # 5. EVALUATE MODELS ON PCA FEATURES
    # =========================

    pca_validation_results = {
        "KNN": evaluate_model(
            KNN(k=3),
            X_train_pca, y_train_subset,
            X_val_pca, y_val_subset
        ),
        "Logistic Regression": evaluate_model(
            LogisticRegression(learning_rate=0.1, num_iterations=1000),
            X_train_pca, y_train_subset,
            X_val_pca, y_val_subset
        ),
        "Gaussian Naive Bayes": evaluate_model(
            GaussianNaiveBayes(),
            X_train_pca, y_train_subset,
            X_val_pca, y_val_subset
        )
    }

    print_results("Validation Results after PCA", pca_validation_results)

    # =========================
    # 6. FINAL COMPARISON SUMMARY
    # =========================

    print("\nFinal Comparison Summary")
    print("------------------------")

    for model_name in raw_validation_results:
        raw_acc = raw_validation_results[model_name]["accuracy"]
        pca_acc = pca_validation_results[model_name]["accuracy"]

        raw_f1 = raw_validation_results[model_name]["f1"]
        pca_f1 = pca_validation_results[model_name]["f1"]

        print(f"\n{model_name}:")
        print(f"Accuracy -> Raw: {raw_acc:.4f}, PCA: {pca_acc:.4f}")
        print(f"F1 Score -> Raw: {raw_f1:.4f}, PCA: {pca_f1:.4f}")


if __name__ == "__main__":
    main()