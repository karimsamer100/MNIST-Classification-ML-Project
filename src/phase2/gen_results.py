import os
import csv
import numpy as np
import matplotlib.pyplot as plt

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


def ensure_output_folders():
    os.makedirs("results/phase2/tables", exist_ok=True)
    os.makedirs("results/phase2/figures", exist_ok=True)
    os.makedirs("results/phase2/logs", exist_ok=True)


# =========================
# MAIN RESULTS GENERATION
# =========================

def main():
    ensure_output_folders()

    # load data
    X_train_full, y_train_full = load_mnist_csv("MNIST-data/mnist_train.csv")
    X_test, y_test = load_mnist_csv("MNIST-data/mnist_test.csv")

    X_train_full = normalize_pixels(X_train_full)
    X_test = normalize_pixels(X_test)

    X_train, X_val, y_train, y_val = split_train_validation(X_train_full, y_train_full)

    # same subsets used in main.py
    X_train_subset, y_train_subset = stratified_subset(X_train, y_train, 2000)
    X_val_subset, y_val_subset = stratified_subset(X_val, y_val, 300)
    X_test_subset, y_test_subset = stratified_subset(X_test, y_test, 300)

    # -------------------------
    # raw validation results
    # -------------------------
    raw_models = {
        "KNN (k=1)": KNN(k=1),
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

    # -------------------------
    # PCA validation results
    # -------------------------
    X_train_pca, X_val_pca, X_test_pca, pca_model = apply_pca(
        X_train_subset,
        X_val_subset,
        X_test_subset,
        n_components=50
    )

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

    # -------------------------
    # final test result
    # -------------------------
    final_model = KNN(k=1)
    final_test_result = evaluate_model(
        final_model,
        X_train_pca, y_train_subset,
        X_test_pca, y_test_subset
    )

    # -------------------------
    # save validation comparison table
    # -------------------------
    validation_table_path = "results/phase2/tables/validation_comparison.csv"
    with open(validation_table_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Feature Setting", "Model", "Accuracy", "Precision", "Recall", "F1"])

        for model_name, metrics in raw_results.items():
            writer.writerow([
                "Raw",
                model_name,
                round(metrics["accuracy"], 4),
                round(metrics["precision"], 4),
                round(metrics["recall"], 4),
                round(metrics["f1"], 4)
            ])

        for model_name, metrics in pca_results.items():
            writer.writerow([
                "PCA",
                model_name,
                round(metrics["accuracy"], 4),
                round(metrics["precision"], 4),
                round(metrics["recall"], 4),
                round(metrics["f1"], 4)
            ])

    # -------------------------
    # save final test result table
    # -------------------------
    final_test_table_path = "results/phase2/tables/final_test_result.csv"
    with open(final_test_table_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Chosen Model", "Feature Setting", "Accuracy", "Precision", "Recall", "F1"])
        writer.writerow([
            "KNN (k=1)",
            "PCA (50 components)",
            round(final_test_result["accuracy"], 4),
            round(final_test_result["precision"], 4),
            round(final_test_result["recall"], 4),
            round(final_test_result["f1"], 4)
        ])

    # -------------------------
    # plot validation F1 comparison
    # -------------------------
    models = list(raw_results.keys())
    raw_f1 = [raw_results[name]["f1"] for name in models]
    pca_f1 = [pca_results[name]["f1"] for name in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, raw_f1, width, label="Raw Features")
    plt.bar(x + width / 2, pca_f1, width, label="PCA Features")
    plt.xticks(x, models, rotation=10)
    plt.ylabel("Macro F1 Score")
    plt.title("Phase 2 Validation Comparison: Raw vs PCA")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/phase2/figures/validation_f1_comparison.png")
    plt.close()

    # -------------------------
    # plot PCA explained variance
    # -------------------------
    cumulative_variance = np.cumsum(pca_model.explained_variance_ratio)
    components = np.arange(1, len(cumulative_variance) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(components, cumulative_variance, marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/phase2/figures/pca_cumulative_variance.png")
    plt.close()

    # -------------------------
    # save short summary log
    # -------------------------
    summary_path = "results/phase2/logs/phase2_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("Phase 2 Summary\n")
        file.write("====================\n\n")
        file.write("Validation comparison was used for model and feature selection.\n")
        file.write("Final chosen model: KNN (k=1) with PCA (50 components).\n\n")

        file.write("Final Test Result:\n")
        file.write(f"Accuracy : {final_test_result['accuracy']:.4f}\n")
        file.write(f"Precision: {final_test_result['precision']:.4f}\n")
        file.write(f"Recall   : {final_test_result['recall']:.4f}\n")
        file.write(f"F1 Score : {final_test_result['f1']:.4f}\n\n")

        file.write("Important note:\n")
        file.write("These final test results were computed on a stratified test subset of 300 samples.\n")

    print("Phase 2 result files generated successfully.")
    print("Saved:", validation_table_path)
    print("Saved:", final_test_table_path)
    print("Saved: results/phase2/figures/validation_f1_comparison.png")
    print("Saved: results/phase2/figures/pca_cumulative_variance.png")
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()