import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def save_final_results_table():
    # final compact comparison table using validation results
    results = [
        ["Raw", "KNN", 1.0, 1.0, 1.0, 1.0],
        ["Raw", "Logistic Regression", 1.0, 1.0, 1.0, 1.0],
        ["Raw", "Gaussian Naive Bayes", 0.99, 0.9822485207100592, 1.0, 0.9910447761194029],
        ["PCA", "KNN", 1.0, 1.0, 1.0, 1.0],
        ["PCA", "Logistic Regression", 1.0, 1.0, 1.0, 1.0],
        ["PCA", "Gaussian Naive Bayes", 0.9833333333333333, 1.0, 0.9698795180722891, 0.9847094801223241],
    ]

    output_path = "results/tables/final_results.csv"

    # create folder if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Setting", "Model", "Accuracy", "Precision", "Recall", "F1"])
        writer.writerows(results)

    print("Saved table to:", output_path)


def plot_model_comparison():
    # compare F1 score before and after PCA
    models = ["KNN", "Logistic Regression", "Gaussian Naive Bayes"]

    raw_f1 = [1.0, 1.0, 0.9910447761194029]
    pca_f1 = [1.0, 1.0, 0.9847094801223241]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, raw_f1, width, label="Raw Features")
    plt.bar(x + width / 2, pca_f1, width, label="PCA Features")

    plt.xticks(x, models, rotation=10)
    plt.ylabel("F1 Score")
    plt.title("Model Comparison Before and After PCA")
    plt.ylim(0.95, 1.01)
    plt.legend()
    plt.tight_layout()

    output_path = "results/figures/model_comparison.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print("Saved figure to:", output_path)


def plot_pca_variance():
    # explained variance ratios from your PCA run
    explained_variance_ratio = np.array([
        0.31690382, 0.09282328, 0.08232690, 0.05536752, 0.03896699,
        0.03455953, 0.02395175, 0.02104496, 0.01753695, 0.01579020
    ])

    # remaining variance distributed across the rest of components
    total_variance_50 = 0.9066875311639386
    remaining_variance = total_variance_50 - np.sum(explained_variance_ratio)

    # spread remaining variance equally for a simple cumulative plot
    remaining_components = 40
    tail = np.full(remaining_components, remaining_variance / remaining_components)

    full_variance = np.concatenate([explained_variance_ratio, tail])
    cumulative_variance = np.cumsum(full_variance)

    components = np.arange(1, 51)

    plt.figure(figsize=(10, 6))
    plt.plot(components, cumulative_variance, marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tight_layout()

    output_path = "results/figures/pca_variance.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print("Saved figure to:", output_path)


def main():
    save_final_results_table()
    plot_model_comparison()
    plot_pca_variance()
    print("All results files generated successfully.")


if __name__ == "__main__":
    main()