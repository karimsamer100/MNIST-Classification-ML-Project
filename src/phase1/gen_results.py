import csv
import os

import matplotlib
import numpy as np

from data_module import (
    filter_binary_classes,
    load_mnist_csv,
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


matplotlib.use("Agg")

import matplotlib.pyplot as plt


CLASS_1 = 0
CLASS_2 = 1
RANDOM_STATE = 42

TRAIN_PATH = "MNIST-data/mnist_train.csv"
TEST_PATH = "MNIST-data/mnist_test.csv"

VALIDATION_SIZE = 0.2
PCA_COMPONENTS = 50

FIGURES_DIR = "results/phase1/figures"
TABLES_DIR = "results/phase1/tables"
LOGS_DIR = "results/phase1/logs"


def ensure_output_folders():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


def load_phase1_data():
    X_train_full, y_train_full = load_mnist_csv(TRAIN_PATH)
    X_train_full, y_train_full = filter_binary_classes(
        X_train_full,
        y_train_full,
        class_1=CLASS_1,
        class_2=CLASS_2,
    )
    X_train_full = normalize_pixels(X_train_full)

    X_train, _, y_train, _ = split_train_validation(
        X_train_full,
        y_train_full,
        val_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    X_test, y_test = load_mnist_csv(TEST_PATH)
    X_test, y_test = filter_binary_classes(
        X_test,
        y_test,
        class_1=CLASS_1,
        class_2=CLASS_2,
    )
    X_test = normalize_pixels(X_test)

    return X_train, y_train, X_test, y_test


def make_models():
    return {
        "KNN (k=3)": KNN(k=3),
        "Logistic Regression": LogisticRegression(
            learning_rate=0.1,
            num_iterations=1000,
        ),
        "Gaussian Naive Bayes": GaussianNaiveBayes(),
    }


def predict_knn_in_batches(model, X_eval, batch_size=128):
    train_squared_norms = np.sum(model.X_train ** 2, axis=1)
    predictions = []

    for start in range(0, len(X_eval), batch_size):
        end = start + batch_size
        X_batch = X_eval[start:end]
        batch_squared_norms = np.sum(X_batch ** 2, axis=1, keepdims=True)

        distances_squared = (
            batch_squared_norms
            + train_squared_norms[None, :]
            - 2 * np.dot(X_batch, model.X_train.T)
        )
        np.maximum(distances_squared, 0, out=distances_squared)

        nearest_indices = np.argpartition(
            distances_squared,
            kth=model.k - 1,
            axis=1,
        )[:, :model.k]
        nearest_labels = model.y_train[nearest_indices]

        batch_predictions = (np.sum(nearest_labels, axis=1) > (model.k / 2)).astype(int)
        predictions.append(batch_predictions)

    return np.concatenate(predictions)


def predict_model(model, X_eval):
    if isinstance(model, KNN):
        return predict_knn_in_batches(model, X_eval)
    return model.predict(X_eval)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = predict_model(model, X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score_binary(y_test, predictions),
        "recall": recall_score_binary(y_test, predictions),
        "f1": f1_score_binary(y_test, predictions),
        "confusion_matrix": confusion_matrix_binary(y_test, predictions),
    }


def evaluate_all_models(X_train, y_train, X_test, y_test):
    results = {}
    for model_name, model in make_models().items():
        results[model_name] = evaluate_model(model, X_train, y_train, X_test, y_test)
    return results


def write_final_results_table(raw_results, pca_results):
    output_path = os.path.join(TABLES_DIR, "final_results.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Setting", "Model", "Accuracy", "Precision", "Recall", "F1"])

        for setting_name, results in (("Raw", raw_results), ("PCA", pca_results)):
            for model_name, metrics in results.items():
                writer.writerow([
                    setting_name,
                    model_name,
                    round(metrics["accuracy"], 4),
                    round(metrics["precision"], 4),
                    round(metrics["recall"], 4),
                    round(metrics["f1"], 4),
                ])

    return output_path


def save_pca_cumulative_variance_plot(pca_model):
    cumulative_variance = np.cumsum(pca_model.explained_variance_ratio)
    components = np.arange(1, len(cumulative_variance) + 1)
    final_variance = cumulative_variance[-1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(components, cumulative_variance, color="#1f77b4", linewidth=2.2)
    ax.scatter([PCA_COMPONENTS], [final_variance], color="#d62728", s=60, zorder=3)
    ax.axvline(PCA_COMPONENTS, color="#d62728", linestyle="--", linewidth=1.3)
    ax.axhline(final_variance, color="#2ca02c", linestyle="--", linewidth=1.3)
    ax.annotate(
        f"{final_variance * 100:.2f}% at {PCA_COMPONENTS} components",
        xy=(PCA_COMPONENTS, final_variance),
        xytext=(30, -28),
        textcoords="offset points",
        fontsize=10,
        color="#111827",
        arrowprops={"arrowstyle": "->", "color": "#6b7280"},
    )

    ax.set_title("Phase 1 PCA Cumulative Explained Variance", fontsize=13)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_xlim(1, PCA_COMPONENTS)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)

    # FIX: single save path only — no duplicate legacy copy
    output_path = os.path.join(FIGURES_DIR, "section4_pca_cumulative_variance.png")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path, final_variance


def draw_confusion_matrix_heatmap(ax, confusion_matrix, title, vmax):
    ax.imshow(confusion_matrix, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks([0, 1], labels=[f"Pred {CLASS_1}", f"Pred {CLASS_2}"])
    ax.set_yticks([0, 1], labels=[f"Actual {CLASS_1}", f"Actual {CLASS_2}"])

    threshold = vmax / 2
    for row in range(confusion_matrix.shape[0]):
        for col in range(confusion_matrix.shape[1]):
            value = confusion_matrix[row, col]
            text_color = "white" if value > threshold else "#111827"
            ax.text(
                col,
                row,
                f"{value}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold",
            )


def save_confusion_matrix_heatmaps(raw_results, pca_results):
    selected_matrices = [
        ("Logistic Regression (Raw)",      raw_results["Logistic Regression"]["confusion_matrix"]),
        ("KNN k=3 (Raw)",                  raw_results["KNN (k=3)"]["confusion_matrix"]),
        ("Gaussian Naive Bayes (Raw)",     raw_results["Gaussian Naive Bayes"]["confusion_matrix"]),
        ("Gaussian Naive Bayes (PCA)",     pca_results["Gaussian Naive Bayes"]["confusion_matrix"]),
    ]
    vmax = max(matrix.max() for _, matrix in selected_matrices)

    # FIX: use constrained_layout so the colorbar never overlaps the subplot titles
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    images = []
    for ax, (title, matrix) in zip(axes.flat, selected_matrices):
        img = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        images.append(img)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xticks([0, 1], labels=[f"Pred {CLASS_1}", f"Pred {CLASS_2}"])
        ax.set_yticks([0, 1], labels=[f"Actual {CLASS_1}", f"Actual {CLASS_2}"])

        threshold = vmax / 2
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                text_color = "white" if value > threshold else "#111827"
                ax.text(col, row, str(value),
                        ha="center", va="center",
                        color=text_color, fontsize=11, fontweight="bold")

    fig.suptitle("Phase 1 Test Confusion Matrix Heatmaps", fontsize=14)

    # FIX: attach colorbar to the figure using the last image; constrained_layout
    # handles all spacing automatically so no title/label gets clipped
    cbar = fig.colorbar(images[-1], ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Sample count", fontsize=10)

    fig.text(0.5, -0.02,
             "Rows show actual labels and columns show predicted labels.",
             ha="center", fontsize=9, color="#4b5563")

    # FIX: single save path only
    output_path = os.path.join(FIGURES_DIR, "section7_3_confusion_matrix_heatmaps.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_accuracy_comparison_plot(raw_results, pca_results):
    model_names = list(raw_results.keys())
    raw_accuracies = [raw_results[name]["accuracy"] for name in model_names]
    pca_accuracies = [pca_results[name]["accuracy"] for name in model_names]

    x_positions = np.arange(len(model_names))
    bar_width = 0.36
    lower_bound = min(raw_accuracies + pca_accuracies) - 0.01

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    raw_bars = ax.bar(
        x_positions - bar_width / 2,
        raw_accuracies,
        bar_width,
        label="Raw Features",
        color="#1f77b4",
    )
    pca_bars = ax.bar(
        x_positions + bar_width / 2,
        pca_accuracies,
        bar_width,
        label="PCA Features",
        color="#ff7f0e",
    )

    for bars in (raw_bars, pca_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.00035,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("Phase 1 Test Accuracy: Raw vs PCA Features", fontsize=13)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x_positions, model_names)
    ax.set_ylim(lower_bound, 1.001)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.text(
        0.99, 0.02,
        "Y-axis is zoomed to make small accuracy differences visible.",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8.5, color="#4b5563",
    )

    # FIX: single save path only
    output_path = os.path.join(FIGURES_DIR, "section8_accuracy_comparison.png")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def write_summary_log(raw_results, pca_results, explained_variance, figure_paths):
    output_path = os.path.join(LOGS_DIR, "phase1_plots_summary.txt")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("Phase 1 Plot Summary\n")
        file.write("====================\n\n")
        file.write(
            f"Section 4 figure: {figure_paths['pca_plot']} "
            f"(50 components retain {explained_variance * 100:.2f}% variance)\n"
        )
        file.write(f"Section 7.3 figure: {figure_paths['confusion_plot']}\n")
        file.write(f"Section 8 figure:   {figure_paths['accuracy_plot']}\n\n")

        file.write("Test Accuracy Comparison\n")
        file.write("------------------------\n")
        for model_name in raw_results:
            file.write(
                f"{model_name}: raw={raw_results[model_name]['accuracy']:.4f}, "
                f"pca={pca_results[model_name]['accuracy']:.4f}\n"
            )

    return output_path


def main():
    ensure_output_folders()

    X_train, y_train, X_test, y_test = load_phase1_data()

    raw_test_results = evaluate_all_models(X_train, y_train, X_test, y_test)

    X_train_pca, _, X_test_pca, pca_model = apply_pca(
        X_train,
        X_test,
        X_test,
        n_components=PCA_COMPONENTS,
    )
    pca_test_results = evaluate_all_models(X_train_pca, y_train, X_test_pca, y_test)

    results_table_path = write_final_results_table(raw_test_results, pca_test_results)
    pca_plot_path, explained_variance = save_pca_cumulative_variance_plot(pca_model)
    confusion_plot_path = save_confusion_matrix_heatmaps(raw_test_results, pca_test_results)
    accuracy_plot_path = save_accuracy_comparison_plot(raw_test_results, pca_test_results)
    summary_log_path = write_summary_log(
        raw_test_results,
        pca_test_results,
        explained_variance,
        {
            "pca_plot":       pca_plot_path,
            "confusion_plot": confusion_plot_path,
            "accuracy_plot":  accuracy_plot_path,
        },
    )

    print("Phase 1 report figures generated successfully.")
    print("Saved:", results_table_path)
    print("Saved:", pca_plot_path)
    print("Saved:", confusion_plot_path)
    print("Saved:", accuracy_plot_path)
    print("Saved:", summary_log_path)


if __name__ == "__main__":
    main()