import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results/phase2"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


# =========================
# RESULTS FROM YOUR RUN
# =========================

results = pd.DataFrame([
    ["Raw", "Logistic Regression", 0.8921, 0.8903],
    ["Raw", "Gaussian Naive Bayes", 0.4814, 0.3770],
    ["Raw", "Nearest Centroid", 0.8200, 0.8180],
    ["PCA", "Logistic Regression", 0.8840, 0.8821],
    ["PCA", "Gaussian Naive Bayes", 0.8785, 0.8776],
    ["PCA", "Nearest Centroid", 0.8162, 0.8141],
], columns=["Setting", "Model", "Accuracy", "Macro F1"])


final_cm = np.array([
    [955, 0, 2, 3, 0, 2, 10, 1, 7, 0],
    [0, 1100, 2, 4, 1, 2, 4, 0, 22, 0],
    [13, 8, 869, 26, 19, 0, 21, 22, 46, 8],
    [6, 1, 18, 890, 1, 36, 7, 15, 24, 12],
    [1, 6, 5, 0, 895, 1, 12, 2, 9, 51],
    [16, 7, 4, 49, 19, 714, 19, 11, 41, 12],
    [16, 3, 8, 2, 14, 19, 890, 1, 5, 0],
    [3, 23, 32, 3, 11, 0, 1, 907, 5, 43],
    [10, 9, 12, 32, 10, 27, 14, 15, 830, 15],
    [12, 9, 10, 12, 46, 15, 0, 27, 7, 871],
])


learning_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
train_f1 = [0.8939, 0.8878, 0.8859, 0.8860, 0.8849]
val_f1 = [0.8766, 0.8801, 0.8790, 0.8798, 0.8809]


def plot_metric(metric_name, filename):
    pivot = results.pivot(index="Model", columns="Setting", values=metric_name)

    ax = pivot.plot(kind="bar", figsize=(9, 5))

    plt.title(f"{metric_name} Comparison Before and After PCA")
    plt.xlabel("Model")
    plt.ylabel(metric_name)
    plt.xticks(rotation=20)
    plt.ylim(0, 1)
    plt.grid(axis="y")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
    plt.show()


def plot_confusion_matrix():
    plt.figure(figsize=(8, 6))
    plt.imshow(final_cm, cmap='Blues')

    plt.title("Final Model Confusion Matrix - Raw Logistic Regression")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(range(10))
    plt.yticks(range(10))

    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(final_cm[i, j]), ha="center", va="center", fontsize=7)

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "final_confusion_matrix.png"), dpi=300)
    plt.show()


def plot_learning_curve():
    plt.figure(figsize=(8, 5))

    plt.plot(learning_sizes, train_f1, marker="o", label="Train F1")
    plt.plot(learning_sizes, val_f1, marker="o", label="Validation F1")

    plt.title("Learning Curve - Logistic Regression")
    plt.xlabel("Training Data Fraction")
    plt.ylabel("Macro F1 Score")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "learning_curve.png"), dpi=300)
    plt.show()


def main():
    plot_metric("Accuracy", "accuracy_comparison.png")
    plot_metric("Macro F1", "f1_comparison.png")
    plot_confusion_matrix()
    plot_learning_curve()

    print("Plots saved in:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()