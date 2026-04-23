import numpy as np
from features_module import apply_pca
from data_module import load_mnist_csv, normalize_pixels, split_train_validation
from models_module import KNN, GaussianNaiveBayes, MulticlassLogisticRegression
from evaluation_module import (
    accuracy_score,
    confusion_matrix_multiclass,
    precision_recall_f1_multiclass,
    classification_report_multiclass,
    print_confusion_matrix
)
# =========================
# LOAD DATA
# =========================

X_train_full, y_train_full = load_mnist_csv("MNIST-data/mnist_train.csv")
X_test, y_test = load_mnist_csv("MNIST-data/mnist_test.csv")

X_train_full = normalize_pixels(X_train_full)
X_test = normalize_pixels(X_test)

X_train, X_val, y_train, y_val = split_train_validation(X_train_full, y_train_full)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

print("Unique train labels:", np.unique(y_train))
print("Unique validation labels:", np.unique(y_val))
print("Unique test labels:", np.unique(y_test))

# =========================
# SMALL SUBSETS FOR QUICK TESTING
# =========================

rng = np.random.default_rng(42)

train_indices = rng.choice(len(X_train), size=2000, replace=False)
val_indices = rng.choice(len(X_val), size=300, replace=False)

X_train_subset = X_train[train_indices]
y_train_subset = y_train[train_indices]

X_val_subset = X_val[val_indices]
y_val_subset = y_val[val_indices]

print("\nSubset shapes:")
print("Train subset:", X_train_subset.shape)
print("Validation subset:", X_val_subset.shape)

print("Train subset labels:", np.unique(y_train_subset))
print("Validation subset labels:", np.unique(y_val_subset))

# =========================
# KNN TEST
# =========================

knn = KNN(k=3)
knn.fit(X_train_subset, y_train_subset)

knn_single = knn.predict_one(X_val_subset[0])
knn_preds = knn.predict(X_val_subset[:10])

print("\nKNN test:")
print("Single predicted label:", knn_single)
print("Actual label:", y_val_subset[0])
print("First 10 predictions:", knn_preds)
print("First 10 actual labels:", y_val_subset[:10])

# =========================
# GAUSSIAN NAIVE BAYES TEST
# =========================

gnb = GaussianNaiveBayes()
gnb.fit(X_train_subset, y_train_subset)

gnb_single = gnb.predict_one(X_val_subset[0])
gnb_preds = gnb.predict(X_val_subset[:10])

print("\nGaussian Naive Bayes test:")
print("Classes:", gnb.classes)
print("Mean shape:", gnb.mean.shape)
print("Variance shape:", gnb.variance.shape)
print("Single predicted label:", gnb_single)
print("Actual label:", y_val_subset[0])
print("First 10 predictions:", gnb_preds)
print("First 10 actual labels:", y_val_subset[:10])

# =========================
# MULTICLASS LOGISTIC REGRESSION TEST
# =========================

log_reg = MulticlassLogisticRegression(learning_rate=0.1, num_iterations=300)
log_reg.fit(X_train_subset, y_train_subset)

log_single = log_reg.predict(X_val_subset[:1])
log_preds = log_reg.predict(X_val_subset[:10])

print("\nMulticlass Logistic Regression test:")
print("Weights shape:", log_reg.weights.shape)
print("Bias shape:", log_reg.bias.shape)
print("Classes:", log_reg.classes)
print("Single predicted label:", log_single[0])
print("Actual label:", y_val_subset[0])
print("First 10 predictions:", log_preds)
print("First 10 actual labels:", y_val_subset[:10])

# =========================
# BASELINE EVALUATION ON VALIDATION SUBSET
# =========================

print("\n=========================")
print("BASELINE VALIDATION EVALUATION")
print("=========================")

# KNN evaluation
knn_val_predictions = knn.predict(X_val_subset)
knn_accuracy = accuracy_score(y_val_subset, knn_val_predictions)
knn_precision, knn_recall, knn_f1 = precision_recall_f1_multiclass(
    y_val_subset, knn_val_predictions, average="macro"
)

print("\nKNN Validation Results:")
print("Accuracy :", round(knn_accuracy, 4))
print("Precision:", round(knn_precision, 4))
print("Recall   :", round(knn_recall, 4))
print("F1 Score :", round(knn_f1, 4))

# Gaussian Naive Bayes evaluation
gnb_val_predictions = gnb.predict(X_val_subset)
gnb_accuracy = accuracy_score(y_val_subset, gnb_val_predictions)
gnb_precision, gnb_recall, gnb_f1 = precision_recall_f1_multiclass(
    y_val_subset, gnb_val_predictions, average="macro"
)

print("\nGaussian Naive Bayes Validation Results:")
print("Accuracy :", round(gnb_accuracy, 4))
print("Precision:", round(gnb_precision, 4))
print("Recall   :", round(gnb_recall, 4))
print("F1 Score :", round(gnb_f1, 4))

# Logistic Regression evaluation
log_val_predictions = log_reg.predict(X_val_subset)
log_accuracy = accuracy_score(y_val_subset, log_val_predictions)
log_precision, log_recall, log_f1 = precision_recall_f1_multiclass(
    y_val_subset, log_val_predictions, average="macro"
)

print("\nMulticlass Logistic Regression Validation Results:")
print("Accuracy :", round(log_accuracy, 4))
print("Precision:", round(log_precision, 4))
print("Recall   :", round(log_recall, 4))
print("F1 Score :", round(log_f1, 4))

# optional: print one confusion matrix example
print("\nKNN confusion matrix example:")
print_confusion_matrix(y_val_subset, knn_val_predictions, num_classes=10)

# optional: class-by-class report for logistic regression
classification_report_multiclass(y_val_subset, log_val_predictions, num_classes=10)
# =========================
# KNN HYPERPARAMETER TUNING
# =========================

print("\n=========================")
print("KNN HYPERPARAMETER TUNING")
print("=========================")

k_values = [1, 3, 5, 7]
knn_results = {}

for k in k_values:
    knn_model = KNN(k=k)
    knn_model.fit(X_train_subset, y_train_subset)

    knn_predictions = knn_model.predict(X_val_subset)

    accuracy = accuracy_score(y_val_subset, knn_predictions)
    precision, recall, f1 = precision_recall_f1_multiclass(
        y_val_subset, knn_predictions, average="macro"
    )

    knn_results[k] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

for k, metrics in knn_results.items():
    print(f"\nK = {k}")
    print("Accuracy :", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall   :", round(metrics["recall"], 4))
    print("F1 Score :", round(metrics["f1"], 4))

best_k = max(knn_results, key=lambda k: knn_results[k]["f1"])

print("\nBest K based on macro F1:", best_k)
print("Best K validation F1:", round(knn_results[best_k]["f1"], 4))

# =========================
# PCA TEST
# =========================

print("\n=========================")
print("PCA TEST")
print("=========================")

X_train_pca, X_val_pca, X_test_pca, pca_model = apply_pca(
    X_train_subset,
    X_val_subset,
    X_test[:300],
    n_components=50
)

print("Original train shape:", X_train_subset.shape)
print("PCA train shape:", X_train_pca.shape)

print("Original validation shape:", X_val_subset.shape)
print("PCA validation shape:", X_val_pca.shape)

print("Components shape:", pca_model.components.shape)
print("Explained variance shape:", pca_model.explained_variance.shape)
print("Explained variance ratio shape:", pca_model.explained_variance_ratio.shape)

print("First 10 explained variance ratios:", pca_model.explained_variance_ratio[:10])
print("Total explained variance ratio:", round(np.sum(pca_model.explained_variance_ratio), 4))

# =========================
# MODELS AFTER PCA
# =========================

print("\n=========================")
print("MODELS AFTER PCA")
print("=========================")

# KNN with best k from tuning
knn_pca = KNN(k=1)
knn_pca.fit(X_train_pca, y_train_subset)
knn_pca_predictions = knn_pca.predict(X_val_pca)

knn_pca_accuracy = accuracy_score(y_val_subset, knn_pca_predictions)
knn_pca_precision, knn_pca_recall, knn_pca_f1 = precision_recall_f1_multiclass(
    y_val_subset, knn_pca_predictions, average="macro"
)

print("\nKNN (PCA) Results:")
print("Accuracy :", round(knn_pca_accuracy, 4))
print("Precision:", round(knn_pca_precision, 4))
print("Recall   :", round(knn_pca_recall, 4))
print("F1 Score :", round(knn_pca_f1, 4))


# Gaussian Naive Bayes with PCA
gnb_pca = GaussianNaiveBayes()
gnb_pca.fit(X_train_pca, y_train_subset)
gnb_pca_predictions = gnb_pca.predict(X_val_pca)

gnb_pca_accuracy = accuracy_score(y_val_subset, gnb_pca_predictions)
gnb_pca_precision, gnb_pca_recall, gnb_pca_f1 = precision_recall_f1_multiclass(
    y_val_subset, gnb_pca_predictions, average="macro"
)

print("\nGaussian Naive Bayes (PCA) Results:")
print("Accuracy :", round(gnb_pca_accuracy, 4))
print("Precision:", round(gnb_pca_precision, 4))
print("Recall   :", round(gnb_pca_recall, 4))
print("F1 Score :", round(gnb_pca_f1, 4))


# Logistic Regression with PCA
log_pca = MulticlassLogisticRegression(learning_rate=0.1, num_iterations=300)
log_pca.fit(X_train_pca, y_train_subset)
log_pca_predictions = log_pca.predict(X_val_pca)

log_pca_accuracy = accuracy_score(y_val_subset, log_pca_predictions)
log_pca_precision, log_pca_recall, log_pca_f1 = precision_recall_f1_multiclass(
    y_val_subset, log_pca_predictions, average="macro"
)

print("\nMulticlass Logistic Regression (PCA) Results:")
print("Accuracy :", round(log_pca_accuracy, 4))
print("Precision:", round(log_pca_precision, 4))
print("Recall   :", round(log_pca_recall, 4))
print("F1 Score :", round(log_pca_f1, 4))


print("\n=========================")
print("BEFORE VS AFTER PCA")
print("=========================")

print("\nKNN:")
print("Before PCA F1:", round(knn_results[best_k]["f1"], 4))
print("After PCA F1 :", round(knn_pca_f1, 4))

print("\nGaussian Naive Bayes:")
print("Before PCA F1:", round(gnb_f1, 4))
print("After PCA F1 :", round(gnb_pca_f1, 4))

print("\nLogistic Regression:")
print("Before PCA F1:", round(log_f1, 4))
print("After PCA F1 :", round(log_pca_f1, 4))

# =========================
# LOGISTIC REGRESSION WITH L2 REGULARIZATION
# =========================

print("\n=========================")
print("LOGISTIC REGRESSION (L2 REGULARIZATION)")
print("=========================")

lambda_values = [0.0, 0.01, 0.1, 1.0]

log_reg_results = {}

for lam in lambda_values:
    model = MulticlassLogisticRegression(
        learning_rate=0.1,
        num_iterations=300,
        lambda_reg=lam
    )

    model.fit(X_train_subset, y_train_subset)
    predictions = model.predict(X_val_subset)

    accuracy = accuracy_score(y_val_subset, predictions)
    precision, recall, f1 = precision_recall_f1_multiclass(
        y_val_subset, predictions, average="macro"
    )

    log_reg_results[lam] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

for lam, metrics in log_reg_results.items():
    print(f"\nLambda = {lam}")
    print("Accuracy :", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall   :", round(metrics["recall"], 4))
    print("F1 Score :", round(metrics["f1"], 4))

best_lambda = max(log_reg_results, key=lambda x: log_reg_results[x]["f1"])

print("\nBest Lambda based on F1:", best_lambda)
print("Best F1:", round(log_reg_results[best_lambda]["f1"], 4))



import matplotlib.pyplot as plt
import os


def evaluate_on_same_split(model, X_train_small, y_train_small, X_val, y_val):
    model.fit(X_train_small, y_train_small)

    train_predictions = model.predict(X_train_small)
    val_predictions = model.predict(X_val)

    train_accuracy = accuracy_score(y_train_small, train_predictions)
    val_accuracy = accuracy_score(y_val, val_predictions)

    _, _, train_f1 = precision_recall_f1_multiclass(
        y_train_small, train_predictions, average="macro"
    )
    _, _, val_f1 = precision_recall_f1_multiclass(
        y_val, val_predictions, average="macro"
    )

    return {
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "train_f1": train_f1,
        "val_f1": val_f1
    }


# =========================
# LEARNING CURVES
# =========================

print("\n=========================")
print("LEARNING CURVES")
print("=========================")

# training sizes to study overfitting / underfitting
train_sizes = [200, 500, 1000, 1500, 2000]

knn_train_f1_scores = []
knn_val_f1_scores = []

log_train_f1_scores = []
log_val_f1_scores = []

# make sure output folder exists
os.makedirs("results/phase2/figures", exist_ok=True)

for size in train_sizes:
    X_small = X_train_subset[:size]
    y_small = y_train_subset[:size]

    # KNN (best tuned value)
    knn_model = KNN(k=1)
    knn_scores = evaluate_on_same_split(
        knn_model,
        X_small, y_small,
        X_val_subset, y_val_subset
    )

    knn_train_f1_scores.append(knn_scores["train_f1"])
    knn_val_f1_scores.append(knn_scores["val_f1"])

    # Logistic Regression
    log_model = MulticlassLogisticRegression(
        learning_rate=0.1,
        num_iterations=300,
        lambda_reg=0.0
    )
    log_scores = evaluate_on_same_split(
        log_model,
        X_small, y_small,
        X_val_subset, y_val_subset
    )

    log_train_f1_scores.append(log_scores["train_f1"])
    log_val_f1_scores.append(log_scores["val_f1"])

    print(f"\nTraining Size = {size}")
    print(f"KNN -> Train F1: {knn_scores['train_f1']:.4f}, Val F1: {knn_scores['val_f1']:.4f}")
    print(f"Logistic Regression -> Train F1: {log_scores['train_f1']:.4f}, Val F1: {log_scores['val_f1']:.4f}")

# -------------------------
# plot KNN learning curve
# -------------------------
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, knn_train_f1_scores, marker="o", label="Training F1")
plt.plot(train_sizes, knn_val_f1_scores, marker="o", label="Validation F1")
plt.xlabel("Training Set Size")
plt.ylabel("Macro F1 Score")
plt.title("Learning Curve - KNN (k=1)")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/phase2/figures/learning_curve_knn.png")
plt.close()

# -------------------------
# plot Logistic Regression learning curve
# -------------------------
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, log_train_f1_scores, marker="o", label="Training F1")
plt.plot(train_sizes, log_val_f1_scores, marker="o", label="Validation F1")
plt.xlabel("Training Set Size")
plt.ylabel("Macro F1 Score")
plt.title("Learning Curve - Multiclass Logistic Regression")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/phase2/figures/learning_curve_logistic.png")
plt.close()

print("\nSaved: results/phase2/figures/learning_curve_knn.png")
print("Saved: results/phase2/figures/learning_curve_logistic.png")