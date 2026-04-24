# MNIST Image Classification — From-Scratch ML Pipeline

CSE382: Introduction to Machine Learning — Spring 2026
Major Task Project | Phases 1 & 2

---

## Project Overview

A complete machine learning pipeline built entirely from scratch to classify handwritten digits from the MNIST dataset. The project is structured in two phases: binary classification (Phase 1) and full multi-class classification (Phase 2). No external ML libraries are used — all models, preprocessing, PCA, and evaluation metrics are implemented manually.

---

## Project Structure

```
ML Project/
├── MNIST-data/
│   ├── mnist_train.csv
│   └── mnist_test.csv
├── docs/
│   ├── phase1/
│   │   ├── phase1_Project_Description.md
│   │   └── phase1_Tasks_Plan.md
│   └── phase2/
│       ├── Phase2_Description.md
│       └── Phase2_Workflow_Guide.md
├── results/
│   ├── phase1/
│   │   ├── figures/
│   │   └── tables/
│   └── phase2/
│       ├── figures/
│       ├── logs/
│       └── tables/
└── src/
    ├── phase1/
    │   ├── data_module.py
    │   ├── evaluation_module.py
    │   ├── features_module.py
    │   ├── gen_results.py
    │   ├── main.py
    │   ├── models_module.py
    │   └── test_pipeline.py
    └── phase2/
        ├── data_module.py
        ├── evaluation_module.py
        ├── features_module.py
        ├── gen_results.py
        ├── main.py
        ├── models_module.py
        └── test_pipeline.py
```

---

## Phase 1 — Binary Classification (Digit 0 vs Digit 1)

### Objective

Distinguish between digit 0 and digit 1 using three from-scratch classifiers, with and without PCA dimensionality reduction.

### Dataset Split

| Set        | Samples | Class 0 | Class 1 |
|------------|---------|---------|---------|
| Train      | 10,133  | 4,739   | 5,394   |
| Validation | 2,532   | 1,184   | 1,348   |
| Test       | 2,115   | 980     | 1,135   |

Stratified split, random state = 42. Pixels normalized to [0, 1].

### Models Implemented

- K-Nearest Neighbors (KNN) — Euclidean distance, majority vote
- Logistic Regression — sigmoid, gradient descent, binary cross-entropy
- Gaussian Naive Bayes — class-conditional Gaussian likelihood, log-space inference

### PCA

- Components kept: 50 (from 784)
- Total explained variance: 90.37%
- Top component variance ratios: [0.3220, 0.0904, 0.0812, 0.0549, 0.0390, ...]

### Results — Validation Set

| Model               | Setting  | Accuracy | Precision | Recall | F1     |
|---------------------|----------|----------|-----------|--------|--------|
| KNN (k=3)           | Raw      | 0.9992   | 0.9985    | 1.0000 | 0.9993 |
| KNN (k=3)           | PCA      | 0.9992   | 0.9985    | 1.0000 | 0.9993 |
| Logistic Regression | Raw      | 0.9972   | 0.9978    | 0.9970 | 0.9974 |
| Logistic Regression | PCA      | 0.9976   | 0.9985    | 0.9970 | 0.9978 |
| Gaussian Naive Bayes| Raw      | 0.9937   | 0.9941    | 0.9941 | 0.9941 |
| Gaussian Naive Bayes| PCA      | 0.9787   | 1.0000    | 0.9599 | 0.9796 |

### Results — Test Set

| Model               | Setting  | Accuracy | Precision | Recall | F1     |
|---------------------|----------|----------|-----------|--------|--------|
| KNN (k=3)           | Raw      | 0.9991   | 0.9982    | 1.0000 | 0.9991 |
| KNN (k=3)           | PCA      | 0.9991   | 0.9982    | 1.0000 | 0.9991 |
| Logistic Regression | Raw      | 0.9995   | 0.9991    | 1.0000 | 0.9996 |
| Logistic Regression | PCA      | 0.9991   | 0.9982    | 1.0000 | 0.9991 |
| Gaussian Naive Bayes| Raw      | 0.9972   | 0.9974    | 0.9974 | 0.9974 |
| Gaussian Naive Bayes| PCA      | 0.9844   | 1.0000    | 0.9709 | 0.9852 |

### Key Observations — Phase 1

- All three models achieve near-perfect performance on this binary task due to high visual separability between digits 0 and 1.
- PCA has minimal effect on KNN and Logistic Regression at this scale — performance is essentially unchanged.
- Gaussian Naive Bayes degrades slightly after PCA on validation (0.9937 → 0.9787), recovering partially on the test set (0.9972 → 0.9844). This is because PCA introduces correlations among components, partially violating the independence assumption.
- Logistic Regression achieves the highest raw test F1 (0.9996), slightly outperforming KNN.

---

## Phase 2 — Multi-Class Classification (Digits 0–9)

### Objective

Scale the pipeline to 10 classes and improve performance through three techniques: PCA, hyperparameter tuning (KNN), and L2 regularization (Logistic Regression). Learning curve analysis is also included.

### Dataset Split

| Set        | Samples | Classes  |
|------------|---------|----------|
| Train      | 48,000  | 0–9      |
| Validation | 12,000  | 0–9      |
| Test       | 10,000  | 0–9      |

Stratified split, random state = 42. Pixels normalized to [0, 1].

### Models Implemented

- KNN — extended to multi-class majority vote
- Multiclass Logistic Regression — softmax activation, categorical cross-entropy, gradient descent with optional L2 regularization
- Gaussian Naive Bayes — extended to 10 classes with per-class Gaussian statistics

### PCA

- Components kept: 50 (from 784)
- Fit on training data only; val/test transformed using training mean and components.

### Improvement Techniques Applied

**1. Hyperparameter Tuning — KNN**

Tested k values {1, 3, 5, 7} on the validation set. Best result: k = 1.

| k | Validation Macro F1 |
|---|---------------------|
| 1 | ~0.92               |
| 3 | ~0.91               |
| 5 | ~0.90               |
| 7 | ~0.89               |

**2. PCA — Dimensionality Reduction**

| Model               | Raw F1 | PCA F1 | Change  |
|---------------------|--------|--------|---------|
| KNN (k=1)           | ~0.91  | ~0.92  | +0.01   |
| Logistic Regression | ~0.90  | ~0.90  | ~0.00   |
| Gaussian Naive Bayes| ~0.35  | ~0.87  | +0.52   |

PCA dramatically improves Naive Bayes by reducing feature correlation, making the independence assumption more valid. KNN improves slightly. Logistic Regression is unaffected.

**3. L2 Regularization — Logistic Regression**

Tested lambda values {0.0, 0.01, 0.1, 1.0}. No meaningful improvement was observed, indicating the model is not overfitting at the tested training sizes.

**4. Learning Curve Analysis**

Training sizes {200, 500, 1000, 1500, 2000} were evaluated for KNN and Logistic Regression. KNN (k=1) shows a classic overfitting signature — training F1 = 1.0 at all sizes, with a persistent gap to validation F1. Logistic Regression shows steadily converging train/val curves, indicating better generalization behavior.

### Final Model Selection

Model selected on validation performance: **KNN (k=1) with PCA (50 components)**.

### Final Test Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9533 |
| Precision | 0.9539 |
| Recall    | 0.9533 |
| F1 Score  | 0.9533 |

Test set evaluated once, after model selection was finalized on validation only.

---

## Evaluation Metrics (All Implemented Manually)

- Accuracy
- Precision, Recall, F1-score (binary and macro-averaged multi-class)
- Confusion matrix (2×2 binary, 10×10 multi-class)

---

## Technologies

- Python 3
- NumPy
- Pandas
- Matplotlib

---

## How to Run

```bash
# Phase 1 pipeline
python src/phase1/main.py

# Phase 2 pipeline
python src/phase2/main.py

# Generate result tables and figures
python src/phase1/gen_results.py
python src/phase2/gen_results.py
```

---

## Author

Karim Samer
Computer Engineering — CAIE Program