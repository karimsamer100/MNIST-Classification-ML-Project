# 🧠 MNIST Image Classification — From Scratch ML Pipeline

> A complete end-to-end Machine Learning system built **from scratch** to classify handwritten digits using classical ML algorithms — with advanced optimization and feature engineering.

---

## 🚀 Project Overview

This project implements a **full machine learning pipeline** for image classification on the MNIST dataset, following a rigorous engineering and scientific approach.

It was developed as part of the **CSE382: Introduction to Machine Learning** major project, focusing on:

* Mathematical understanding
* Implementation from scratch
* Experimental evaluation
* Model improvement techniques

---

## 🎯 Problem Statement

* **Input:** 28×28 grayscale image → flattened into 784 features
* **Output:** Digit label ∈ {0, 1, 2, ..., 9}
* **Task:** Multi-class classification

---

## 🧩 Key Features

✅ Built **completely from scratch** (no sklearn models)
✅ Modular pipeline design
✅ Multiple ML algorithms implemented manually
✅ Feature engineering using PCA
✅ Hyperparameter tuning
✅ Regularization analysis
✅ Clean evaluation pipeline
✅ Visualization + automated results generation

---

## 🏗️ Project Architecture

```
📁 src/
│
├── data_module.py        # data loading, preprocessing, splitting
├── models_module.py      # ML models (KNN, Logistic, Naive Bayes)
├── features_module.py    # PCA implementation
├── evaluation_module.py  # metrics and evaluation
│
├── test_pipeline.py      # experiments & debugging
├── main.py               # final clean pipeline (for demo)
├── gen_results.py        # result generation (tables + plots)
│
📁 results/
├── tables/
├── figures/
├── logs/
```

---

## ⚙️ Implemented Models

### 🔹 K-Nearest Neighbors (KNN)

* Distance-based classification
* Fully implemented from scratch
* Supports multi-class voting

### 🔹 Multiclass Logistic Regression

* Softmax-based classification
* Gradient descent optimization
* Supports L2 Regularization

### 🔹 Gaussian Naive Bayes

* Probabilistic classifier
* Assumes feature independence
* Uses Gaussian likelihood estimation

---

## 📊 Feature Engineering — PCA

Custom PCA implementation:

* Eigen decomposition of covariance matrix
* Dimensionality reduction from **784 → 50 features**
* Preserves ~**83% of total variance**

> PCA significantly reduces feature correlation and improves certain models.

---

## 🔬 Experimental Workflow

1. Data loading and normalization
2. Stratified train/validation/test split
3. Baseline model evaluation
4. Hyperparameter tuning (KNN)
5. Feature engineering (PCA)
6. Model comparison
7. Final model selection
8. Final test evaluation

---

## 📈 Results Summary

### 🧪 Validation Performance (Macro F1)

| Model                | Raw Features | PCA Features |
| -------------------- | ------------ | ------------ |
| KNN (k=1)            | ~0.91        | ~0.92        |
| Logistic Regression  | ~0.90        | ~0.90        |
| Gaussian Naive Bayes | ~0.35        | ~0.87 🔥     |

---

## 🏆 Final Model

**KNN (k=1) + PCA (50 Components)**

### 📊 Final Test Results

* **Accuracy:** 0.9533
* **Precision:** 0.9539
* **Recall:** 0.9533
* **F1 Score:** 0.9533

> ⚠️ Test set was used **only once** after model selection to avoid data leakage.

---

## 🔍 Key Insights

### 1️⃣ PCA is not just dimensionality reduction

It also **removes feature correlation**, which:

* Dramatically improves Gaussian Naive Bayes
* Slightly improves KNN
* May not benefit Logistic Regression

---

### 2️⃣ Model assumptions matter

Naive Bayes improved significantly:

```
F1 Score: 0.35 → 0.87
```

Because PCA made features more independent.

---

### 3️⃣ Best KNN performance occurs with small k

* Best k = 1
* Increasing k reduced performance

---

### 4️⃣ Regularization had no noticeable impact

* Indicates **no overfitting**
* Dataset already well-structured

---

## 📊 Visualizations

### 📈 PCA Explained Variance

* 50 components retain ~83% variance
* Strong redundancy exists in original feature space

### 📊 Model Comparison

* PCA dramatically boosts Naive Bayes
* Minor impact on other models

---

## 🧪 Evaluation Metrics

All implemented from scratch:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Supports **multi-class evaluation**.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib

---

## ⚠️ Important Notes

* No external ML libraries (e.g., sklearn models) were used
* Entire pipeline implemented manually
* Test set used only once for final evaluation
* Final results computed on a stratified subset for efficiency

---

## 🧠 What Makes This Project Strong

✔ Full ML pipeline from scratch
✔ Strong mathematical understanding
✔ Clean and modular implementation
✔ Real experimental evaluation
✔ Clear performance improvements
✔ Professional visualizations
✔ No black-box models

---

## 🚀 How to Run

```bash
# Run final pipeline
python src/phase2/main.py

# Generate results (tables + figures)
python src/phase2/gen_results.py
```

---

## 👨‍💻 Author

**Karim Samer**

* Computer Engineering Student — AI Track
* Passionate about Machine Learning, Systems, and Problem Solving

---

## ⭐ Final Note

This project is not just about achieving high accuracy —
it is about **understanding, building, and analyzing machine learning systems from the ground up.**
