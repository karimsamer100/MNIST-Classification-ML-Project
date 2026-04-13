# 🧠 MNIST Image Classification — From Scratch ML Pipeline (Phase 1 & Phase 2)

> A complete end-to-end Machine Learning system built **from scratch** to classify handwritten digits — evolving from binary classification (Phase 1) to full multi-class classification (Phase 2).

---

## 🚀 Project Overview

This project implements a **full machine learning pipeline from scratch**, developed in two progressive phases:

* **Phase 1:** Binary classification (digits 0 vs 1)
* **Phase 2:** Multi-class classification (digits 0–9) with improvements

The goal is not only to achieve good accuracy, but to:

* Understand ML algorithms deeply
* Build everything manually (no black-box models)
* Analyze model behavior
* Apply real improvement techniques

---

## 🎯 Problem Definition

* **Input:** 28×28 grayscale image → 784 features
* **Output:** Digit label

### Phase 1:

* Binary classification → {0, 1}

### Phase 2:

* Multi-class classification → {0–9}

---

## 🧩 Key Features

- ✅ Fully implemented **from scratch** (no sklearn models)
- ✅ Modular pipeline design
- - ✅ Custom ML models
- - ✅ Feature engineering (PCA)
- - ✅ Hyperparameter tuning
- - ✅ Regularization analysis
- - ✅ Multi-class evaluation
- - ✅ Automated results generation
- - ✅ Visualization & reporting

---

## 🏗️ Project Architecture


```text
ML Project/
+-- MNIST-data/
|   +-- mnist_train.csv
|   \-- mnist_test.csv
+-- docs/
|   +-- CSE382_MajorTask_Spring2026.pdf
|   +-- phase1/
|   |   +-- phase1_Project_Description.md
|   |   \-- phase1_Tasks_Plan.md
|   \-- phase2/
|       +-- Phase2_Description.md
|       \-- Phase2_Workflow_Guide.md
+-- results/
|   +-- phase1/
|   |   +-- figures/
|   |   \-- tables/
|   \-- phase2/
|       +-- figures/
|       +-- logs/
|       \-- tables/
\-- src/
    +-- phase1/
    |   +-- data_module.py
    |   +-- evaluation_module.py
    |   +-- features_module.py
    |   +-- gen_results.py
    |   +-- main.py
    |   +-- models_module.py
    |   \-- test_pipeline.py
    \-- phase2/
        +-- data_module.py
        +-- evaluation_module.py
        +-- features_module.py
        +-- gen_results.py
        +-- main.py
        +-- models_module.py
        \-- test_pipeline.py
```

---

# 🧪 Phase 1 — Binary Classification

## 🎯 Objective

Classify digits:

```
0 vs 1
```

## ⚙️ Models Used

* KNN
* Logistic Regression

## 🧠 Key Steps

* Data filtering (only classes 0 & 1)
* Normalization
* Train/validation split
* Model training
* Evaluation

## 📊 Observations

* Very high accuracy (close to 100%)
* Task is simple due to clear separation
* Helped validate correctness of implementations

---

# 🚀 Phase 2 — Multi-Class Classification

## 🎯 Objective

Classify:

```
0 → 9 (10 classes)
```

---

## ⚙️ Models Implemented

### 🔹 K-Nearest Neighbors (KNN)

* Distance-based classification
* Best performing model

### 🔹 Multiclass Logistic Regression

* Softmax implementation
* Gradient descent training
* Supports L2 Regularization

### 🔹 Gaussian Naive Bayes

* Probabilistic model
* Assumes feature independence

---

## 📊 Feature Engineering — PCA

* Reduced features:

```
784 → 50
```

* Preserved:

```
~83% variance
```

### 🧠 Why PCA?

* Reduce dimensionality
* Remove feature correlation
* Improve model assumptions

---

## 🔬 Improvements Applied

### 1️⃣ Hyperparameter Tuning (KNN)

* Tested k values: {1, 3, 5, 7}
* Best result:

```
k = 1
```

---

### 2️⃣ PCA (Feature Engineering)

* Reduced dimensionality
* Improved model performance
* Huge impact on Naive Bayes

---

### 3️⃣ L2 Regularization

* Applied to Logistic Regression
* No significant improvement
* Indicates low overfitting

---

## 📈 Results Summary

### 🧪 Validation (Macro F1)

| Model                | Raw   | PCA      |
| -------------------- | ----- | -------- |
| KNN (k=1)            | ~0.91 | ~0.92    |
| Logistic Regression  | ~0.90 | ~0.90    |
| Gaussian Naive Bayes | ~0.35 | ~0.87 🔥 |

---

## 🏆 Final Model

**KNN (k=1) + PCA**

---

## 📊 Final Test Results

* **Accuracy:** 0.9533
* **Precision:** 0.9539
* **Recall:** 0.9533
* **F1 Score:** 0.9533

> Test set used only once after model selection (no data leakage).

---

## 🔍 Key Insights

### 🔥 1. PCA dramatically improves Naive Bayes

```
F1: 0.35 → 0.87
```

Reason:

* PCA reduces feature correlation
* Makes independence assumption more valid

---

### 📉 2. PCA has mixed effects

* Improves KNN slightly
* Slightly reduces Logistic Regression

---

### 🎯 3. Best KNN uses small k

```
k = 1 performs best
```

---

### 🧠 4. Regularization not needed

* No overfitting observed
* Model already stable

---

## 📊 Visualizations

* PCA Explained Variance Curve
* Model Comparison (Raw vs PCA)

---

## 🧪 Evaluation Metrics

Implemented manually:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib

---

## ⚠️ Important Notes

* No ML libraries used (from scratch implementation)
* Validation used for tuning
* Test used once for final evaluation
* Results based on stratified subsets

---

## 🧠 What Makes This Project Strong

✔ Full ML pipeline from scratch
✔ Deep understanding of algorithms
✔ Real experimental workflow
✔ Strong performance improvements
✔ Clean modular code
✔ Professional results & analysis

---

## 🚀 How to Run

```bash
# run final pipeline
python src/phase2/main.py

# generate results
python src/phase2/gen_results.py
```

---

## 👨‍💻 Author

**Karim Samer**

Computer Engineering Student
Interested in Machine Learning & Systems

---

## ⭐ Final Note

This project demonstrates not just model performance,
but the ability to **build, analyze, and improve machine learning systems from the ground up.**



