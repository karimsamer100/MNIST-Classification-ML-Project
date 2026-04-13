# Image Classification Project — Phase 2

## 1. Project Overview

Phase 2 focuses on extending the binary classification pipeline into a full **multi-class classification system (10 classes)** using the MNIST dataset.

The goal is to:
- Scale the model from 2 classes → 10 classes
- Improve model performance
- Apply advanced machine learning techniques
- Analyze and compare improvements

---

## 2. Objective

Build an improved ML pipeline that:
- Classifies all digits (0 → 9)
- Enhances baseline performance from Phase 1
- Applies at least **3 improvement techniques**

:contentReference[oaicite:0]{index=0}

---

## 3. Problem Definition

### Input:
- Image represented as a vector of 784 features (28×28 pixels)

### Output:
- Multi-class label ∈ {0, 1, 2, ..., 9}

---

## 4. Dataset

- Dataset: MNIST
- Format: CSV (flattened images)

Each sample:
- First column → label
- Remaining columns → pixel values

---

## 5. Data Processing Pipeline

Steps:

1. Load dataset
2. Separate features and labels
3. Use **all classes (0–9)** instead of filtering
4. Normalize pixel values (0 → 1)
5. Train / Validation / Test split

---

## 6. Feature Representation

### Baseline:
- Flattened pixel vector (784 features)

### Improvements:
- PCA (dimensionality reduction)
- HOG (optional)
- CNN feature extraction (advanced)

---

## 7. Models

We extend Phase 1 models to multi-class:

### 1. KNN (Multi-class)
- Same concept
- Predict majority class among neighbors

### 2. Logistic Regression (Multinomial)
- Softmax instead of sigmoid
- Multi-class classification

### 3. Naive Bayes
- Multi-class probabilistic model

---

## 8. Improvement Techniques

At least **3 improvements must be applied**:

### Option 1: Feature Engineering
- PCA
- HOG
- CNN feature extraction (optional advanced)

### Option 2: Hyperparameter Tuning
- Choosing best k (KNN)
- Learning rate tuning (Logistic)
- Cross-validation

### Option 3: Regularization
- L1 / L2 regularization
- Prevent overfitting

### Option 4: Ensemble Methods
- Random Forest
- Boosting (optional advanced)

### Option 5: Learning Analysis
- Learning curves
- Overfitting vs Underfitting analysis

---

## 9. Loss Function & Optimization

### Logistic Regression:
- Loss: Categorical Cross Entropy
- Optimization: Gradient Descent

Other models:
- KNN → no training loss
- Naive Bayes → probability estimation

---

## 10. Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix (10×10)

---

## 11. Model Comparison

Compare based on:

- Performance metrics
- Generalization ability
- Speed
- Robustness

---

## 12. Workflow Strategy

### Step 1 — Adapt Pipeline
- Remove binary filter
- Support multi-class labels

### Step 2 — Adapt Models
- Modify Logistic → Softmax
- Ensure all models support multi-class

### Step 3 — Baseline Multi-class
- Train models without improvements

### Step 4 — Apply Improvements
- PCA
- Hyperparameter tuning
- Regularization

### Step 5 — Evaluate & Compare
- Before vs After improvements

---

## 13. Project Structure (Phase 2)

src/phase2/
- data_module.py
- features_module.py
- models_module.py
- evaluation_module.py
- main.py

results/phase2/
- figures/
- tables/
- logs/

docs/phase2/
- Phase2_Description.md
- Phase2_Tasks_Plan.md
- Experiments_Log.md

---

## 14. Development Strategy

1. Start from Phase 1 code
2. Modify for multi-class
3. Build working pipeline
4. Add improvements gradually
5. Evaluate after each improvement
6. Keep experiment logs

---

## 15. Key Notes

- Do NOT rewrite everything from scratch
- Reuse Phase 1 logic
- Focus on correctness first
- Improvements come after baseline works

---

## 16. Expected Outcome

- Full multi-class classifier
- Improved performance over baseline
- Clear comparison between techniques
- Strong analysis for final report