# Image Classification Project — Phase 1

## 1. Project Overview
This project aims to build a complete machine learning pipeline for binary image classification using the MNIST dataset.

The goal is to:
- Implement machine learning models from scratch
- Apply proper data preprocessing
- Evaluate and compare model performance
- Understand the full ML pipeline

---

## 2. Problem Definition

We aim to classify images of handwritten digits into two classes.

### Input:
- Image represented as a vector of 784 features (28×28 pixels)

### Output:
- Binary label (0 or 1)

---

## 3. Dataset

- Dataset: MNIST
- Format: CSV (flattened images)

Each row contains:
- First column → label
- Remaining 784 columns → pixel values

Dataset split:
- Training: ~60,000 samples
- Testing: ~10,000 samples

---

## 4. Selected Classes

We selected:
- Class 0
- Class 1

Reason:
- Clear visual separation
- Easier baseline
- Allows focus on implementation rather than data complexity

---

## 5. Data Processing Pipeline

Steps:

1. Load dataset from CSV
2. Separate features and labels
3. Filter selected classes (0 and 1)
4. Normalize pixel values (0 → 1)
5. Split data into:
   - Training set
   - Validation set
   - Test set

---

## 6. Feature Representation

### Baseline:
- Flattened pixel vector (784 features)

### Optional Improvements:
- PCA (dimensionality reduction)
- HOG (if time allows)

---

## 7. Models (From Scratch)

We will implement:

### 1. K-Nearest Neighbors (KNN)
- Distance-based classification
- No training phase

### 2. Logistic Regression
- Linear model
- Uses sigmoid function
- Optimized using gradient descent

### 3. Naive Bayes
- Probabilistic model
- Assumes feature independence

---

## 8. Alternative Models (Optional)

If needed or time allows:

- Linear SVM (alternative to Logistic Regression)
- Kernel SVM (only if necessary — more complex)

---

## 9. Loss Function & Optimization

### Logistic Regression:
- Loss: Binary Cross Entropy
- Optimization: Gradient Descent

Other models:
- KNN → no training loss
- Naive Bayes → probabilistic estimation

---

## 10. Evaluation Metrics

We will evaluate models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 11. Model Comparison

We will compare models based on:

- Performance metrics
- Simplicity
- Speed
- Generalization ability

---

## 12. Improvement Strategy

If time allows:

- Apply PCA and re-evaluate models
- Compare performance before and after dimensionality reduction
- Tune hyperparameters (e.g., k in KNN)

---

## 13. Project Structure

ML Project/
│
├── MNIST-data/
├── src/
│   ├── data_module.py
│   ├── features_module.py
│   ├── models_module.py
│   ├── evaluation_module.py
│   └── main.py
│
├── results/
│   ├── figures/
│   └── tables/
│
├── Project_Description.md
├── Tasks_Plan.md
└── Experiments_Log.md

---

## 14. Notes

- Models must be implemented from scratch
- Preprocessing utilities (split, scaling, PCA) can use libraries
- Focus on correctness first, then optimization
- Build a working pipeline before adding improvements

---

## 15. Development Strategy

1. Build a simple working pipeline
2. Implement models one by one
3. Test each model individually
4. Compare results
5. Apply improvements
6. Document everything