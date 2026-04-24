# MNIST Classification --- From Scratch ML Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-Used-success)
![Status](https://img.shields.io/badge/Project-Complete-brightgreen)

CSE382 --- Introduction to Machine Learning (Spring 2026)

------------------------------------------------------------------------

## Overview

End-to-end ML pipeline built **from scratch** for MNIST digit
classification.

-   No sklearn models
-   Manual implementation of:
    -   Models
    -   PCA
    -   Metrics
    -   Data pipeline

------------------------------------------------------------------------

## Pipeline

Images → Flatten (784) → Normalize → Split\
→ Models (Raw)\
→ PCA (50)\
→ Models (PCA)\
→ Evaluation (Accuracy + Macro F1)

------------------------------------------------------------------------

## Phase 1 --- Binary (0 vs 1)

### Models

-   KNN (k=3)
-   Logistic Regression
-   Gaussian Naive Bayes

### Results (Test)

  Model                 Raw F1       PCA F1
  --------------------- ------------ --------
  KNN                   0.9991       0.9991
  Logistic Regression   **0.9996**   0.9991
  Gaussian NB           0.9974       0.9852

------------------------------------------------------------------------

## Phase 2 --- Multi-Class (0--9)

### Models

-   Logistic Regression (Softmax)
-   Gaussian Naive Bayes
-   Nearest Centroid

### Improvements

-   PCA (784 → 50)
-   L2 Regularization
-   Overfitting Analysis

------------------------------------------------------------------------

### Results (Test)

  Model                 Raw F1       PCA F1
  --------------------- ------------ ------------
  Logistic Regression   **0.8903**   0.8821
  Gaussian NB           0.3770       **0.8776**
  Nearest Centroid      0.8180       0.8141

------------------------------------------------------------------------

## Key Insights

-   Logistic Regression → best overall
-   Gaussian NB → huge gain after PCA
-   Nearest Centroid → fast baseline
-   No significant overfitting

------------------------------------------------------------------------

## Final Model

Logistic Regression (Raw Features)

Accuracy : 0.8921\
F1 Score : 0.8903

------------------------------------------------------------------------

## Evaluation

-   Accuracy
-   Precision (macro)
-   Recall (macro)
-   F1-score (macro)
-   Confusion Matrix

------------------------------------------------------------------------

## Run

python src/phase1/main.py\
python src/phase2/main.py

------------------------------------------------------------------------

## Notes

-   No data leakage (PCA fit on train only)
-   Test set used once
-   Fully modular code

------------------------------------------------------------------------

## Author

Karim Samer\
Computer Engineering --- CAIE
