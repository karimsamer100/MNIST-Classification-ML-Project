<div align="center">

<pre>

███╗   ███╗███╗   ██╗██╗███████╗████████╗
████╗ ████║████╗  ██║██║██╔════╝╚══██╔══╝
██╔████╔██║██╔██╗ ██║██║███████╗   ██║   
██║╚██╔╝██║██║╚██╗██║██║╚════██║   ██║   
██║ ╚═╝ ██║██║ ╚████║██║███████║   ██║   
╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝╚══════╝   ╚═╝  

</pre>
### **From-Scratch ML Pipeline · CSE382 Spring 2026**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-only-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![MNIST](https://img.shields.io/badge/Dataset-MNIST-FF6B6B?style=flat-square)](http://yann.lecun.com/exdb/mnist/)
[![Phases](https://img.shields.io/badge/Phases-1%20%26%202-4ECDC4?style=flat-square)]()
[![No ML Libs](https://img.shields.io/badge/No%20sklearn-built%20from%20scratch-F7DC6F?style=flat-square)]()

*Handwritten digit classification · No ML libraries · Pure NumPy*

</div>

---

## `>_ what is this`

A complete machine learning pipeline built **entirely from scratch** to classify handwritten digits from MNIST. Every algorithm — from KNN to Softmax regression — is implemented manually using only NumPy.

```
Phase 1  →  Binary Classification   (0 vs 1)      Best F1: 0.9996
Phase 2  →  Multi-Class             (0 through 9)  Best F1: 0.8903
```

---

## `>_ pipeline`

```
  [28×28 Image]
       │
       ▼
  [Flatten → 784]  ──►  [Normalize ÷255 → range: 0.0–1.0]
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
      [Raw Features 784]              [PCA → 50 components]
                                      Covariance → Eigen Decomp
              │                                 │
              └────────────┬────────────────────┘
                           ▼
                     [Model Training]
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    KNN (Phase 1)    Log. Reg.        Naive Bayes
    k=3, Euclidean   Softmax+GD       Gaussian likelihood
                                           ▼
                                   Nearest Centroid
                                   (Phase 2 only)
                           │
                           ▼
                    [Evaluation]
          Accuracy · Precision · Recall · F1  (all macro, all manual)
```

---

## `>_ configuration`

```
Random State     : 42
Validation Split : 20%
Feature Range    : [0.0, 1.0]  (after normalization)
Original Dim     : 784
PCA Dim          : 50

Phase 1 splits   →  Train: 10,133 · Val: 2,532  · Test: 2,115
Phase 2 splits   →  Train: 48,004 · Val: 11,996 · Test: 10,000
```

---

## `>_ project structure`

```
ML Project/
├── MNIST-data/
│   ├── mnist_train.csv
│   └── mnist_test.csv
├── src/
│   ├── phase1/
│   │   ├── data_module.py        ← load, filter (0 vs 1), normalize, split
│   │   ├── features_module.py    ← PCA from scratch
│   │   ├── models_module.py      ← KNN · LogReg · GNB
│   │   ├── evaluation_module.py  ← accuracy, precision, recall, F1, CM
│   │   └── main.py
│   └── phase2/
│       ├── data_module.py        ← full 10-class pipeline
│       ├── features_module.py    ← same PCA, all 10 classes
│       ├── models_module.py      ← Softmax LogReg · GNB · NearestCentroid
│       ├── evaluation_module.py  ← macro-averaged multiclass metrics
│       └── main.py
└── docs/
    ├── Phase1_Report.docx
    └── Phase2_Report.docx
```

---

## `>_ phase 1 — binary (0 vs 1)`

**Dataset:** 10,133 train · 2,532 val · 2,115 test  
**PCA:** 50 components · **90.37% variance retained**

#### Validation Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| KNN *(k=3)* | `0.9992` | `0.9985` | `1.0000` | `0.9993` |
| Logistic Regression | `0.9972` | `0.9978` | `0.9970` | `0.9974` |
| Gaussian Naive Bayes | `0.9937` | `0.9941` | `0.9941` | `0.9941` |

#### Test — Raw Features

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| KNN *(k=3)* | `0.9991` | `0.9982` | `1.0000` | `0.9991` |
| **Logistic Regression** | **`0.9995`** | **`0.9991`** | **`1.0000`** | **`0.9996`** |
| Gaussian Naive Bayes | `0.9972` | `0.9974` | `0.9974` | `0.9974` |

#### Test — After PCA

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| KNN *(k=3)* | `0.9991` | `0.9982` | `1.0000` | `0.9991` |
| Logistic Regression | `0.9991` | `0.9982` | `1.0000` | `0.9991` |
| Gaussian Naive Bayes | `0.9844` | `1.0000` | `0.9709` | `0.9852` |

```
★  Best: Logistic Regression (Raw)  →  Accuracy: 0.9995 · F1: 0.9996
```

> Digits 0 and 1 are nearly linearly separable , all models exceed 99% F1. PCA barely affects KNN and LogReg. GNB drops slightly after PCA: perfect precision but recall falls, meaning it becomes overly conservative on digit 1.

---

## `>_ phase 2 — multi-class (0–9)`

**Dataset:** 48,004 train · 11,996 val · 10,000 test  
**PCA:** 50 components · **82.47% variance retained** *(lower than Phase 1 — 10-class data has wider variance structure)*

#### Validation Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | `0.8827` | `0.8816` | `0.8809` | `0.8809` |
| Gaussian Naive Bayes ⚠️ | `0.4777` | `0.6294` | `0.4695` | `0.3726` |
| Nearest Centroid | `0.8054` | `0.8093` | `0.8020` | `0.8032` |

#### Test — Raw Features

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** | **`0.8921`** | **`0.8910`** | **`0.8905`** | **`0.8903`** |
| Gaussian Naive Bayes ⚠️ | `0.4814` | `0.6355` | `0.4740` | `0.3770` |
| Nearest Centroid | `0.8200` | `0.8231` | `0.8170` | `0.8180` |

#### Test — After PCA

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | `0.8840` | `0.8832` | `0.8824` | `0.8821` |
| Gaussian Naive Bayes ✅ | `0.8785` | `0.8780` | `0.8778` | `0.8776` |
| Nearest Centroid | `0.8162` | `0.8194` | `0.8131` | `0.8141` |

```
★  Best: Logistic Regression (Raw)
   Accuracy: 0.8921 · Precision: 0.8910 · Recall: 0.8905 · F1: 0.8903
```

---

## `>_ pca breakdown`

```
                     Phase 1       Phase 2
  ─────────────────────────────────────────
  Original features    784           784
  Components kept       50            50
  Explained variance  90.37%        82.47%
  Dim reduction       93.6%         93.6%

  Method: Covariance Matrix → Eigen Decomposition → Top-K eigenvectors
  Fit on training only → applied to val/test (no leakage)

  Note: Phase 2 variance is lower because 10-class data spreads
        variance across far more principal directions.
```

---

## `>_ model notes`

```
Logistic Regression
  ├── Softmax for multiclass (Phase 2 generalization)
  ├── Cross-entropy loss
  ├── Batch Gradient Descent  (lr=0.1, iterations=300)
  ├── L2 regularization available (λ=0.0 in experiments)
  └── Most stable — best in both phases

Gaussian Naive Bayes
  ├── Gaussian likelihood per feature
  ├── Log-probabilities (avoids numerical underflow)
  ├── Assumes feature independence ← violated by raw pixels
  ├── Raw F1 Phase 2: 0.3770 → PCA F1: 0.8776  (+0.50)
  └── Reason: PCA decorrelates features, partially restores assumption

KNN  (Phase 1 only)
  ├── k=3, Euclidean distance
  └── Dropped in Phase 2 → O(n×d) prediction too slow on 60K samples

Nearest Centroid  (Phase 2 only)
  ├── One mean vector per class
  ├── Euclidean distance, O(k) prediction at inference
  └── Fast baseline — no hyperparameters, linear boundaries
```

---

## `>_ key findings`

```
1. Phase 1 is nearly perfectly separable
   → All models achieve >99% F1 on test set

2. Logistic Regression is the most stable model
   → Best in Phase 1 (F1: 0.9996) and Phase 2 (F1: 0.8903)
   → Consistent across raw and PCA features

3. Gaussian Naive Bayes is highly feature-sensitive
   → Raw pixels: F1 = 0.3770  (class collapse due to correlation)
   → After PCA:  F1 = 0.8776  (biggest PCA beneficiary)

4. PCA effect differs by phase
   → Phase 1: negligible (task already easy, 90.37% variance)
   → Phase 2: critical — rescues Naive Bayes (82.47% variance)

5. Nearest Centroid is a solid fast baseline
   → F1: 0.8180 with O(k) inference and zero hyperparameters
   → Used in Phase 2 as a replacement for slow KNN
```

---

## `>_ run`

```bash
# Phase 1 — Binary (0 vs 1)
python src/phase1/main.py

# Phase 2 — Multi-class (0–9)
python src/phase2/main.py
```

**Requirements:** `numpy` · `pandas` · Python 3.12

---

## `>_ guarantees`

```
✓  No sklearn or ML libraries — all algorithms built from scratch
✓  No data leakage — PCA fitted on training data only
✓  Test set used exactly once — final evaluation only
✓  Stratified splits — class balance preserved (random state: 42)
✓  Fully reproducible — deterministic pipeline end to end
```

---

<div align="center">

**Karim Samer** · Computer Engineering · CAIE Program · Spring 2026

*CSE382: Introduction to Machine Learning · Major Task Project*

</div>