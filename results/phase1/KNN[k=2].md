
================================================================================
Phase 1: Binary MNIST Classification
================================================================================
Task: classify digit 0 vs digit 1
Encoded labels: 0 means original digit 0, 1 means original digit 1
Random state: 42

================================================================================
Prepared Full Dataset
================================================================================
Train        shape=(10133, 784)   class_counts=[4739, 5394]
Validation   shape=(2532, 784)    class_counts=[1184, 1348]
Test         shape=(2115, 784)    class_counts=[980, 1135]
Pixel range after normalization: [0.0, 1.0]

================================================================================
PCA Summary
================================================================================
Components kept: 50
Raw feature count: 784
PCA feature count: 50
Total explained variance ratio: 0.9037
First 10 explained variance ratios:
[0.322  0.0904 0.0812 0.0549 0.039  0.034  0.0237 0.021  0.0178 0.0156]

================================================================================
Validation Scores Before PCA
================================================================================
Model                       Accuracy  Precision     Recall         F1
------------------------------------------------------------------------
KNN (k=2)                     0.9988     0.9985     0.9993     0.9989
Logistic Regression           0.9972     0.9978     0.9970     0.9974
Gaussian Naive Bayes          0.9937     0.9941     0.9941     0.9941

================================================================================
Test Scores Before PCA
================================================================================
Model                       Accuracy  Precision     Recall         F1
------------------------------------------------------------------------
KNN (k=2)                     0.9991     0.9982     1.0000     0.9991
Logistic Regression           0.9995     0.9991     1.0000     0.9996
Gaussian Naive Bayes          0.9972     0.9974     0.9974     0.9974

================================================================================
Validation Scores After PCA
================================================================================
Model                       Accuracy  Precision     Recall         F1
------------------------------------------------------------------------
KNN (k=2)                     0.9992     0.9985     1.0000     0.9993
Logistic Regression           0.9976     0.9985     0.9970     0.9978
Gaussian Naive Bayes          0.9787     1.0000     0.9599     0.9796

================================================================================
Test Scores After PCA
================================================================================
Model                       Accuracy  Precision     Recall         F1
------------------------------------------------------------------------
KNN (k=2)                     0.9991     0.9982     1.0000     0.9991
Logistic Regression           0.9991     0.9982     1.0000     0.9991
Gaussian Naive Bayes          0.9844     1.0000     0.9709     0.9852

================================================================================
Accuracy Comparison Before and After PCA
================================================================================
Model                       Raw Val  Raw Test   PCA Val  PCA Test
--------------------------------------------------------------------
KNN (k=2)                    0.9988    0.9991    0.9992    0.9991
Logistic Regression          0.9972    0.9995    0.9976    0.9991
Gaussian Naive Bayes         0.9937    0.9972    0.9787    0.9844

================================================================================
Validation Confusion Matrices Before PCA
================================================================================
Rows are actual labels, columns are predicted labels. 0=0, 1=1

KNN (k=2)
              Pred 0   Pred 1
Actual 0        1182        2
Actual 1           1     1347

Logistic Regression
              Pred 0   Pred 1
Actual 0        1181        3
Actual 1           4     1344

Gaussian Naive Bayes
              Pred 0   Pred 1
Actual 0        1176        8
Actual 1           8     1340

================================================================================
Test Confusion Matrices Before PCA
================================================================================
Rows are actual labels, columns are predicted labels. 0=0, 1=1

KNN (k=2)
              Pred 0   Pred 1
Actual 0         978        2
Actual 1           0     1135

Logistic Regression
              Pred 0   Pred 1
Actual 0         979        1
Actual 1           0     1135

Gaussian Naive Bayes
              Pred 0   Pred 1
Actual 0         977        3
Actual 1           3     1132

================================================================================
Validation Confusion Matrices After PCA
================================================================================
Rows are actual labels, columns are predicted labels. 0=0, 1=1

KNN (k=2)
              Pred 0   Pred 1
Actual 0        1182        2
Actual 1           0     1348

Logistic Regression
              Pred 0   Pred 1
Actual 0        1182        2
Actual 1           4     1344

Gaussian Naive Bayes
              Pred 0   Pred 1
Actual 0        1184        0
Actual 1          54     1294

================================================================================
Test Confusion Matrices After PCA
================================================================================
Rows are actual labels, columns are predicted labels. 0=0, 1=1

KNN (k=2)
              Pred 0   Pred 1
Actual 0         978        2
Actual 1           0     1135

Logistic Regression
              Pred 0   Pred 1
Actual 0         978        2
Actual 1           0     1135

Gaussian Naive Bayes
              Pred 0   Pred 1
Actual 0         980        0
Actual 1          33     1102
PS C:\Users\Karim\Desktop\Machine Learning\ML Project> 