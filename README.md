# MNIST Image Classification Project

This repository contains a complete machine learning pipeline for handwritten digit classification using the MNIST dataset. The project is organized into two development phases:

- `Phase 1`: binary classification for digits `0` vs `1`
- `Phase 2`: multi-class classification for digits `0-9`

The implementation focuses on understanding core machine learning concepts by building the main models and feature pipeline from scratch with NumPy, while using standard Python tools for data loading, plotting, and CSV-based experiment outputs.

## Project Highlights

- End-to-end MNIST classification workflow
- Clean separation of data, features, models, evaluation, and experiment scripts
- From-scratch implementations of:
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Gaussian Naive Bayes
  - Principal Component Analysis (PCA)
- Binary and multi-class evaluation pipelines
- Saved figures, tables, and summary logs for reporting

## Repository Structure

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

## Problem Statement

The project uses the MNIST handwritten digit dataset, where each image is represented as a flattened vector of `784` pixel values (`28 x 28` grayscale image).

- Input: flattened pixel vector
- Output in Phase 1: binary class label for digits `0` and `1`
- Output in Phase 2: multi-class label for digits `0-9`

## Methodology

### Phase 1

Phase 1 builds a binary classification baseline:

- Loads MNIST CSV files
- Filters the dataset to digits `0` and `1`
- Normalizes pixel values to `[0, 1]`
- Applies a stratified train/validation split
- Trains and evaluates:
  - KNN
  - Logistic Regression
  - Gaussian Naive Bayes
- Compares raw pixel features against PCA-transformed features

### Phase 2

Phase 2 extends the pipeline to full multi-class classification:

- Uses all digit classes from `0` to `9`
- Reuses the preprocessing pipeline with stratified splitting
- Adapts the models for multi-class prediction
- Evaluates models using multi-class accuracy, precision, recall, and macro F1
- Compares raw features with PCA-based dimensionality reduction
- Selects a final model based on validation performance

## Models Implemented

### K-Nearest Neighbors

- Distance-based classifier using Euclidean distance
- Phase 1: binary prediction
- Phase 2: majority voting across multi-class neighbors

### Logistic Regression

- Phase 1: binary logistic regression with sigmoid activation
- Phase 2: multinomial logistic regression with softmax
- Gradient descent optimization
- Phase 2 includes optional L2 regularization support through `lambda_reg`

### Gaussian Naive Bayes

- Estimates class-wise feature mean, variance, and prior probabilities
- Uses Gaussian likelihoods for prediction
- Extended from binary to multi-class classification in Phase 2

### Principal Component Analysis

- Implemented from scratch using covariance matrices and eigen-decomposition
- Used to reduce dimensionality before model training
- Phase 2 uses `50` principal components in the main experiments

## Evaluation Metrics

### Phase 1

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Phase 2

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Multi-class confusion matrix

## Final Phase 2 Result

The final selected model in Phase 2 is:

- `KNN (k=1)` with `PCA (50 components)`

Saved final test result from `results/phase2/tables/final_test_result.csv`:

| Model | Feature Setting | Accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| KNN (k=1) | PCA (50 components) | 0.9533 | 0.9539 | 0.9533 | 0.9533 |

Note: these metrics were computed on a stratified test subset of `300` samples, as documented in `results/phase2/logs/phase2_summary.txt`.

## Requirements

This project uses Python 3 and the following libraries:

- `numpy`
- `pandas`
- `matplotlib`

Install them with:

```bash
pip install numpy pandas matplotlib
```

## Dataset Setup

Place the MNIST CSV files inside the `MNIST-data/` directory:

```text
MNIST-data/
├── mnist_train.csv
└── mnist_test.csv
```

The code expects CSV files where:

- column `0` is the label
- columns `1-784` are pixel values

## How To Run

Run commands from the repository root.

### Phase 1 pipeline

```bash
python src/phase1/main.py
```

### Phase 2 pipeline

```bash
python src/phase2/main.py
```

### Generate Phase 2 result files

```bash
python src/phase2/gen_results.py
```

This script generates:

- `results/phase2/tables/validation_comparison.csv`
- `results/phase2/tables/final_test_result.csv`
- `results/phase2/figures/validation_f1_comparison.png`
- `results/phase2/figures/pca_cumulative_variance.png`
- `results/phase2/logs/phase2_summary.txt`

## Example Output Artifacts

The repository already includes generated outputs such as:

- `results/phase1/figures/model_comparison.png`
- `results/phase1/figures/pca_variance.png`
- `results/phase1/tables/final_results.xlsx`
- `results/phase2/figures/validation_f1_comparison.png`
- `results/phase2/figures/pca_cumulative_variance.png`
- `results/phase2/tables/validation_comparison.csv`
- `results/phase2/tables/final_test_result.csv`
- `results/phase2/logs/phase2_summary.txt`

## Design Notes

- The project is intentionally modular to make experimentation easier.
- The main training and evaluation flow lives in `main.py` for each phase.
- `gen_results.py` is used to export tables, plots, and summary files for reports.
- `test_pipeline.py` can be used as a sandbox for quick experiments and debugging.

## Limitations

- The current experiments use stratified subsets for faster iteration instead of the full MNIST dataset in every run.
- There is no `requirements.txt` file yet, so dependencies are listed manually in this README.
- The code is educational and optimized for clarity and learning more than runtime performance.

## Future Improvements

- Add a `requirements.txt` file for easier environment setup
- Add unit tests for model and evaluation modules
- Benchmark on larger subsets or the full dataset
- Add hyperparameter search utilities
- Extend feature engineering beyond PCA

## Documentation

Additional project documentation is available in:

- `docs/phase1/phase1_Project_Description.md`
- `docs/phase1/phase1_Tasks_Plan.md`
- `docs/phase2/Phase2_Description.md`
- `docs/phase2/Phase2_Workflow_Guide.md`
- `docs/CSE382_MajorTask_Spring2026.pdf`

## Project Context

This project is structured like an academic machine learning assignment focused on implementing core classification algorithms, comparing feature representations, and documenting results in a clear experimental workflow.
