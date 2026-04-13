# Image Classification Project — Phase 2 Workflow Guide

## 1. Purpose of This File

This file explains **how Phase 2 will be developed and executed**.

It is not the theoretical project description.  
Instead, it is the practical execution guide that explains:

- how we will continue from Milestone 1
- how we will build Phase 2 step by step
- how each module will be handled
- how improvements will be applied
- how experiments will be organized
- how we will avoid breaking the working Phase 1 baseline

---

## 2. Main Working Principle

Phase 2 will be developed as a **continuation of Milestone 1**, not as a completely new project.

This means:

- Milestone 1 remains the stable baseline
- Phase 2 reuses as much logic as possible
- any reused code should be adapted carefully for multi-class classification
- new improvements should be added only after the baseline multi-class version works correctly

---

## 3. Development Style

The project will continue using the same workflow that was used successfully in Milestone 1.

### Working style
For every module or major change:

1. implement one small block
2. test it immediately
3. confirm correctness
4. move to the next block

### Why this style is used
- helps understand each part deeply
- catches bugs early
- prevents hidden issues from spreading
- keeps the pipeline stable during development

### Coding style rules
- code should be clean and readable
- comments should be simple and natural
- code should not feel overly AI-generated
- avoid unnecessary complexity
- correctness comes before optimization

---

## 4. Phase 2 Starting Point

Phase 2 starts from a validated Milestone 1 codebase that already includes:

- data loading and preprocessing
- train / validation / test methodology
- KNN from scratch
- Logistic Regression from scratch
- Gaussian Naive Bayes from scratch
- evaluation functions
- PCA from scratch
- clean pipeline organization

So the goal now is **not** to rebuild everything, but to extend and adapt it.

---

## 5. General Execution Plan

Phase 2 will be developed in the following order:

### Stage 1 — Prepare Phase 2 Files
- create or organize `src/phase2`
- create or organize `results/phase2`
- create Phase 2 documentation files
- confirm work is being done on the correct Git branch

### Stage 2 — Adapt Data Pipeline
- reuse MNIST CSV loading
- remove binary-only filtering logic
- keep all labels from 0 to 9
- normalize the data
- prepare train / validation / test sets
- verify class distribution

### Stage 3 — Build Baseline Multi-class Version
- adapt KNN for multi-class prediction
- adapt Gaussian Naive Bayes for multi-class
- adapt Logistic Regression for multi-class
- run a first baseline experiment without improvements

### Stage 4 — Adapt Evaluation
- compute multi-class accuracy
- compute multi-class confusion matrix
- compute precision / recall / F1 in a suitable multi-class way
- compare baseline model results

### Stage 5 — Apply Improvements
- add one improvement at a time
- test after each improvement
- compare before vs after
- keep experiment notes after every important run

### Stage 6 — Finalize Outputs
- save final tables
- save useful figures
- summarize observations
- prepare notes for final report and presentation

---

## 6. How Each Module Will Be Handled

### `data_module.py`
This module will be adapted first.

#### Main goal
Make the data pipeline support all 10 classes instead of only 2.

#### What will likely stay the same
- CSV loading
- feature / label separation
- normalization logic
- train / validation / test methodology

#### What will change
- binary filtering logic must be removed or replaced
- labels should remain as multi-class labels
- class distribution should be checked across all digits

#### Success condition
The module should return correct multi-class datasets ready for training and evaluation.

---

### `models_module.py`
This module will be adapted after the data pipeline works.

#### KNN
- easiest model to adapt
- same nearest-neighbor logic
- prediction becomes majority vote among 10 possible classes

#### Gaussian Naive Bayes
- extend class statistics from 2 classes to 10 classes
- compute mean, variance, and prior for each class

#### Logistic Regression
- binary version is not enough anymore
- must be adapted for multi-class classification
- the simplest correct implementation path should be chosen first

#### Success condition
Each model should train and predict correctly on all classes before any improvement is added.

---

### `evaluation_module.py`
This module should be adapted after at least one multi-class model works.

#### What should be supported
- multi-class accuracy
- multi-class confusion matrix
- precision
- recall
- F1-score

#### Important note
The confusion matrix should clearly show class labels, just like the binary version did in Milestone 1.

#### Success condition
The module should produce clear and interpretable evaluation outputs for 10-class classification.

---

### `features_module.py`
This module will be used after the multi-class baseline works.

#### Baseline
- flattened pixels only

#### Possible feature improvements
- PCA reuse from Milestone 1
- HOG if feasible
- advanced feature extraction only if time and stability allow

#### Important rule
Feature improvements come **after** the baseline pipeline works.

---

### `main.py`
This file will become the clean final runner for Phase 2.

#### It should eventually do the following
- load data
- preprocess data
- run baseline multi-class models
- apply improvements
- evaluate results
- print clean summaries

#### Important note
During development, experiments can be done in `test_pipeline.py`, but `main.py` should remain the clean final pipeline.

---

### `test_pipeline.py`
This file will be the development sandbox.

#### It will be used for
- quick testing
- debugging
- sanity checks
- temporary experiments
- checking one model at a time
- trying improvements before putting them in the clean pipeline

#### Important note
This file can be larger and noisier than `main.py`, and that is acceptable.

---

### `gen_results.py`
This file will be used after the main experiments are stable.

#### Purpose
- export final tables
- generate useful figures
- organize final outputs for the report

---

## 7. Improvement Strategy

Phase 2 officially requires at least **three improvement techniques**. :contentReference[oaicite:1]{index=1}

The practical strategy is:

### First
Build a correct multi-class baseline.

### Then
Apply improvements one by one.

### Recommended improvement order
1. hyperparameter tuning
2. PCA reuse / dimensionality reduction
3. regularization or learning analysis

### Optional later improvements
- HOG
- advanced CNN feature extraction
- ensemble methods

### Important rule
Do not stack many improvements at once before testing each one separately.

---

## 8. Experiment Methodology

All experiments should follow the same general rule:

### For each experiment
- define what changed
- run the experiment
- record metrics
- note observations
- compare against the previous version

### Example experiment types
- baseline KNN on all 10 classes
- baseline Naive Bayes on all 10 classes
- baseline Logistic Regression on all 10 classes
- KNN after tuning `k`
- Naive Bayes after PCA
- Logistic Regression after regularization

### Important rule
Validation should be used for experimentation and tuning.  
Test should remain reserved for final evaluation only.

---

## 9. Practical Order of Work

The recommended actual working order is:

1. prepare Phase 2 folders and files
2. adapt `data_module.py`
3. test data shapes and class distributions
4. adapt KNN first
5. test KNN
6. adapt Gaussian Naive Bayes
7. test Gaussian Naive Bayes
8. adapt Logistic Regression
9. test Logistic Regression
10. adapt evaluation functions
11. run baseline comparison
12. apply one improvement at a time
13. save final results
14. prepare report notes

---

## 10. Git and Workflow Rules

### Branch strategy
- keep Milestone 1 as a stable branch
- do Phase 2 work in a separate branch such as `phase2-multiclass`

### Commit strategy
Make a commit after each meaningful completed block, for example:
- after adapting data pipeline
- after finishing one model
- after finishing evaluation
- after finishing one major improvement

### Push strategy
Push regularly after stable commits to keep GitHub updated and history clean.

---

## 11. Rules That Must Be Preserved

The following rules from Milestone 1 must continue in Phase 2:

- build block by block
- test immediately after each block
- do not mix validation with final test evaluation
- keep comments simple and natural
- keep code readable
- do not break the working Phase 1 baseline
- use Milestone 1 as the reference point

---

## 12. Final Execution Philosophy

The correct philosophy for Phase 2 is:

- reuse what already works
- adapt carefully
- verify each step
- build the baseline first
- improve gradually
- compare results clearly
- keep the project organized at all times

This will make the project easier to debug, easier to explain, and stronger in the final report.

---

## 13. Short Action Summary

Phase 2 should be built in this order:

1. adapt the data pipeline to all 10 classes
2. adapt the three models to multi-class classification
3. adapt evaluation to multi-class metrics
4. run the first clean baseline
5. apply improvements one by one
6. save final results and observations