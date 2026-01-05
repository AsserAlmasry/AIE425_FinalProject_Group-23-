# AIE425 – Intelligent Recommender Systems

## Section 1: Dimensionality Reduction & Collaborative Filtering

This repository contains the **complete and reproducible implementation of Section 1** of the AIE425 course project. The work follows the official project specification and implements **all required tasks for Section 1**, with clean, portable code and no placeholder or toy data.

---
## Dataset link : https://grouplens.org/datasets/movielens/20m/
  
##  Section 1 Overview

Section 1 investigates **dimensionality reduction techniques for recommender systems**, including:

1. **PCA with Mean-Filling**
2. **PCA with Maximum Likelihood Estimation (MLE)**
3. **Singular Value Decomposition (SVD) and Truncated SVD**

All experiments are conducted on the real ratings dataset and include full evaluation, comparisons, and discussion.

---

## Project Structure

```
project_root/
│
├── data/
│   └── ratings.csv                  # Ratings dataset
│
├── results/
│   ├── target_items.csv
│   ├── predictions_part1_top5.csv
│   ├── predictions_part1_top10.csv
│   ├── predictions_mle_top5.csv
│   ├── predictions_mle_top10.csv
│   ├── compare_part1_part2.csv
│   └── svd_results/
│
├── Statistical_analysis.ipynb  # Section 1 – Part 3.1
├── part1_.ipynb                 # Section 1 – Part 3.2 (PCA Mean-Filling)
├── Part2.ipynb                 # Section 1 – Part 3.3 (PCA with MLE)
├── part3.ipynb                 # Section 1 – Part 3.4 (SVD)
│
├── README.md
├── requirements.txt
```

---

## Task Completion Checklist (Section 1)

###  Part 3.1 – Dataset Analysis & Target Selection

- Dataset validation (size, rating scale)
- Item popularity computation
- Selection of:
  - Low-popularity item **I1**
  - High-popularity item **I2**
  - Cold / medium / rich users
- Outputs saved for reuse in later parts

**Notebook:** `Statistical_analysis.ipynb`

---

### Part 3.2 – PCA with Mean-Filling

- Construction of real user–item matrix (I1, I2)
- Mean-filling of missing ratings
- Mean-centering and covariance computation
- Top-5 and top-10 peer identification
- Reduced-space prediction of missing ratings
- Comparison of top-5 vs top-10 results

**Notebook:** `part1.ipynb`

---

###  Part 3.3 – PCA with MLE Covariance

- Construction of real user–item matrix (no imputation)
- MLE covariance using **only common users**
- Top-5 and top-10 peer identification
- Prediction with covariance-weighted reduced space
- Mean fallback when denominator is zero
- Comparison:
  - Top-5 vs Top-10 (Part 2)
  - PCA Mean-Filling vs PCA MLE

**Notebook:** `Part2_FIXED.ipynb`

---

### Part 3.4 – SVD & Truncated SVD

- Mean-filled rating matrix construction
- Full SVD with orthogonality checks
- Singular value and variance plots
- Truncated SVD (k = 5, 20, 50, 100)
- MAE / RMSE evaluation
- Optimal k selection
- Target user/item prediction
- PCA vs SVD comparison
- Sensitivity analysis (missingness & initialization)
- Cold-start simulation and mitigation

**Notebook:** `part3.ipynb`

---

##  Execution Order

Run the notebooks **in the following order**:

1. `Statistical_analysis.ipynb`
2. `part1.ipynb`
3. `Part2.ipynb`
4. `part3.ipynb`

All notebooks use **relative paths** and can be executed using **Run All** from a clean environment.

---

##
