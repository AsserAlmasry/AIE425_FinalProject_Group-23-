# AIE425 – Intelligent Recommender Systems
## Course Project Implementation

This repository contains the **complete implementation** of both sections of the AIE425 course project, implementing dimensionality reduction techniques and multi-modal recommendation systems with comprehensive evaluation.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset Information](#dataset-information)
- [Section 1: Dimensionality Reduction & Collaborative Filtering](#section-1-dimensionality-reduction--collaborative-filtering)
- [Section 2: Multi-Modal Product Recommendation](#section-2-multi-modal-product-recommendation)
- [Installation](#installation)
- [Execution Guide](#execution-guide)
- [Results Summary](#results-summary)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Section 1: Dimensionality Reduction & Collaborative Filtering
Investigates classical dimensionality reduction techniques on **MovieLens 20M dataset**:
- PCA with Mean-Filling
- PCA with Maximum Likelihood Estimation (MLE)
- SVD and Truncated SVD analysis

### Section 2: Multi-Modal Product Recommendation Engine
Implements hybrid recommendation system for **Amazon Fashion dataset**:
- Multi-modal feature fusion (Text, Image, Category, Price)
- Content-based, Collaborative Filtering, and Hybrid approaches
- Achieves **18.7% improvement** over single-modal baselines

---

## Project Structure
```
AIE425_CourseProject/
│
├── README.md                        # This file
├── requirements.txt                 # All dependencies
│
├── SECTION_1/                       # Dimensionality Reduction & CF
│   ├── data/
│   │   └── ratings.csv             # MovieLens 20M ratings
│   │
│   ├── results/
│   │   ├── target_items.csv
│   │   ├── predictions_part1_top5.csv
│   │   ├── predictions_part1_top10.csv
│   │   ├── predictions_mle_top5.csv
│   │   ├── predictions_mle_top10.csv
│   │   ├── compare_part1_part2.csv
│   │   └── svd_results/
│   │       ├── singular_values.png
│   │       ├── variance_explained.png
│   │       ├── reconstruction_errors.csv
│   │       └── cold_start_analysis.csv
│   │
│   ├── statistical_analysis.py      # Part 3.1 - Dataset Analysis
│   ├── pca_mean_filling.py          # Part 3.2 - PCA Mean-Filling
│   ├── pca_mle.py                   # Part 3.3 - PCA with MLE
│   ├── svd_analysis.py              # Part 3.4 - SVD Analysis
│   └── README_SECTION1.md
│
├── SECTION_2/                       # Multi-Modal Recommendation
│   ├── data/
│   │   ├── raw/
│   │   │   ├── reviews.csv
│   │   │   └── products.csv
│   │   ├── processed/
│   │   │   ├── train.csv
│   │   │   ├── test.csv
│   │   │   ├── products.csv
│   │   │   └── mappings.pkl
│   │   ├── features/
│   │   │   ├── text_features.npy
│   │   │   ├── image_features.npy
│   │   │   ├── fused_features.npy
│   │   │   └── feature_objects.pkl
│   │   └── images/
│   │
│   ├── results/
│   │   ├── recommendations/
│   │   │   ├── sample_recommendations.json
│   │   │   └── summary.json
│   │   ├── plots/
│   │   └── tables/
│   │
│   ├── download_dataset.py
│   ├── data_preprocessing.py
│   ├── extract_features.py
│   ├── content_based.py
│   ├── collaborative.py
│   ├── hybrid.py
│   ├── main.py
│   ├── evaluation.py
│   ├── numerical_example.py
│   └── README_SECTION2.md
│
└── docs/
    ├── project_specifications.pdf
    └── presentation.pdf
```

---

## Dataset Information

### Section 1: MovieLens 20M
- **Source**: https://grouplens.org/datasets/movielens/20m/
- **Size**: 20 million ratings, ~138K users, ~27K movies
- **Rating Scale**: 0.5 to 5.0 stars

### Section 2: Amazon Fashion
- **Size**: ~62,450 interactions, ~5,234 users, ~587 products
- **Sparsity**: 97.8%
- **Modalities**: Text, Images, Categories, Prices

---

## Section 1: Dimensionality Reduction & Collaborative Filtering

### Implementation Overview

#### Part 3.1 – Dataset Analysis (`statistical_analysis.py`)
- Dataset validation and statistical analysis
- Item popularity computation
- Target selection: Low-popularity (I1), High-popularity (I2) items
- User categorization: Cold (<10 ratings), Medium (10-50), Rich (>50)

#### Part 3.2 – PCA Mean-Filling (`pca_mean_filling.py`)
- User-item matrix construction
- Mean-filling for missing ratings
- PCA with top-5 and top-10 peer identification
- Rating prediction and comparison

#### Part 3.3 – PCA with MLE (`pca_mle.py`)
- MLE covariance using only common users
- Sparse data handling without imputation
- Top-k peer identification and prediction
- Comparison: Mean-Filling vs MLE

#### Part 3.4 – SVD Analysis (`svd_analysis.py`)
- Full SVD decomposition (R = UΣV^T)
- Truncated SVD (k = 5, 20, 50, 100)
- Optimal k selection via MAE/RMSE
- Sensitivity and cold-start analysis

### Execution (Section 1)
```bash
cd SECTION_1

# Download MovieLens 20M dataset
# Place ratings.csv in data/ folder

# Run pipeline sequentially
python statistical_analysis.py
python pca_mean_filling.py
python pca_mle.py
python svd_analysis.py
```

**Runtime**: ~50-75 minutes total

---

## Section 2: Multi-Modal Product Recommendation

### Features

**Multi-Modal Feature Extraction**:
- **Text (TF-IDF)**: 500-dimensional vectors
- **Image (ResNet50)**: 2048-dimensional embeddings
- **Category**: One-hot encoding (23 categories)
- **Price**: Standardized features

**Feature Fusion**: `[0.3×Text | 0.4×Image | 0.2×Category | 0.1×Price]`

### Algorithms

1. **Content-Based**: Cosine similarity with user profiles
2. **Collaborative Filtering**: Item-based CF, User-based CF, SVD
3. **Hybrid Strategies**: Weighted, Switching, Cascade, Feature-Weighted, Mixed

### Execution (Section 2)
```bash
cd SECTION_2

# Complete pipeline
python download_dataset.py      # ~2 min
python data_preprocessing.py    # ~3 min
python extract_features.py      # ~10-15 min
python main.py                  # ~5 min
python evaluation.py            # ~8 min
```

**Total Runtime**: ~30-35 minutes

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- 4GB RAM (8GB recommended)
- 2GB disk space

### Setup
```bash
# Clone repository
git clone <repo-url>
cd AIE425_CourseProject

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For Section 2: Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Dependencies
**Core**: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn

**Section 2 Additional**: torch, torchvision, Pillow, scikit-surprise, tqdm

---

## Execution Guide

### Quick Start

**Section 1**:
```bash
cd SECTION_1
python statistical_analysis.py
python pca_mean_filling.py
python pca_mle.py
python svd_analysis.py
```

**Section 2**:
```bash
cd SECTION_2
python download_dataset.py && \
python data_preprocessing.py && \
python extract_features.py && \
python main.py && \
python evaluation.py
```

### Advanced Options

**Section 1 with custom parameters**:
```bash
# Custom k values for SVD
python svd_analysis.py --k-values 5 10 20 50 100 --cv-folds 5

# PCA with different peer counts
python pca_mean_filling.py --top-k 5 10 15 20
```

**Section 2 custom configuration**:
```bash
# Use CPU for feature extraction
python extract_features.py --device cpu --batch-size 8

# Evaluate specific hybrid method
python evaluation.py --method weighted_hybrid
```

---

## Results Summary

### Section 1 Performance

| Method | MAE (I1) | MAE (I2) | Optimal k | Coverage |
|--------|----------|----------|-----------|----------|
| PCA Mean-Fill | 0.87 | 0.64 | - | 100% |
| PCA MLE | 0.72 | 0.61 | - | 85% |
| SVD | 0.64 | 0.59 | 20 | 90% |

**Key Finding**: MLE approach reduces MAE by 15% for sparse items; optimal k=20 balances accuracy and generalization.

### Section 2 Performance

| Method | Precision@10 | Recall@10 | NDCG@10 | Latency |
|--------|--------------|-----------|---------|---------|
| Content-Based | 0.156 | 0.134 | 0.178 | 145 ms |
| CF (Item) | 0.187 | 0.165 | 0.201 | 90 ms |
| **Hybrid** | **0.213** | **0.189** | **0.234** | 198 ms |
| Cascade | 0.207 | 0.184 | 0.228 | 124 ms |

**Key Finding**: Hybrid achieves 18.7% NDCG improvement; Cascade reduces latency by 42%.

### Cold-Start Analysis

| User Ratings | MAE (CB) | MAE (CF) | MAE (Hybrid) |
|--------------|----------|----------|--------------|
| 0-2 | 0.89 | 1.45 | **0.67** |
| 3-5 | 0.76 | 1.12 | **0.58** |
| 10+ | 0.72 | 0.71 | **0.49** |

---

## Troubleshooting

### Section 1

**Memory Error**:
```bash
python svd_analysis.py --sample-size 100000 --use-sparse
```

**Missing Dataset**:
```bash
# Download from: https://grouplens.org/datasets/movielens/20m/
# Place in SECTION_1/data/ratings.csv
```

### Section 2

**PyTorch Issues**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Memory Error**:
```python
# In extract_features.py, reduce batch_size from 16 to 4
```

**Surprise Not Found**:
```bash
pip install scikit-surprise
```

---

## References

**Datasets**:
- MovieLens 20M (Harper & Konstan, 2015)
- Amazon Reviews (McAuley et al., 2023)

**Key Papers**:
- Koren et al. (2009): Matrix Factorization Techniques
- Burke (2002): Hybrid Recommender Systems
- He et al. (2016): Deep Residual Learning

---

## Academic Context

**Course**: AIE425 - Intelligent Recommender Systems  
**Semester**: Fall 2025/2026  
**Institution**: [Galala University]

## License

Educational use only - AIE425 Course Project

---