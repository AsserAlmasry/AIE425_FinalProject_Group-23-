"""
utils.py
========
Shared helper functions used across:
- Statistical Analysis
- Part 1 (Content-based / TF-IDF)
- Part 2 (Feature engineering & similarity)
- Part 3 (Evaluation & robustness)

All functions are:
- Pure Python / NumPy / Pandas
- No scikit-learn dependency
- Deterministic when seed is provided
"""

import os
import re
import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict

# ------------------------------------------------------------------
# General Utilities
# ------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------------
# Text Processing Utilities (Manual TF‑IDF)
# ------------------------------------------------------------------

def tokenize(text: str):
    """Lowercase, keep letters only, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def term_frequency(tokens):
    """Compute log‑scaled TF: tf = 1 + log10(count)."""
    tf = defaultdict(float)
    counts = defaultdict(int)

    for t in tokens:
        counts[t] += 1

    for term, c in counts.items():
        tf[term] = 1.0 + math.log10(c)

    return tf


def compute_idf(docs_tokens):
    """Compute IDF = log10(N / df)."""
    N = len(docs_tokens)
    df = defaultdict(int)

    for tokens in docs_tokens:
        for term in set(tokens):
            df[term] += 1

    idf = {term: math.log10(N / df_val) for term, df_val in df.items()}
    return idf


def compute_tfidf(tokens, idf_dict):
    """Compute TF‑IDF vector for a single document."""
    tf = term_frequency(tokens)
    tfidf = {}

    for term, tf_val in tf.items():
        if term in idf_dict:
            tfidf[term] = tf_val * idf_dict[term]

    return tfidf


def dict_to_vector(tfidf_dict, vocab):
    """Convert sparse TF‑IDF dict to dense vector."""
    return np.array([tfidf_dict.get(term, 0.0) for term in vocab], dtype=np.float32)


# ------------------------------------------------------------------
# Numeric Feature Utilities
# ------------------------------------------------------------------

def zscore(series: pd.Series):
    """Manual Z‑score normalization."""
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return series * 0.0
    return (series - mean) / std


# ------------------------------------------------------------------
# User Profile Construction
# ------------------------------------------------------------------

def build_user_profile(item_vectors, ratings, min_rating=4.0):
    """
    Build user profile as simple average of liked item vectors.
    item_vectors: dict[item_id] -> np.array
    ratings: list of (item_id, rating)
    """
    liked = [item_vectors[i] for i, r in ratings if r >= min_rating and i in item_vectors]

    if not liked:
        return None

    return np.mean(liked, axis=0)


# ------------------------------------------------------------------
# Similarity Utilities (Manual)
# ------------------------------------------------------------------

def cosine_similarity(vec_a, vec_b):
    """Manual cosine similarity."""
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def top_n_similar(user_vec, item_vectors, rated_items=set(), n=5):
    """Return top‑N items with similarity scores."""
    scores = []

    for item_id, vec in item_vectors.items():
        if item_id in rated_items:
            continue
        sim = cosine_similarity(user_vec, vec)
        scores.append((item_id, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n]


# ------------------------------------------------------------------
# Sampling Utilities
# ------------------------------------------------------------------

def sample_users(df, n_users, min_interactions=5, seed=42):
    """Sample users with at least min_interactions."""
    set_seed(seed)
    counts = df.groupby("user_id").size()
    eligible = counts[counts >= min_interactions].index.tolist()
    return random.sample(eligible, min(n_users, len(eligible)))


def sample_items(df, n_items, seed=42):
    """Random item sampling."""
    set_seed(seed)
    items = df["item_id"].unique().tolist()
    return random.sample(items, min(n_items, len(items)))


# ------------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------------

def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


# ------------------------------------------------------------------
# Missing Data / Robustness
# ------------------------------------------------------------------

def mask_matrix(matrix, ratio, seed=42):
    """Randomly mask entries with NaN."""
    set_seed(seed)
    masked = matrix.copy()
    mask = np.random.rand(*matrix.shape) < ratio
    masked[mask] = np.nan
    return masked, mask


def mean_fill(matrix, axis=0):
    """Fill NaNs with row or column mean."""
    filled = matrix.copy()
    means = np.nanmean(filled, axis=axis)

    if axis == 0:
        for j in range(filled.shape[1]):
            filled[np.isnan(filled[:, j]), j] = means[j]
    else:
        for i in range(filled.shape[0]):
            filled[i, np.isnan(filled[i, :])] = means[i]

    return filled