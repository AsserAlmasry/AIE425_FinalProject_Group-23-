"""
Collaborative Filtering Recommendation System
Implements Item-Based CF and Matrix Factorization (SVD)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle

# Try to import Surprise for SVD
try:
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import cross_validate
    SURPRISE_AVAILABLE = True
except:
    SURPRISE_AVAILABLE = False
    print("⚠️  Surprise library not available - SVD will be limited")

class CollaborativeFiltering:
    def __init__(self, train_data, products_df):
        """
        Initialize collaborative filtering system
        
        Args:
            train_data: Training interactions DataFrame
            products_df: Product metadata DataFrame
        """
        self.train_data = train_data
        self.products = products_df
        
        # Create user-item matrix
        self._create_user_item_matrix()
        
        # Precompute item similarity for item-based CF
        print("Precomputing item-item similarity matrix...")
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        print("✓ Item similarity matrix ready")
    
    def _create_user_item_matrix(self):
        """Create sparse user-item rating matrix"""
        print("Creating user-item matrix...")
        
        n_users = self.train_data['user_idx'].max() + 1
        n_items = self.train_data['item_idx'].max() + 1
        
        # Create pivot table
        self.user_item_df = self.train_data.pivot_table(
            index='user_idx',
            columns='item_idx',
            values='rating',
            fill_value=0
        )
        
        self.user_item_matrix = self.user_item_df.values
        
        print(f"✓ User-item matrix shape: {self.user_item_matrix.shape}")
        print(f"✓ Sparsity: {1 - np.count_nonzero(self.user_item_matrix) / self.user_item_matrix.size:.2%}")
    
    def item_based_cf(self, user_idx, k=20, top_n=10):
        """
        Item-based collaborative filtering
        
        Args:
            user_idx: User index
            k: Number of similar items to consider
            top_n: Number of recommendations
        
        Returns:
            List of (item_idx, predicted_score) tuples
        """
        # Get user's ratings
        if user_idx >= len(self.user_item_matrix):
            # Cold user - return popular items
            return self._get_popular_items(top_n)
        
        user_ratings = self.user_item_matrix[user_idx]
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            # No ratings - return popular items
            return self._get_popular_items(top_n)
        
        # Predict ratings for unrated items
        predictions = {}
        unrated_items = np.where(user_ratings == 0)[0]
        
        for item_idx in unrated_items:
            # Get k most similar items that user has rated
            item_similarities = self.item_similarity[item_idx]
            
            # Find similar items among rated items
            similar_rated = []
            for rated_item in rated_items:
                sim = item_similarities[rated_item]
                if sim > 0:  # Only consider positive similarities
                    similar_rated.append((rated_item, sim, user_ratings[rated_item]))
            
            # Sort by similarity and take top-k
            similar_rated.sort(key=lambda x: x[1], reverse=True)
            similar_rated = similar_rated[:k]
            
            if len(similar_rated) == 0:
                continue
            
            # Weighted average prediction
            numerator = sum(sim * rating for _, sim, rating in similar_rated)
            denominator = sum(sim for _, sim, _ in similar_rated)
            
            if denominator > 0:
                predictions[item_idx] = numerator / denominator
        
        # Sort by predicted score and return top-N
        top_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return top_items
    
    def user_based_cf(self, user_idx, k=30, top_n=10):
        """
        User-based collaborative filtering
        
        Args:
            user_idx: User index
            k: Number of similar users to consider
            top_n: Number of recommendations
        
        Returns:
            List of (item_idx, predicted_score) tuples
        """
        if user_idx >= len(self.user_item_matrix):
            return self._get_popular_items(top_n)
        
        # Compute user similarity
        user_vector = self.user_item_matrix[user_idx].reshape(1, -1)
        user_similarities = cosine_similarity(user_vector, self.user_item_matrix)[0]
        
        # Get k most similar users (excluding self)
        user_similarities[user_idx] = -1
        similar_users = np.argsort(user_similarities)[::-1][:k]
        
        # Get items rated by similar users but not by target user
        user_ratings = self.user_item_matrix[user_idx]
        rated_items = set(np.where(user_ratings > 0)[0])
        
        predictions = {}
        
        for item_idx in range(self.user_item_matrix.shape[1]):
            if item_idx in rated_items:
                continue
            
            # Weighted average from similar users
            numerator = 0
            denominator = 0
            
            for similar_user in similar_users:
                rating = self.user_item_matrix[similar_user, item_idx]
                if rating > 0:
                    sim = user_similarities[similar_user]
                    numerator += sim * rating
                    denominator += sim
            
            if denominator > 0:
                predictions[item_idx] = numerator / denominator
        
        # Sort and return top-N
        top_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return top_items
    
    def matrix_factorization_svd(self, n_factors=50, n_epochs=20):
        """
        Matrix factorization using SVD from Surprise library
        
        Args:
            n_factors: Number of latent factors
            n_epochs: Number of training epochs
        """
        if not SURPRISE_AVAILABLE:
            print("⚠️  Surprise not available - skipping SVD training")
            self.svd_model = None
            return None
        
        print(f"\nTraining SVD (factors={n_factors}, epochs={n_epochs})...")
        
        # Prepare data for Surprise
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            self.train_data[['user_idx', 'item_idx', 'rating']], 
            reader
        )
        
        # Train SVD
        self.svd_model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            random_state=42,
            verbose=True
        )
        
        trainset = data.build_full_trainset()
        self.svd_model.fit(trainset)
        
        print("✓ SVD model trained")
        
        return self.svd_model
    
    def predict_with_svd(self, user_idx, item_idx):
        """
        Predict rating using trained SVD model
        
        Args:
            user_idx: User index
            item_idx: Item index
        
        Returns:
            Predicted rating
        """
        if self.svd_model is None:
            return 3.0  # Default middle rating
        
        prediction = self.svd_model.predict(user_idx, item_idx)
        return prediction.est
    
    def recommend_with_svd(self, user_idx, top_n=10):
        """
        Generate recommendations using SVD
        
        Args:
            user_idx: User index
            top_n: Number of recommendations
        
        Returns:
            List of (item_idx, predicted_rating) tuples
        """
        if self.svd_model is None:
            return self._get_popular_items(top_n)
        
        # Get user's rated items
        user_ratings = self.user_item_matrix[user_idx] if user_idx < len(self.user_item_matrix) else np.zeros(self.user_item_matrix.shape[1])
        rated_items = set(np.where(user_ratings > 0)[0])
        
        # Predict for all unrated items
        predictions = []
        for item_idx in range(self.user_item_matrix.shape[1]):
            if item_idx not in rated_items:
                pred = self.predict_with_svd(user_idx, item_idx)
                predictions.append((item_idx, pred))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:top_n]
    
    def _get_popular_items(self, top_n=10):
        """
        Get most popular items (fallback for cold users)
        
        Args:
            top_n: Number of items to return
        
        Returns:
            List of (item_idx, popularity_score) tuples
        """
        # Count ratings per item
        item_counts = self.train_data['item_idx'].value_counts()
        
        # Get top-N
        top_items = [(idx, count) for idx, count in item_counts.head(top_n).items()]
        
        return top_items
    
    def get_recommendations_df(self, recommendations, method='item_cf'):
        """
        Convert recommendation tuples to DataFrame with product info
        
        Args:
            recommendations: List of (item_idx, score) tuples
            method: Method name for the score column
        
        Returns:
            DataFrame with product information and scores
        """
        if len(recommendations) == 0:
            return pd.DataFrame()
        
        item_indices = [item_idx for item_idx, _ in recommendations]
        scores = [score for _, score in recommendations]
        
        recs_df = self.products[self.products['item_idx'].isin(item_indices)].copy()
        
        # Add scores in correct order
        score_map = dict(recommendations)
        recs_df['score'] = recs_df['item_idx'].map(score_map)
        recs_df = recs_df.sort_values('score', ascending=False)
        recs_df['rank'] = range(1, len(recs_df) + 1)
        
        return recs_df


def demonstrate_collaborative():
    """
    Demonstrate collaborative filtering methods
    """
    print("="*60)
    print("  Collaborative Filtering Demo")
    print("="*60)
    
    # Load data
    train_data = pd.read_csv('data/processed/train.csv')
    products = pd.read_csv('data/processed/products.csv')
    
    # Initialize CF system
    cf_model = CollaborativeFiltering(train_data, products)
    
    # Example user
    sample_user_idx = train_data['user_idx'].iloc[10]
    
    print(f"\n--- Recommendations for User {sample_user_idx} ---")
    
    # Item-based CF
    print("\n1. Item-Based Collaborative Filtering:")
    item_cf_recs = cf_model.item_based_cf(sample_user_idx, k=20, top_n=5)
    item_cf_df = cf_model.get_recommendations_df(item_cf_recs, 'item_cf')
    
    for idx, row in item_cf_df.iterrows():
        print(f"  {row['rank']}. {row['title']} (Score: {row['score']:.3f})")
    
    # User-based CF
    print("\n2. User-Based Collaborative Filtering:")
    user_cf_recs = cf_model.user_based_cf(sample_user_idx, k=30, top_n=5)
    user_cf_df = cf_model.get_recommendations_df(user_cf_recs, 'user_cf')
    
    for idx, row in user_cf_df.iterrows():
        print(f"  {row['rank']}. {row['title']} (Score: {row['score']:.3f})")
    
    # SVD (if available)
    if SURPRISE_AVAILABLE:
        print("\n3. Training SVD Model...")
        cf_model.matrix_factorization_svd(n_factors=20, n_epochs=10)
        
        print("\nSVD Recommendations:")
        svd_recs = cf_model.recommend_with_svd(sample_user_idx, top_n=5)
        svd_df = cf_model.get_recommendations_df(svd_recs, 'svd')
        
        for idx, row in svd_df.iterrows():
            print(f"  {row['rank']}. {row['title']} (Predicted Rating: {row['score']:.3f})")
    
    return cf_model


if __name__ == "__main__":
    demonstrate_collaborative()