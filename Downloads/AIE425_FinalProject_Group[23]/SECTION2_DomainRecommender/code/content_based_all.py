"""
Content-Based Recommendation System
Uses multi-modal features (text, image, category, price) AND user review text for recommendations
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class MultiModalContentBased:
    def __init__(self, products_df, fused_features, train_data):
        """
        Initialize content-based recommender
        
        Args:
            products_df: Product metadata DataFrame
            fused_features: Multi-modal fused feature matrix (n_items × n_features)
            train_data: Training interactions DataFrame (must include review_text column)
        """
        self.products = products_df
        self.item_profiles = fused_features
        self.train_data = train_data
        
        print(f"Initialized CB system with {len(products_df)} items")
        print(f"Feature dimensionality: {fused_features.shape[1]}")
        
        # Build review-enhanced profiles
        self._build_review_enhanced_profiles()
        
        # Precompute item-item similarity matrix for efficiency
        print("Precomputing item similarity matrix...")
        self.item_similarity = cosine_similarity(self.item_profiles)
        print("✓ Item similarity matrix computed")
    
    def _build_review_enhanced_profiles(self):
        """
        Enhance item profiles with aggregated review text
        Combines original multi-modal features with review-based features
        """
        print("\nBuilding review-enhanced item profiles...")
        
        # Check if review_text exists
        if 'review_text' not in self.train_data.columns:
            print("⚠️  No review_text column found, using original profiles only")
            return
        
        # Aggregate reviews per product
        print("Aggregating reviews per product...")
        product_reviews = self.train_data.groupby('item_idx').agg({
            'review_text': lambda x: ' '.join(x.dropna().astype(str)),
            'rating': ['mean', 'count']
        }).reset_index()
        
        product_reviews.columns = ['item_idx', 'aggregated_reviews', 'avg_rating', 'num_reviews']
        
        # Merge with products
        self.products = self.products.merge(
            product_reviews,
            on='item_idx',
            how='left'
        )
        
        # Fill missing
        self.products['aggregated_reviews'] = self.products['aggregated_reviews'].fillna('')
        self.products['avg_rating'] = self.products['avg_rating'].fillna(3.0)
        self.products['num_reviews'] = self.products['num_reviews'].fillna(0)
        
        # Extract review features
        print("Extracting TF-IDF from reviews...")
        self.review_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            min_df=2,
            max_df=0.7,
            ngram_range=(1, 2)
        )
        
        review_texts = self.products['aggregated_reviews'].fillna('')
        self.review_features = self.review_vectorizer.fit_transform(review_texts).toarray()
        
        print(f"✓ Extracted {self.review_features.shape[1]} review features")
        
        # Normalize review features
        review_norm = self.review_features / (np.linalg.norm(self.review_features, axis=1, keepdims=True) + 1e-10)
        
        # Normalize original features
        original_norm = self.item_profiles / (np.linalg.norm(self.item_profiles, axis=1, keepdims=True) + 1e-10)
        
        # Combine: 60% original multi-modal + 40% review text
        self.item_profiles = np.concatenate([
            original_norm * 0.6,
            review_norm * 0.4
        ], axis=1)
        
        print(f"✓ Enhanced profiles: {self.item_profiles.shape}")
        print(f"  - Original features: {original_norm.shape[1]} dims × 0.6")
        print(f"  - Review features: {review_norm.shape[1]} dims × 0.4")
    
    def create_user_profile(self, user_ratings, use_review_text=True):
        """
        Create user profile from rated items
        Uses weighted average based on ratings
        Optionally incorporates user's own review text
        
        Args:
            user_ratings: DataFrame with user's ratings (columns: item_idx, rating, review_text)
            use_review_text: Whether to use user's review text (if available)
        
        Returns:
            User profile vector
        """
        if len(user_ratings) == 0:
            # Return zero vector for cold users
            return np.zeros(self.item_profiles.shape[1])
        
        profile = np.zeros(self.item_profiles.shape[1])
        total_weight = 0
        
        # Standard weighted profile from item features
        for _, row in user_ratings.iterrows():
            item_idx = row['item_idx']
            rating = row['rating']
            
            # Find item position in products dataframe
            item_pos = self.products[self.products['item_idx'] == item_idx].index
            
            if len(item_pos) > 0:
                item_pos = item_pos[0]
                # Weighted accumulation (higher ratings contribute more)
                profile += self.item_profiles[item_pos] * rating
                total_weight += rating
        
        # Normalize by total weight
        if total_weight > 0:
            profile = profile / total_weight
        
        # Additional: Enhance with user's own review text if available
        if use_review_text and 'review_text' in user_ratings.columns and hasattr(self, 'review_vectorizer'):
            user_review_text = ' '.join(user_ratings['review_text'].dropna().astype(str))
            
            if len(user_review_text.strip()) > 0:
                try:
                    # Extract features from user's reviews
                    user_review_features = self.review_vectorizer.transform([user_review_text]).toarray()[0]
                    user_review_norm = user_review_features / (np.linalg.norm(user_review_features) + 1e-10)
                    
                    # Get the review portion of the profile (last part)
                    review_dim = len(user_review_norm)
                    
                    # Blend user's actual review language (30%) with item-based profile (70%)
                    if profile.shape[0] >= review_dim:
                        profile[-review_dim:] = 0.7 * profile[-review_dim:] + 0.3 * user_review_norm
                except:
                    pass  # If extraction fails, just use item-based profile
        
        return profile
    
    def compute_similarity(self, user_profile, exclude_items=None):
        """
        Compute cosine similarity between user profile and all items
        
        Args:
            user_profile: User profile vector
            exclude_items: Set of item indices to exclude from recommendations
        
        Returns:
            Similarity scores array
        """
        # Handle cold-start case
        if np.allclose(user_profile, 0):
            # Return popularity-based scores (rating × reviews)
            if 'avg_rating' in self.products.columns and 'num_reviews' in self.products.columns:
                return (self.products['avg_rating'].fillna(0).values * 
                       np.sqrt(self.products['num_reviews'].fillna(0).values))
            else:
                return self.products['popularity_score'].values
        
        # Compute cosine similarity
        similarities = cosine_similarity(
            user_profile.reshape(1, -1),
            self.item_profiles
        )[0]
        
        # Exclude already rated items
        if exclude_items is not None:
            similarities[list(exclude_items)] = -np.inf
        
        return similarities
    
    def recommend(self, user_id=None, user_idx=None, top_n=10, 
                  exclude_rated=True, return_scores=True, use_review_text=True):
        """
        Generate top-N recommendations for a user
        
        Args:
            user_id: User ID (string)
            user_idx: User index (integer) - if provided, overrides user_id
            top_n: Number of recommendations
            exclude_rated: Whether to exclude already rated items
            return_scores: Whether to include similarity scores
            use_review_text: Whether to use user's review text in profile
        
        Returns:
            DataFrame with recommendations
        """
        # Get user's rating history
        if user_idx is not None:
            user_ratings = self.train_data[self.train_data['user_idx'] == user_idx]
        elif user_id is not None:
            user_ratings = self.train_data[self.train_data['user_id'] == user_id]
        else:
            raise ValueError("Must provide either user_id or user_idx")
        
        # Create user profile (with review text if available)
        user_profile = self.create_user_profile(user_ratings, use_review_text=use_review_text)
        
        # Get items to exclude
        exclude_items = set(user_ratings['item_idx'].tolist()) if exclude_rated else None
        
        # Compute similarities
        similarity_scores = self.compute_similarity(user_profile, exclude_items)
        
        # Get top-N items
        top_indices = np.argsort(similarity_scores)[::-1][:top_n]
        
        # Create recommendations DataFrame
        recommendations = self.products.iloc[top_indices].copy()
        
        if return_scores:
            recommendations['score'] = similarity_scores[top_indices]
            recommendations['rank'] = range(1, len(recommendations) + 1)
        
        return recommendations
    
    def item_based_knn(self, target_item_idx, k=20, exclude_self=True):
        """
        Find k most similar items to target item
        
        Args:
            target_item_idx: Index of target item
            k: Number of similar items to return
            exclude_self: Whether to exclude the item itself
        
        Returns:
            DataFrame with similar items and similarity scores
        """
        # Get similarities from precomputed matrix
        similarities = self.item_similarity[target_item_idx]
        
        # Exclude self if requested
        if exclude_self:
            similarities[target_item_idx] = -np.inf
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:k]
        
        similar_items = self.products.iloc[top_indices].copy()
        similar_items['similarity'] = similarities[top_indices]
        
        return similar_items
    
    def explain_recommendation(self, user_profile, item_idx, top_k_features=5):
        """
        Explain why an item was recommended
        
        Args:
            user_profile: User profile vector
            item_idx: Item index
            top_k_features: Number of top contributing features to show
        
        Returns:
            Dictionary with explanation
        """
        item_profile = self.item_profiles[item_idx]
        
        # Compute contribution of each feature
        contributions = user_profile * item_profile
        
        # Get top contributing features
        top_feature_indices = np.argsort(np.abs(contributions))[::-1][:top_k_features]
        
        explanation = {
            'item': self.products.iloc[item_idx]['title'],
            'overall_similarity': cosine_similarity(
                user_profile.reshape(1, -1),
                item_profile.reshape(1, -1)
            )[0, 0],
            'top_features': [
                {
                    'feature_idx': int(idx),
                    'contribution': float(contributions[idx])
                }
                for idx in top_feature_indices
            ]
        }
        
        return explanation
    
    def batch_recommend(self, user_indices, top_n=10):
        """
        Generate recommendations for multiple users efficiently
        
        Args:
            user_indices: List of user indices
            top_n: Number of recommendations per user
        
        Returns:
            Dictionary mapping user_idx -> recommendations DataFrame
        """
        recommendations = {}
        
        for user_idx in user_indices:
            try:
                recs = self.recommend(user_idx=user_idx, top_n=top_n)
                recommendations[user_idx] = recs
            except Exception as e:
                print(f"Warning: Could not generate recs for user {user_idx}: {e}")
                recommendations[user_idx] = pd.DataFrame()
        
        return recommendations
    
    def get_cold_start_recommendations(self, user_profile_items, top_n=10):
        """
        Recommendations for completely new users based on a few liked items
        
        Args:
            user_profile_items: List of item indices the new user likes
            top_n: Number of recommendations
        
        Returns:
            DataFrame with recommendations
        """
        # Create profile from given items (equal weight)
        profile = np.zeros(self.item_profiles.shape[1])
        
        for item_idx in user_profile_items:
            profile += self.item_profiles[item_idx]
        
        profile = profile / len(user_profile_items)
        
        # Exclude the input items
        similarities = self.compute_similarity(profile, exclude_items=set(user_profile_items))
        
        # Get top-N
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        recommendations = self.products.iloc[top_indices].copy()
        recommendations['score'] = similarities[top_indices]
        
        return recommendations


def demonstrate_content_based():
    """
    Demonstrate content-based recommendation with numerical example
    """
    print("="*60)
    print("  Content-Based Recommendation Demo")
    print("="*60)
    
    # Load data
    products = pd.read_csv('data/processed/products.csv')
    train_data = pd.read_csv('data/processed/train.csv')
    fused_features = np.load('data/features/fused_features.npy')
    
    # Initialize system
    cb_model = MultiModalContentBased(products, fused_features, train_data)
    
    # Example: Recommend for a random user
    sample_user_idx = train_data['user_idx'].iloc[0]
    
    print(f"\n--- Recommendations for User {sample_user_idx} ---")
    
    # Get user's history
    user_history = train_data[train_data['user_idx'] == sample_user_idx]
    print(f"\nUser has rated {len(user_history)} items:")
    for _, row in user_history.head(5).iterrows():
        item = products[products['item_idx'] == row['item_idx']].iloc[0]
        print(f"  • {item['title']} (Rating: {row['rating']})")
    
    # Generate recommendations
    recommendations = cb_model.recommend(user_idx=sample_user_idx, top_n=10)
    
    print(f"\nTop-10 Recommendations:")
    for idx, row in recommendations.iterrows():
        print(f"  {row['rank']}. {row['title']}")
        print(f"     Category: {row['main_category']} | Score: {row['score']:.3f}")
    
    # Item-based KNN demo
    print(f"\n--- Similar Items Demo ---")
    sample_item_idx = products['item_idx'].iloc[0]
    sample_item = products[products['item_idx'] == sample_item_idx].iloc[0]
    
    print(f"\nFinding items similar to: {sample_item['title']}")
    
    similar = cb_model.item_based_knn(sample_item_idx, k=5)
    print("\nTop-5 Similar Items:")
    for idx, row in similar.iterrows():
        print(f"  • {row['title']} (Similarity: {row['similarity']:.3f})")
    
    return cb_model


if __name__ == "__main__":
    demonstrate_content_based()