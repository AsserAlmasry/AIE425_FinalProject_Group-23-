"""
Hybrid Recommendation System
Combines Content-Based and Collaborative Filtering using multiple strategies
"""

import numpy as np
import pandas as pd
from content_based import MultiModalContentBased
from collaborative import CollaborativeFiltering

class HybridRecommender:
    def __init__(self, content_based_model, collaborative_model, 
                 products_df, train_data, alpha=0.5):
        """
        Initialize hybrid recommender
        
        Args:
            content_based_model: Trained content-based model
            collaborative_model: Trained collaborative filtering model
            products_df: Product metadata
            train_data: Training data
            alpha: Weight for content-based in weighted hybrid (0-1)
        """
        self.cb = content_based_model
        self.cf = collaborative_model
        self.products = products_df
        self.train_data = train_data
        self.alpha = alpha
        
        print(f"Initialized Hybrid Recommender (α={alpha})")
    
    def weighted_hybrid(self, user_idx, top_n=10):
        """
        Weighted hybrid: α × CB + (1-α) × CF
        
        Args:
            user_idx: User index
            top_n: Number of recommendations
        
        Returns:
            DataFrame with recommendations
        """
        # Get CB recommendations
        cb_recs = self.cb.recommend(user_idx=user_idx, top_n=100, return_scores=True)
        cb_scores = dict(zip(cb_recs['item_idx'], cb_recs['score']))
        
        # Get CF recommendations (item-based)
        cf_recs = self.cf.item_based_cf(user_idx, k=20, top_n=100)
        cf_scores = dict(cf_recs)
        
        # Normalize scores to [0, 1]
        max_cb = max(cb_scores.values()) if cb_scores else 1
        max_cf = max(score for _, score in cf_recs) if cf_recs else 1
        
        cb_scores_norm = {k: v/max_cb for k, v in cb_scores.items()}
        cf_scores_norm = {k: v/max_cf for k, v in cf_scores.items()}
        
        # Combine scores
        all_items = set(cb_scores_norm.keys()) | set(cf_scores_norm.keys())
        hybrid_scores = {}
        
        for item_idx in all_items:
            cb_score = cb_scores_norm.get(item_idx, 0)
            cf_score = cf_scores_norm.get(item_idx, 0)
            
            hybrid_scores[item_idx] = (
                self.alpha * cb_score + 
                (1 - self.alpha) * cf_score
            )
        
        # Get top-N
        top_items = sorted(hybrid_scores.items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:top_n]
        
        # Convert to DataFrame
        item_indices = [item_idx for item_idx, _ in top_items]
        scores = [score for _, score in top_items]
        
        recommendations = self.products[self.products['item_idx'].isin(item_indices)].copy()
        score_map = dict(top_items)
        recommendations['score'] = recommendations['item_idx'].map(score_map)
        recommendations = recommendations.sort_values('score', ascending=False)
        recommendations['rank'] = range(1, len(recommendations) + 1)
        
        # Add breakdown for interpretability
        recommendations['cb_score'] = recommendations['item_idx'].map(cb_scores_norm)
        recommendations['cf_score'] = recommendations['item_idx'].map(cf_scores_norm)
        
        return recommendations
    
    def switching_hybrid(self, user_idx, top_n=10, cold_threshold=5):
        """
        Switching hybrid: Use CB for cold users, CF for warm users
        
        Args:
            user_idx: User index
            top_n: Number of recommendations
            cold_threshold: Minimum ratings to be considered "warm"
        
        Returns:
            Tuple of (DataFrame, method_used)
        """
        # Count user's ratings
        user_ratings = self.train_data[self.train_data['user_idx'] == user_idx]
        n_ratings = len(user_ratings)
        
        if n_ratings < cold_threshold:
            # Cold user - use content-based
            recommendations = self.cb.recommend(user_idx=user_idx, top_n=top_n)
            method = 'content_based'
        else:
            # Warm user - use collaborative filtering
            cf_recs = self.cf.item_based_cf(user_idx, top_n=top_n)
            recommendations = self.cf.get_recommendations_df(cf_recs)
            method = 'collaborative'
        
        return recommendations, method
    
    def cascade_hybrid(self, user_idx, top_n=10, candidate_n=50):
        """
        Cascade hybrid: CB generates candidates, CF ranks them
        Two-stage process reduces computation while maintaining quality
        
        Args:
            user_idx: User index
            top_n: Final number of recommendations
            candidate_n: Number of candidates from CB stage
        
        Returns:
            DataFrame with recommendations
        """
        # Stage 1: Content-based generates candidates
        cb_candidates = self.cb.recommend(
            user_idx=user_idx, 
            top_n=candidate_n,
            return_scores=True
        )
        
        candidate_items = cb_candidates['item_idx'].tolist()
        
        # Stage 2: Collaborative filtering ranks candidates
        # Use item-based CF to predict scores for candidates
        user_ratings = self.cf.user_item_matrix[user_idx] if user_idx < len(self.cf.user_item_matrix) else np.zeros(self.cf.user_item_matrix.shape[1])
        rated_items = np.where(user_ratings > 0)[0]
        
        cf_scores = {}
        
        for item_idx in candidate_items:
            # Get similarities to rated items
            similarities = []
            for rated_item in rated_items:
                sim = self.cf.item_similarity[item_idx, rated_item]
                if sim > 0:
                    similarities.append((sim, user_ratings[rated_item]))
            
            if similarities:
                # Weighted average
                similarities.sort(key=lambda x: x[0], reverse=True)
                top_k = similarities[:20]
                
                numerator = sum(sim * rating for sim, rating in top_k)
                denominator = sum(sim for sim, _ in top_k)
                
                cf_scores[item_idx] = numerator / denominator if denominator > 0 else 3.0
            else:
                cf_scores[item_idx] = 3.0
        
        # Sort by CF scores
        ranked_items = sorted(cf_scores.items(), 
                            key=lambda x: x[1], 
                            reverse=True)[:top_n]
        
        # Convert to DataFrame
        recommendations = self.products[
            self.products['item_idx'].isin([item for item, _ in ranked_items])
        ].copy()
        
        score_map = dict(ranked_items)
        recommendations['score'] = recommendations['item_idx'].map(score_map)
        recommendations = recommendations.sort_values('score', ascending=False)
        recommendations['rank'] = range(1, len(recommendations) + 1)
        
        return recommendations
    
    def feature_weighted_hybrid(self, user_idx, top_n=10):
        """
        Feature-weighted hybrid: Use different weights based on user characteristics
        Adapts α based on user's rating history diversity
        
        Args:
            user_idx: User index
            top_n: Number of recommendations
        
        Returns:
            DataFrame with recommendations
        """
        # Analyze user's rating pattern
        user_ratings = self.train_data[self.train_data['user_idx'] == user_idx]
        
        # Calculate diversity (number of unique categories rated)
        rated_items = user_ratings['item_idx'].tolist()
        rated_products = self.products[self.products['item_idx'].isin(rated_items)]
        n_categories = rated_products['main_category'].nunique()
        
        # Adaptive weight: More diverse users benefit more from CF
        adaptive_alpha = 0.7 if n_categories <= 2 else 0.3
        
        # Temporarily adjust alpha
        original_alpha = self.alpha
        self.alpha = adaptive_alpha
        
        # Use weighted hybrid with adaptive alpha
        recommendations = self.weighted_hybrid(user_idx, top_n)
        
        # Restore original alpha
        self.alpha = original_alpha
        
        recommendations['adaptive_alpha'] = adaptive_alpha
        
        return recommendations
    
    def mixed_hybrid(self, user_idx, top_n=10, cb_ratio=0.5):
        """
        Mixed hybrid: Combine recommendations from both methods
        Take top items from each and merge
        
        Args:
            user_idx: User index
            top_n: Total number of recommendations
            cb_ratio: Proportion from content-based (0-1)
        
        Returns:
            DataFrame with recommendations
        """
        n_cb = int(top_n * cb_ratio)
        n_cf = top_n - n_cb
        
        # Get recommendations from each method
        cb_recs = self.cb.recommend(user_idx=user_idx, top_n=n_cb)
        cf_recs = self.cf.item_based_cf(user_idx, top_n=n_cf)
        cf_df = self.cf.get_recommendations_df(cf_recs)
        
        # Combine (avoid duplicates)
        cb_items = set(cb_recs['item_idx'])
        cf_items = set(cf_df['item_idx'])
        
        # If overlap, get additional items
        overlap = cb_items & cf_items
        if overlap:
            # Get more CF items to compensate
            additional_cf = self.cf.item_based_cf(user_idx, top_n=n_cf + len(overlap))
            cf_df = self.cf.get_recommendations_df(additional_cf)
            cf_df = cf_df[~cf_df['item_idx'].isin(cb_items)].head(n_cf)
        
        # Combine
        cb_recs['method'] = 'content_based'
        cf_df['method'] = 'collaborative'
        
        recommendations = pd.concat([cb_recs, cf_df], ignore_index=True)
        recommendations['rank'] = range(1, len(recommendations) + 1)
        
        return recommendations
    
    def evaluate_cold_start(self, test_user_indices, rating_thresholds=[0, 3, 5, 10]):
        """
        Evaluate hybrid performance on cold-start users
        
        Args:
            test_user_indices: List of user indices to test
            rating_thresholds: List of rating count thresholds
        
        Returns:
            Dictionary with results per threshold
        """
        results = {}
        
        for threshold in rating_thresholds:
            # Filter users by rating count
            user_rating_counts = self.train_data.groupby('user_idx').size()
            
            if threshold == 0:
                # Brand new users (simulate)
                cold_users = test_user_indices[:10]
            else:
                cold_users = user_rating_counts[
                    user_rating_counts == threshold
                ].index.tolist()
                cold_users = [u for u in cold_users if u in test_user_indices][:10]
            
            if len(cold_users) == 0:
                continue
            
            # Generate recommendations using switching strategy
            recommendations_list = []
            methods_used = []
            
            for user_idx in cold_users:
                recs, method = self.switching_hybrid(user_idx, top_n=10, 
                                                    cold_threshold=5)
                recommendations_list.append(recs)
                methods_used.append(method)
            
            results[f'{threshold}_ratings'] = {
                'n_users': len(cold_users),
                'cb_used': methods_used.count('content_based'),
                'cf_used': methods_used.count('collaborative'),
                'avg_recs': np.mean([len(r) for r in recommendations_list])
            }
        
        return pd.DataFrame(results).T


def demonstrate_hybrid():
    """
    Demonstrate all hybrid strategies
    """
    print("="*60)
    print("  Hybrid Recommendation Demo")
    print("="*60)
    
    # Load data
    import numpy as np
    products = pd.read_csv('data/processed/products.csv')
    train_data = pd.read_csv('data/processed/train.csv')
    fused_features = np.load('data/features/fused_features.npy')
    
    # Initialize models
    from content_based import MultiModalContentBased
    from collaborative import CollaborativeFiltering
    
    print("\nInitializing models...")
    cb_model = MultiModalContentBased(products, fused_features, train_data)
    cf_model = CollaborativeFiltering(train_data, products)
    
    # Initialize hybrid
    hybrid = HybridRecommender(cb_model, cf_model, products, train_data, alpha=0.5)
    
    # Example user
    sample_user_idx = train_data['user_idx'].iloc[5]
    
    print(f"\n--- Recommendations for User {sample_user_idx} ---")
    
    # 1. Weighted Hybrid
    print("\n1. Weighted Hybrid (α=0.5):")
    weighted_recs = hybrid.weighted_hybrid(sample_user_idx, top_n=5)
    for idx, row in weighted_recs.iterrows():
        print(f"  {row['rank']}. {row['title']}")
        print(f"     Score: {row['score']:.3f} (CB: {row['cb_score']:.3f}, CF: {row['cf_score']:.3f})")
    
    # 2. Switching Hybrid
    print("\n2. Switching Hybrid:")
    switch_recs, method = hybrid.switching_hybrid(sample_user_idx, top_n=5)
    print(f"   Method used: {method}")
    for idx, row in switch_recs.head(5).iterrows():
        print(f"  {row.get('rank', idx+1)}. {row['title']}")
    
    # 3. Cascade Hybrid
    print("\n3. Cascade Hybrid:")
    cascade_recs = hybrid.cascade_hybrid(sample_user_idx, top_n=5)
    for idx, row in cascade_recs.iterrows():
        print(f"  {row['rank']}. {row['title']} (Score: {row['score']:.3f})")
    
    return hybrid


if __name__ == "__main__":
    demonstrate_hybrid()