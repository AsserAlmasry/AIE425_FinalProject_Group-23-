"""
Main Pipeline for Multi-Modal Product Recommendation System
Executes complete workflow and generates final recommendations
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import time

# Import custom modules
from content_based import MultiModalContentBased
from collaborative import CollaborativeFiltering
from hybrid import HybridRecommender

def load_all_data():
    """Load all preprocessed data and features"""
    print("Loading data and features...")
    
    # Load processed data
    products = pd.read_csv('data/processed/products.csv')
    train_data = pd.read_csv('data/processed/train.csv')
    test_data = pd.read_csv('data/processed/test.csv')
    
    # Load features
    fused_features = np.load('data/features/fused_features.npy')
    
    print(f"✓ Products: {len(products)}")
    print(f"✓ Train interactions: {len(train_data)}")
    print(f"✓ Test interactions: {len(test_data)}")
    print(f"✓ Feature dimensions: {fused_features.shape}")
    
    return products, train_data, test_data, fused_features

def initialize_models(products, train_data, fused_features):
    """Initialize all recommendation models"""
    print("\nInitializing models...")
    
    # Content-Based Model
    print("\n1. Content-Based Model")
    cb_model = MultiModalContentBased(products, fused_features, train_data)
    
    # Collaborative Filtering Model
    print("\n2. Collaborative Filtering Model")
    cf_model = CollaborativeFiltering(train_data, products)
    
    # Train SVD if available
    try:
        cf_model.matrix_factorization_svd(n_factors=20, n_epochs=10)
    except:
        print("⚠️  SVD training skipped")
    
    # Hybrid Model
    print("\n3. Hybrid Model")
    hybrid_model = HybridRecommender(
        cb_model, 
        cf_model, 
        products, 
        train_data, 
        alpha=0.5
    )
    
    return cb_model, cf_model, hybrid_model

def generate_sample_recommendations(models, train_data, products, n_users=5):
    """
    Generate recommendations for sample users using all methods
    
    Args:
        models: Tuple of (cb_model, cf_model, hybrid_model)
        train_data: Training data
        products: Product metadata
        n_users: Number of sample users
    
    Returns:
        Dictionary with recommendations
    """
    cb_model, cf_model, hybrid_model = models
    
    print("\n" + "="*60)
    print("  Generating Sample Recommendations")
    print("="*60)
    
    # Select diverse sample users
    user_rating_counts = train_data.groupby('user_idx').size()
    
    # Get users with different activity levels
    cold_users = user_rating_counts[user_rating_counts <= 5].index.tolist()[:1]
    medium_users = user_rating_counts[user_rating_counts.between(6, 15)].index.tolist()[:2]
    active_users = user_rating_counts[user_rating_counts > 15].index.tolist()[:2]
    
    sample_users = cold_users + medium_users + active_users
    
    all_recommendations = {}
    
    for user_idx in sample_users:
        print(f"\n--- User {user_idx} ---")
        
        # Get user info
        user_history = train_data[train_data['user_idx'] == user_idx]
        n_ratings = len(user_history)
        print(f"Ratings: {n_ratings}")
        
        user_results = {
            'user_idx': int(user_idx),
            'n_ratings': int(n_ratings),
            'methods': {}
        }
        
        # 1. Content-Based
        print("  Generating CB recommendations...")
        start_time = time.time()
        cb_recs = cb_model.recommend(user_idx=user_idx, top_n=10)
        cb_time = time.time() - start_time
        
        user_results['methods']['content_based'] = {
            'recommendations': [
                {
                    'rank': int(row['rank']),
                    'item_idx': int(row['item_idx']),
                    'asin': row['asin'],
                    'title': row['title'],
                    'category': row['main_category'],
                    'score': float(row['score'])
                }
                for _, row in cb_recs.head(10).iterrows()
            ],
            'time_ms': int(cb_time * 1000)
        }
        
        # 2. Collaborative Filtering (Item-based)
        print("  Generating CF recommendations...")
        start_time = time.time()
        cf_recs = cf_model.item_based_cf(user_idx, top_n=10)
        cf_df = cf_model.get_recommendations_df(cf_recs)
        cf_time = time.time() - start_time
        
        user_results['methods']['collaborative'] = {
            'recommendations': [
                {
                    'rank': int(row['rank']),
                    'item_idx': int(row['item_idx']),
                    'asin': row['asin'],
                    'title': row['title'],
                    'category': row['main_category'],
                    'score': float(row['score'])
                }
                for _, row in cf_df.head(10).iterrows()
            ],
            'time_ms': int(cf_time * 1000)
        }
        
        # 3. Hybrid (Weighted)
        print("  Generating Hybrid recommendations...")
        start_time = time.time()
        hybrid_recs = hybrid_model.weighted_hybrid(user_idx, top_n=10)
        hybrid_time = time.time() - start_time
        
        user_results['methods']['hybrid_weighted'] = {
            'recommendations': [
                {
                    'rank': int(row['rank']),
                    'item_idx': int(row['item_idx']),
                    'asin': row['asin'],
                    'title': row['title'],
                    'category': row['main_category'],
                    'score': float(row['score']),
                    'cb_score': float(row['cb_score']),
                    'cf_score': float(row['cf_score'])
                }
                for _, row in hybrid_recs.head(10).iterrows()
            ],
            'time_ms': int(hybrid_time * 1000)
        }
        
        # 4. Hybrid (Cascade)
        print("  Generating Cascade recommendations...")
        start_time = time.time()
        cascade_recs = hybrid_model.cascade_hybrid(user_idx, top_n=10)
        cascade_time = time.time() - start_time
        
        user_results['methods']['hybrid_cascade'] = {
            'recommendations': [
                {
                    'rank': int(row['rank']),
                    'item_idx': int(row['item_idx']),
                    'asin': row['asin'],
                    'title': row['title'],
                    'category': row['main_category'],
                    'score': float(row['score'])
                }
                for _, row in cascade_recs.head(10).iterrows()
            ],
            'time_ms': int(cascade_time * 1000)
        }
        
        all_recommendations[f'user_{user_idx}'] = user_results
        
        # Display top-3 from hybrid
        print(f"\n  Top-3 Hybrid Recommendations:")
        for _, row in hybrid_recs.head(3).iterrows():
            print(f"    {row['rank']}. {row['title']}")
            print(f"       Category: {row['main_category']} | Score: {row['score']:.3f}")
    
    return all_recommendations

def save_results(recommendations):
    """Save recommendation results"""
    print("\n💾 Saving results...")
    
    # Save as JSON
    results_dir = Path('results/recommendations')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'sample_recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print(f"✓ Saved to {results_dir / 'sample_recommendations.json'}")
    
    # Create summary statistics
    summary = {
        'n_users': len(recommendations),
        'avg_time_ms': {
            'content_based': np.mean([
                r['methods']['content_based']['time_ms'] 
                for r in recommendations.values()
            ]),
            'collaborative': np.mean([
                r['methods']['collaborative']['time_ms'] 
                for r in recommendations.values()
            ]),
            'hybrid_weighted': np.mean([
                r['methods']['hybrid_weighted']['time_ms'] 
                for r in recommendations.values()
            ]),
            'hybrid_cascade': np.mean([
                r['methods']['hybrid_cascade']['time_ms'] 
                for r in recommendations.values()
            ])
        }
    }
    
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary to {results_dir / 'summary.json'}")
    
    return summary

def display_performance_summary(summary):
    """Display performance summary"""
    print("\n" + "="*60)
    print("  Performance Summary")
    print("="*60)
    print(f"\nSample Users: {summary['n_users']}")
    print(f"\nAverage Response Time:")
    for method, time_ms in summary['avg_time_ms'].items():
        print(f"  {method:20s}: {time_ms:6.1f} ms")
    print("="*60)

def main():
    """Main execution pipeline"""
    print("="*60)
    print("  Multi-Modal Product Recommendation System")
    print("  Complete Pipeline Execution")
    print("="*60)
    
    # Check if data is prepared
    if not os.path.exists('data/processed/products.csv'):
        print("\n❌ Error: Processed data not found!")
        print("Please run the following steps first:")
        print("  1. python download_dataset.py")
        print("  2. python data_preprocessing.py")
        print("  3. python extract_features.py")
        return
    
    # Load data
    products, train_data, test_data, fused_features = load_all_data()
    
    # Initialize models
    models = initialize_models(products, train_data, fused_features)
    
    # Generate recommendations
    recommendations = generate_sample_recommendations(
        models, 
        train_data, 
        products,
        n_users=5
    )
    
    # Save results
    summary = save_results(recommendations)
    
    # Display summary
    display_performance_summary(summary)
    
    print("\n✅ Pipeline complete!")
    print(f"\n📁 Results saved to: results/recommendations/")
    print("\n▶️  Next steps:")
    print("   - Review recommendations in sample_recommendations.json")
    print("   - Run evaluation.py for detailed metrics")
    print("   - Explore visualizations in results/plots/")

if __name__ == "__main__":
    main()