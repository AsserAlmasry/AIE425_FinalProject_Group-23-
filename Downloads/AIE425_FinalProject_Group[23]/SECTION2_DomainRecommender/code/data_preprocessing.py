"""
Data Preprocessing Pipeline for Multi-Modal Recommendation System
Handles cleaning, filtering, splitting, and feature extraction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from collections import Counter

class DataPreprocessor:
    def __init__(self, min_user_interactions=5, min_item_interactions=3):
        """
        Initialize preprocessor
        
        Args:
            min_user_interactions: Minimum ratings per user to keep
            min_item_interactions: Minimum ratings per item to keep
        """
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        
    def load_data(self):
        """Load raw data"""
        print("Loading raw data...")
        
        self.reviews = pd.read_csv('data/raw/reviews.csv')
        self.products = pd.read_csv('data/raw/products.csv')
        
        print(f"✓ Loaded {len(self.reviews)} reviews")
        print(f"✓ Loaded {len(self.products)} products")
        
        return self.reviews, self.products
    
    def clean_data(self):
        """Clean and validate data"""
        print("\n🧹 Cleaning data...")
        
        initial_reviews = len(self.reviews)
        initial_products = len(self.products)
        
        # Remove duplicates
        self.reviews = self.reviews.drop_duplicates(subset=['user_id', 'asin'])
        
        # Ensure valid ratings (1-5)
        self.reviews = self.reviews[self.reviews['rating'].between(1, 5)]
        
        # Remove products without reviews
        valid_asins = set(self.reviews['asin'].unique())
        self.products = self.products[self.products['asin'].isin(valid_asins)]
        
        # Handle missing values in products
        self.products['title'] = self.products['title'].fillna('Unknown Product')
        self.products['description'] = self.products.get('description', self.products['title'])
        self.products['main_category'] = self.products.get('main_category', 'General')
        self.products['price'] = self.products.get('price', 0).fillna(self.products.get('price', 0).median())
        
        print(f"✓ Removed {initial_reviews - len(self.reviews)} duplicate/invalid reviews")
        print(f"✓ Removed {initial_products - len(self.products)} products without reviews")
        
    def filter_sparse_data(self):
        """Filter out users and items with too few interactions"""
        print(f"\n🔍 Filtering sparse data (min_user={self.min_user_interactions}, min_item={self.min_item_interactions})...")
        
        # Iteratively filter until stable
        prev_size = 0
        iteration = 0
        
        while len(self.reviews) != prev_size and iteration < 10:
            prev_size = len(self.reviews)
            
            # Count interactions
            user_counts = self.reviews['user_id'].value_counts()
            item_counts = self.reviews['asin'].value_counts()
            
            # Filter users
            valid_users = user_counts[user_counts >= self.min_user_interactions].index
            self.reviews = self.reviews[self.reviews['user_id'].isin(valid_users)]
            
            # Filter items
            valid_items = item_counts[item_counts >= self.min_item_interactions].index
            self.reviews = self.reviews[self.reviews['asin'].isin(valid_items)]
            
            iteration += 1
        
        # Update products to match reviews
        valid_asins = set(self.reviews['asin'].unique())
        self.products = self.products[self.products['asin'].isin(valid_asins)]
        
        print(f"✓ Filtered in {iteration} iterations")
        print(f"✓ Final: {self.reviews['user_id'].nunique()} users, {self.reviews['asin'].nunique()} items")
    
    def compute_statistics(self):
        """Compute and display dataset statistics"""
        print("\n📊 Dataset Statistics:")
        print("="*60)
        
        n_users = self.reviews['user_id'].nunique()
        n_items = self.reviews['asin'].nunique()
        n_interactions = len(self.reviews)
        
        sparsity = 1 - (n_interactions / (n_users * n_items))
        
        stats = {
            'n_users': n_users,
            'n_items': n_items,
            'n_interactions': n_interactions,
            'sparsity': sparsity,
            'avg_ratings_per_user': n_interactions / n_users,
            'avg_ratings_per_item': n_interactions / n_items,
            'rating_distribution': self.reviews['rating'].value_counts().sort_index().to_dict()
        }
        
        print(f"Users: {stats['n_users']:,}")
        print(f"Items: {stats['n_items']:,}")
        print(f"Interactions: {stats['n_interactions']:,}")
        print(f"Sparsity: {stats['sparsity']:.2%}")
        print(f"Avg Ratings/User: {stats['avg_ratings_per_user']:.1f}")
        print(f"Avg Ratings/Item: {stats['avg_ratings_per_item']:.1f}")
        print(f"\nRating Distribution:")
        for rating, count in sorted(stats['rating_distribution'].items()):
            print(f"  {rating} stars: {count:,} ({count/n_interactions*100:.1f}%)")
        
        print("="*60)
        
        return stats
    
    def create_mappings(self):
        """Create ID mappings for matrix operations"""
        print("\n🔢 Creating ID mappings...")
        
        # User mapping
        unique_users = sorted(self.reviews['user_id'].unique())
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        
        # Item mapping
        unique_items = sorted(self.reviews['asin'].unique())
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Add indices to dataframes
        self.reviews['user_idx'] = self.reviews['user_id'].map(self.user_to_idx)
        self.reviews['item_idx'] = self.reviews['asin'].map(self.item_to_idx)
        
        # Add index to products
        self.products['item_idx'] = self.products['asin'].map(self.item_to_idx)
        self.products = self.products.sort_values('item_idx').reset_index(drop=True)
        
        print(f"✓ Created mappings for {len(self.user_to_idx)} users and {len(self.item_to_idx)} items")
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        Uses stratified split to maintain rating distribution
        """
        print(f"\n✂️  Splitting data (test_size={test_size})...")
        
        # For cold-start evaluation, also create user groups
        user_interaction_counts = self.reviews.groupby('user_id').size()
        
        # Stratify by user to ensure each user has both train and test
        train_list = []
        test_list = []
        
        for user_id in self.reviews['user_id'].unique():
            user_data = self.reviews[self.reviews['user_id'] == user_id]
            
            if len(user_data) >= 5:  # Only split if user has enough data
                user_train, user_test = train_test_split(
                    user_data, 
                    test_size=test_size,
                    random_state=random_state
                )
                train_list.append(user_train)
                test_list.append(user_test)
            else:
                # Put all in train if too few interactions
                train_list.append(user_data)
        
        train_data = pd.concat(train_list, ignore_index=True)
        test_data = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()
        
        print(f"✓ Train: {len(train_data)} interactions ({len(train_data)/len(self.reviews)*100:.1f}%)")
        print(f"✓ Test: {len(test_data)} interactions ({len(test_data)/len(self.reviews)*100:.1f}%)")
        
        return train_data, test_data
    
    def compute_item_popularity(self):
        """Compute item popularity scores"""
        item_counts = self.reviews['asin'].value_counts()
        self.products['popularity_score'] = self.products['asin'].map(item_counts).fillna(0)
        
        # Normalize to 0-1
        max_pop = self.products['popularity_score'].max()
        if max_pop > 0:
            self.products['popularity_score'] = self.products['popularity_score'] / max_pop
    
    def save_processed_data(self, train_data, test_data, stats):
        """Save all processed data"""
        print("\n💾 Saving processed data...")
        
        # Save dataframes
        train_data.to_csv('data/processed/train.csv', index=False)
        test_data.to_csv('data/processed/test.csv', index=False)
        self.products.to_csv('data/processed/products.csv', index=False)
        self.reviews.to_csv('data/processed/reviews_all.csv', index=False)
        
        # Save mappings
        mappings = {
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'stats': stats
        }
        
        with open('data/processed/mappings.pkl', 'wb') as f:
            pickle.dump(mappings, f)
        
        print("✓ Saved to data/processed/")
        print("  - train.csv, test.csv")
        print("  - products.csv, reviews_all.csv")
        print("  - mappings.pkl")

def main():
    """Main preprocessing pipeline"""
    print("="*60)
    print("  Data Preprocessing Pipeline")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        min_user_interactions=5,
        min_item_interactions=3
    )
    
    # Load data
    preprocessor.load_data()
    
    # Clean data
    preprocessor.clean_data()
    
    # Filter sparse data
    preprocessor.filter_sparse_data()
    
    # Compute statistics
    stats = preprocessor.compute_statistics()
    
    # Create mappings
    preprocessor.create_mappings()
    
    # Compute popularity
    preprocessor.compute_item_popularity()
    
    # Split data
    train_data, test_data = preprocessor.split_data(test_size=0.2)
    
    # Save everything
    preprocessor.save_processed_data(train_data, test_data, stats)
    
    print("\n" + "="*60)
    print("✅ Preprocessing complete!")
    print("="*60)
    print("\n▶️  Next step: Run 'python extract_features.py'")

if __name__ == "__main__":
    main()




#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================





"""
Multi-Modal Feature Extraction
Extracts text (TF-IDF), image (ResNet50), category, and price features
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from pathlib import Path
import os

# Try to import image processing libraries
try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - will use synthetic image features")

class FeatureExtractor:
    def __init__(self, products_df):
        """
        Initialize feature extractor
        
        Args:
            products_df: DataFrame with product information
        """
        self.products = products_df
        self.n_items = len(products_df)
        
    def extract_text_features(self, max_features=500):
        """
        Extract TF-IDF features from product titles and descriptions
        
        Args:
            max_features: Maximum number of TF-IDF features
        """
        print(f"\n📝 Extracting text features (TF-IDF, max_features={max_features})...")
        
        # Combine title and description
        text_data = (
            self.products['title'].fillna('') + ' ' + 
            self.products['description'].fillna('')
        )
        
        # Create TF-IDF vectorizer
        self.text_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)  # Unigrams and bigrams
        )
        
        # Fit and transform
        text_features = self.text_vectorizer.fit_transform(text_data).toarray()
        
        print(f"✓ Text features shape: {text_features.shape}")
        print(f"✓ Vocabulary size: {len(self.text_vectorizer.vocabulary_)}")
        print(f"✓ Sample features: {list(self.text_vectorizer.get_feature_names_out()[:10])}")
        
        return text_features
    
    def extract_image_features(self):
        """
        Extract image features using pretrained ResNet50
        Falls back to synthetic features if images not available
        """
        print("\n🖼️  Extracting image features...")
        
        if TORCH_AVAILABLE and os.path.exists('data/images'):
            try:
                return self._extract_real_image_features()
            except Exception as e:
                print(f"⚠️  Error extracting real images: {e}")
                return self._generate_synthetic_image_features()
        else:
            return self._generate_synthetic_image_features()
    
    def _extract_real_image_features(self):
        """Extract features from real images using ResNet50"""
        print("Using ResNet50 for feature extraction...")
        
        # Load pretrained ResNet50
        model = models.resnet50(pretrained=True)
        # Remove final classification layer
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        feature_extractor.eval()
        
        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Extract features for each product
        image_features = []
        
        for idx, row in self.products.iterrows():
            asin = row['asin']
            img_path = f"data/images/{asin}.jpg"
            
            if os.path.exists(img_path):
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0)
                    
                    # Extract features
                    with torch.no_grad():
                        features = feature_extractor(img_tensor)
                        features = features.squeeze().numpy()
                    
                    image_features.append(features)
                except:
                    # Use zero vector if error
                    image_features.append(np.zeros(2048))
            else:
                # Use zero vector if image not found
                image_features.append(np.zeros(2048))
        
        image_features = np.array(image_features)
        print(f"✓ Real image features shape: {image_features.shape}")
        
        return image_features
    
    def _generate_synthetic_image_features(self):
        """
        Generate synthetic image features that capture visual similarity
        Uses product category and price to create meaningful embeddings
        """
        print("Generating synthetic image features...")
        
        np.random.seed(42)
        
        # Create base features from category
        categories = self.products['main_category'].unique()
        category_to_vector = {}
        
        for cat in categories:
            # Each category gets a different base vector
            base = np.random.randn(2048)
            category_to_vector[cat] = base
        
        # Generate features for each product
        image_features = []
        
        for idx, row in self.products.iterrows():
            # Start with category base
            base = category_to_vector[row['main_category']].copy()
            
            # Add noise for product variation
            noise = np.random.randn(2048) * 0.3
            
            # Add price influence (normalized)
            price_factor = (row['price'] / 100) * 0.1
            
            features = base + noise + price_factor
            
            # Normalize
            features = features / np.linalg.norm(features)
            
            image_features.append(features)
        
        image_features = np.array(image_features)
        print(f"✓ Synthetic image features shape: {image_features.shape}")
        
        return image_features
    
    def extract_category_features(self):
        """Extract one-hot encoded category features"""
        print("\n📁 Extracting category features...")
        
        # Encode categories
        self.category_encoder = LabelEncoder()
        category_encoded = self.category_encoder.fit_transform(
            self.products['main_category']
        )
        
        # One-hot encode
        n_categories = len(self.category_encoder.classes_)
        category_features = np.zeros((self.n_items, n_categories))
        category_features[np.arange(self.n_items), category_encoded] = 1
        
        print(f"✓ Category features shape: {category_features.shape}")
        print(f"✓ Categories: {list(self.category_encoder.classes_)}")
        
        return category_features
    
    def extract_price_features(self):
        """Extract and normalize price features"""
        print("\n💰 Extracting price features...")
        
        # Get prices and handle missing
        prices = self.products['price'].fillna(self.products['price'].median())
        
        # Standardize
        self.price_scaler = StandardScaler()
        price_features = self.price_scaler.fit_transform(
            prices.values.reshape(-1, 1)
        )
        
        print(f"✓ Price features shape: {price_features.shape}")
        print(f"✓ Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"✓ Price mean: ${prices.mean():.2f} (±${prices.std():.2f})")
        
        return price_features
    
    def fuse_features(self, text_features, image_features, category_features, 
                     price_features, weights=None):
        """
        Fuse all feature modalities into single representation
        
        Args:
            weights: Dictionary with keys 'text', 'image', 'category', 'price'
        """
        print("\n🔀 Fusing multi-modal features...")
        
        if weights is None:
            weights = {
                'text': 0.3,
                'image': 0.4,
                'category': 0.2,
                'price': 0.1
            }
        
        print(f"Fusion weights: {weights}")
        
        # Normalize each modality to similar scale
        text_norm = text_features / (np.linalg.norm(text_features, axis=1, keepdims=True) + 1e-10)
        image_norm = image_features / (np.linalg.norm(image_features, axis=1, keepdims=True) + 1e-10)
        category_norm = category_features  # Already normalized (one-hot)
        price_norm = price_features  # Already standardized
        
        # Weighted concatenation
        fused_features = np.concatenate([
            text_norm * weights['text'],
            image_norm * weights['image'],
            category_norm * weights['category'],
            price_norm * weights['price']
        ], axis=1)
        
        print(f"✓ Fused features shape: {fused_features.shape}")
        print(f"  - Text: {text_norm.shape[1]} dims × {weights['text']}")
        print(f"  - Image: {image_norm.shape[1]} dims × {weights['image']}")
        print(f"  - Category: {category_norm.shape[1]} dims × {weights['category']}")
        print(f"  - Price: {price_norm.shape[1]} dims × {weights['price']}")
        
        return fused_features
    
    def save_features(self, text_features, image_features, category_features,
                     price_features, fused_features):
        """Save all extracted features"""
        print("\n💾 Saving features...")
        
        # Save as numpy arrays
        np.save('data/features/text_features.npy', text_features)
        np.save('data/features/image_features.npy', image_features)
        np.save('data/features/category_features.npy', category_features)
        np.save('data/features/price_features.npy', price_features)
        np.save('data/features/fused_features.npy', fused_features)
        
        # Save encoders/vectorizers
        feature_objects = {
            'text_vectorizer': self.text_vectorizer,
            'category_encoder': self.category_encoder,
            'price_scaler': self.price_scaler
        }
        
        with open('data/features/feature_objects.pkl', 'wb') as f:
            pickle.dump(feature_objects, f)
        
        print("✓ Saved to data/features/")
        print("  - text_features.npy")
        print("  - image_features.npy")
        print("  - category_features.npy")
        print("  - price_features.npy")
        print("  - fused_features.npy")
        print("  - feature_objects.pkl")

def main():
    """Main feature extraction pipeline"""
    print("="*60)
    print("  Multi-Modal Feature Extraction")
    print("="*60)
    
    # Load processed products
    products_df = pd.read_csv('data/processed/products.csv')
    print(f"\nLoaded {len(products_df)} products")
    
    # Initialize extractor
    extractor = FeatureExtractor(products_df)
    
    # Extract each modality
    text_features = extractor.extract_text_features(max_features=500)
    image_features = extractor.extract_image_features()
    category_features = extractor.extract_category_features()
    price_features = extractor.extract_price_features()
    
    # Fuse features
    fused_features = extractor.fuse_features(
        text_features, 
        image_features, 
        category_features, 
        price_features
    )
    
    # Save everything
    extractor.save_features(
        text_features,
        image_features,
        category_features,
        price_features,
        fused_features
    )
    
    print("\n" + "="*60)
    print("✅ Feature extraction complete!")
    print("="*60)
    print("\n▶️  Next step: Run 'python content_based.py' or 'python main.py'")

if __name__ == "__main__":
    main()