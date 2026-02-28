"""
Data loading and preprocessing module
"""
import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """Load dataset from CSV file"""
        if self.data_path and os.path.exists(self.data_path):
            print(f"Loading data from: {self.data_path}")
            self.df = pd.read_csv(self.data_path)
        else:
            print("No data file found. Creating sample dataset...")
            self.df = self.create_sample_data()
        
        print(f"Dataset shape: {self.df.shape}")
        return self.df
    
    def create_sample_data(self):
        """Create sample Netflix-like data for testing"""
        np.random.seed(42)
        n_users = 2000
        n_movies = 1000
        n_ratings = 100000
        
        # Generate IDs
        user_ids = np.random.randint(1001, 1001+n_users, n_ratings)
        movie_ids = np.random.randint(1, n_movies+1, n_ratings)
        ratings = np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.05, 0.1, 0.2, 0.3, 0.35])
        
        # Generate genres
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 
                  'Sci-Fi', 'Documentary', 'Thriller', 'Animation', 'Adventure']
        movie_genres = {i: np.random.choice(genres) for i in range(1, n_movies+1)}
        
        # Create DataFrame
        df = pd.DataFrame({
            'user_id': user_ids,
            'movie_id': movie_ids,
            'rating': ratings,
            'genre': [movie_genres[mid] for mid in movie_ids],
            'movie_name': [f"Movie_{mid}" for mid in movie_ids]
        })
        
        return df
    
    def preprocess_data(self):
        """Preprocess data for modeling"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Create numeric indices
        self.df['user_num'] = self.df['user_id'].astype('category').cat.codes
        self.df['movie_num'] = self.df['movie_id'].astype('category').cat.codes
        
        print(f"Unique users: {self.df['user_num'].nunique()}")
        print(f"Unique movies: {self.df['movie_num'].nunique()}")
        
        return self.df