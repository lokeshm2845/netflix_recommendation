"""
Utility functions for visualization and analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Analyzer:
    def __init__(self, df):
        self.df = df
        
    def genre_analysis(self):
        """Analyze genre popularity and ratings and return the data"""
        print("="*60)
        print("GENRE ANALYSIS")
        print("="*60)
        
        # Popular genres
        print("\n📊 MOST POPULAR GENRES:")
        genre_counts = self.df['genre'].value_counts()
        for genre, count in genre_counts.head(10).items():
            print(f"  {genre}: {count:,} ratings")
        
        # Best rated
        print("\n⭐ BEST RATED GENRES:")
        genre_avg = self.df.groupby('genre')['rating'].agg(['mean', 'count']).round(2)
        genre_avg = genre_avg[genre_avg['count'] > 100].sort_values('mean', ascending=False)
        for genre, row in genre_avg.head(10).iterrows():
            print(f"  {genre}: {row['mean']} ({row['count']:,} ratings)")
        
        # Worst rated
        print("\n📉 WORST RATED GENRES:")
        genre_worst = genre_avg.sort_values('mean').head(10)
        for genre, row in genre_worst.iterrows():
            print(f"  {genre}: {row['mean']} ({row['count']:,} ratings)")
        
        # Return the data for later use
        return genre_counts, genre_avg
    
    def plot_dashboard(self, save_path='output/dashboard.png'):
        """Create visualization dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Rating distribution
        axes[0, 0].hist(self.df['rating'], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Distribution of Ratings')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Genre popularity
        genre_counts = self.df['genre'].value_counts().head(15)
        axes[0, 1].bar(range(len(genre_counts)), genre_counts.values)
        axes[0, 1].set_title('Top 15 Most Popular Genres')
        axes[0, 1].set_xlabel('Genre')
        axes[0, 1].set_ylabel('Number of Ratings')
        axes[0, 1].set_xticks(range(len(genre_counts)))
        axes[0, 1].set_xticklabels(genre_counts.index, rotation=45, ha='right')
        
        # 3. Genre ratings
        genre_avg = self.df.groupby('genre')['rating'].mean().sort_values(ascending=False).head(15)
        axes[0, 2].barh(range(len(genre_avg)), genre_avg.values)
        axes[0, 2].set_title('Top 15 Best Rated Genres')
        axes[0, 2].set_xlabel('Average Rating')
        axes[0, 2].set_yticks(range(len(genre_avg)))
        axes[0, 2].set_yticklabels(genre_avg.index)
        
        # 4. User activity
        user_activity = self.df.groupby('user_id')['rating'].count()
        axes[1, 0].hist(user_activity, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('User Rating Activity')
        axes[1, 0].set_xlabel('Number of Ratings per User')
        axes[1, 0].set_ylabel('Number of Users')
        
        # 5. Movie popularity
        movie_pop = self.df.groupby('movie_name')['rating'].count().sort_values(ascending=False).head(20)
        axes[1, 1].bar(range(len(movie_pop)), movie_pop.values)
        axes[1, 1].set_title('Top 20 Most Rated Movies')
        axes[1, 1].set_xlabel('Movie')
        axes[1, 1].set_ylabel('Number of Ratings')
        axes[1, 1].set_xticks([])
        
        # 6. Rating correlation
        sample = self.df.sample(n=min(1000, len(self.df)))
        axes[1, 2].scatter(sample['rating'], 
                          sample['rating'] + np.random.normal(0, 0.1, len(sample)), 
                          alpha=0.5)
        axes[1, 2].set_title('Rating Distribution with Jitter')
        axes[1, 2].set_xlabel('Rating')
        axes[1, 2].set_ylabel('Jittered Rating')
        
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Dashboard saved to {save_path}")