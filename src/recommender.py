"""
Recommendation algorithms module
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, df):
        print("Initializing Recommender...")
        self.df = df
        self.user_movie_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        
    def create_matrices(self):
        """Create user-movie matrix and similarity matrices"""
        print("Creating matrices...")
        # Create pivot table
        self.user_movie_matrix = self.df.pivot_table(
            index='user_num', 
            columns='movie_num', 
            values='rating'
        )
        
        # Fill NaN with 0 for similarity
        matrix_filled = self.user_movie_matrix.fillna(0)
        
        # Calculate similarities
        self.user_similarity = cosine_similarity(matrix_filled)
        self.item_similarity = cosine_similarity(matrix_filled.T)
        
        print(f"User-movie matrix shape: {self.user_movie_matrix.shape}")
        return self.user_movie_matrix
    
    def recommend_user_based(self, user_id, n_recommendations=10):
        """User-based collaborative filtering"""
        print(f"Getting user-based recommendations for user {user_id}...")
        # Get user's numeric index
        user_num = self.df[self.df['user_id'] == user_id]['user_num'].iloc[0]
        
        # Find similar users
        user_sim_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index
        )
        similar_users = user_sim_df[user_num].sort_values(ascending=False)[1:11]
        
        # Get movies user has rated
        user_rated = self.df[self.df['user_num'] == user_num]['movie_num'].tolist()
        
        # Calculate scores
        movie_scores = {}
        similar_users_ratings = self.user_movie_matrix.loc[similar_users.index]
        
        for movie in self.user_movie_matrix.columns:
            if movie not in user_rated:
                movie_ratings = similar_users_ratings[movie].dropna()
                if len(movie_ratings) > 0:
                    weights = similar_users[movie_ratings.index]
                    weighted_score = np.average(movie_ratings, weights=weights)
                    movie_scores[movie] = weighted_score
        
        # Get top recommendations
        top_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        return self._format_recommendations(top_movies)
    
    def recommend_item_based(self, user_id, n_recommendations=10):
        """Item-based collaborative filtering"""
        print(f"Getting item-based recommendations for user {user_id}...")
        user_num = self.df[self.df['user_id'] == user_id]['user_num'].iloc[0]
        
        # Get user's ratings
        user_ratings = self.user_movie_matrix.loc[user_num].dropna()
        
        if len(user_ratings) == 0:
            return pd.DataFrame()
        
        # Calculate scores
        movie_scores = {}
        item_sim_df = pd.DataFrame(
            self.item_similarity,
            index=self.user_movie_matrix.columns,
            columns=self.user_movie_matrix.columns
        )
        
        for movie in self.user_movie_matrix.columns:
            if movie not in user_ratings.index:
                similar_movies = item_sim_df[movie].sort_values(ascending=False)
                rated_similar = [m for m in similar_movies.index if m in user_ratings.index][:10]
                
                if rated_similar:
                    weights = similar_movies[rated_similar]
                    ratings = user_ratings[rated_similar]
                    weighted_score = np.average(ratings, weights=weights)
                    movie_scores[movie] = weighted_score
        
        top_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        return self._format_recommendations(top_movies)
    
    def recommend_popular(self, genre=None, n_recommendations=10):
        """Popularity-based recommendations (for new users)"""
        print(f"Getting popular recommendations... Genre: {genre}")
        
        if genre:
            movie_stats = self.df[self.df['genre'] == genre].groupby(['movie_num', 'movie_name', 'genre']).agg({
                'rating': ['count', 'mean']
            }).round(2)
        else:
            movie_stats = self.df.groupby(['movie_num', 'movie_name', 'genre']).agg({
                'rating': ['count', 'mean']
            }).round(2)
        
        # Fix column names
        movie_stats.columns = ['rating_count', 'avg_rating']
        movie_stats = movie_stats.reset_index()
        movie_stats = movie_stats[movie_stats['rating_count'] > 10]
        
        # Convert to float to avoid string issues
        movie_stats['rating_count'] = pd.to_numeric(movie_stats['rating_count'], errors='coerce')
        movie_stats['avg_rating'] = pd.to_numeric(movie_stats['avg_rating'], errors='coerce')
        
        print(f"Movie stats shape: {movie_stats.shape}")
        print(f"Rating count type: {movie_stats['rating_count'].dtype}")
        print(f"Avg rating type: {movie_stats['avg_rating'].dtype}")
        
        # Popularity score
        movie_stats['popularity'] = (
            (movie_stats['rating_count'] / movie_stats['rating_count'].max()) * 0.3 +
            (movie_stats['avg_rating'] / 5) * 0.7
        ) * 100
        
        top_movies = movie_stats.nlargest(n_recommendations, 'popularity')
        
        recommendations = []
        for _, row in top_movies.iterrows():
            score = float(row['popularity'] / 20)
            print(f"Movie: {row['movie_name']}, Score: {score}, Type: {type(score)}")
            recommendations.append((
                row['movie_num'],
                score
            ))
        
        return self._format_recommendations(recommendations)
    
    def _format_recommendations(self, top_movies):
        """Format recommendations with movie details"""
        print(f"Formatting {len(top_movies)} recommendations...")
        movie_details = self.df[['movie_num', 'movie_name', 'genre']].drop_duplicates('movie_num')
        movie_dict = movie_details.set_index('movie_num').to_dict('index')
        
        recommendations = []
        for movie_num, score in top_movies:
            movie_data = movie_dict.get(movie_num, {})
            
            # Debug print
            print(f"Movie_num: {movie_num}, Score: {score}, Score type: {type(score)}")
            
            # Ensure score is a number before rounding
            try:
                pred_rating = round(float(score), 2)
            except:
                print(f"WARNING: Could not convert score {score} to float")
                pred_rating = 0.0
                
            recommendations.append({
                'Movie': movie_data.get('movie_name', f'Movie_{movie_num}'),
                'Genre': movie_data.get('genre', 'Unknown'),
                'Predicted Rating': pred_rating
            })
        
        return pd.DataFrame(recommendations)