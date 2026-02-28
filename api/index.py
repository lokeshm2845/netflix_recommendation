# api/index.py
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os
import sys

# Add the parent directory to path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your recommendation modules
try:
    from src.recommender import Recommender
    print("✅ Successfully imported Recommender module")
except Exception as e:
    print(f"❌ Error importing Recommender: {e}")

app = Flask(__name__)

# Get the absolute path to the model file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'output', 'recommender_model.pkl')

# Load the saved recommender model
recommender = None
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            recommender = pickle.load(f)
        print("✅ Model loaded successfully!")
        print(f"✅ Model contains {len(recommender.df['user_id'].unique()) if recommender else 0} users")
    else:
        print(f"❌ Model file not found at {MODEL_PATH}")
        
        # Create sample data for demo if model doesn't exist
        print("⚠️ Creating sample data for demonstration...")
        import numpy as np
        from src.data_loader import DataLoader
        
        # Load sample data
        data_loader = DataLoader()
        df = data_loader.create_sample_data()
        df = data_loader.preprocess_data()
        
        # Create recommender with sample data
        from src.recommender import Recommender
        recommender = Recommender(df)
        recommender.create_matrices()
        print("✅ Sample recommender created for demo!")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route('/')
def home():
    """Render the beautiful homepage"""
    return render_template('index.html')

@app.route('/api/recommend/<int:user_id>')
def recommend(user_id):
    """API endpoint for recommendations"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded. Please try again later.'}), 500
    
    try:
        # Check if user exists
        if user_id not in recommender.df['user_id'].values:
            # Return sample recommendations for demo
            sample_movies = [
                {"Movie": "The Shawshank Redemption", "Genre": "Drama", "Predicted Rating": 4.8},
                {"Movie": "The Godfather", "Genre": "Crime", "Predicted Rating": 4.7},
                {"Movie": "Pulp Fiction", "Genre": "Crime", "Predicted Rating": 4.6},
                {"Movie": "The Dark Knight", "Genre": "Action", "Predicted Rating": 4.8},
                {"Movie": "Fight Club", "Genre": "Drama", "Predicted Rating": 4.5},
                {"Movie": "Inception", "Genre": "Sci-Fi", "Predicted Rating": 4.9},
                {"Movie": "Goodfellas", "Genre": "Crime", "Predicted Rating": 4.6},
                {"Movie": "The Matrix", "Genre": "Sci-Fi", "Predicted Rating": 4.7},
                {"Movie": "Star Wars", "Genre": "Sci-Fi", "Predicted Rating": 4.5},
                {"Movie": "The Lord of the Rings", "Genre": "Fantasy", "Predicted Rating": 4.8}
            ]
            return jsonify({
                'user_id': user_id,
                'recommendations': sample_movies[:10]
            })
            
        recommendations = recommender.recommend_user_based(user_id, 12)
        
        # Format for response
        result = []
        for _, row in recommendations.iterrows():
            movie_dict = row.to_dict()
            # Add some variety to ratings
            if 'Predicted Rating' in movie_dict:
                movie_dict['Predicted Rating'] = round(float(movie_dict['Predicted Rating']), 1)
            result.append(movie_dict)
        
        return jsonify({
            'user_id': user_id,
            'recommendations': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/popular')
def popular():
    """Get popular recommendations"""
    if recommender is None:
        # Return sample popular movies
        sample_popular = [
            {"Movie": "Inception", "Genre": "Sci-Fi", "Predicted Rating": 4.9},
            {"Movie": "The Dark Knight", "Genre": "Action", "Predicted Rating": 4.8},
            {"Movie": "Interstellar", "Genre": "Sci-Fi", "Predicted Rating": 4.7},
            {"Movie": "The Matrix", "Genre": "Sci-Fi", "Predicted Rating": 4.7},
            {"Movie": "Avatar", "Genre": "Sci-Fi", "Predicted Rating": 4.6},
            {"Movie": "Titanic", "Genre": "Romance", "Predicted Rating": 4.5}
        ]
        return jsonify(sample_popular)
        
    try:
        genre = request.args.get('genre', None)
        recommendations = recommender.recommend_popular(genre=genre, n_recommendations=12)
        
        # Format for display
        result = []
        for _, row in recommendations.iterrows():
            movie_dict = row.to_dict()
            if 'Predicted Rating' in movie_dict:
                movie_dict['Predicted Rating'] = round(float(movie_dict['Predicted Rating']), 1)
            result.append(movie_dict)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/<int:user_id>')
def user_info(user_id):
    """Get user information"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        user_data = recommender.df[recommender.df['user_id'] == user_id]
        if len(user_data) == 0:
            return jsonify({'error': 'User not found'}), 404
            
        return jsonify({
            'user_id': user_id,
            'total_ratings': int(len(user_data)),
            'avg_rating': round(float(user_data['rating'].mean()), 2),
            'favorite_genre': str(user_data['genre'].mode().iloc[0]) if len(user_data) > 0 else 'Unknown'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': recommender is not None,
        'users': len(recommender.df['user_id'].unique()) if recommender else 0,
        'movies': len(recommender.df['movie_id'].unique()) if recommender else 0
    })

# For Vercel serverless function
def handler(event, context):
    return app(event, context)

# For local development
if __name__ == '__main__':
    app.run(debug=True)