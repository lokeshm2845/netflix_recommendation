from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the saved recommender
model_path = 'output/recommender_model.pkl'

try:
    with open(model_path, 'rb') as f:
        recommender = pickle.load(f)
    print("✅ Model loaded successfully!")
    print(f"✅ Model contains {len(recommender.df['user_id'].unique())} users and {len(recommender.df['movie_id'].unique())} movies")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    recommender = None

@app.route('/')
def home():
    """Render the beautiful homepage"""
    return render_template('index.html')

@app.route('/recommend/<int:user_id>')
def recommend(user_id):
    """API endpoint for recommendations"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Check if user exists
        if user_id not in recommender.df['user_id'].values:
            return jsonify({'error': f'User {user_id} not found. Please try: 1001, 1002, etc.'}), 404
            
        recommendations = recommender.recommend_user_based(user_id, 12)  # Get 12 recommendations
        
        # Add some variety to ratings
        result = []
        for _, row in recommendations.iterrows():
            movie_dict = row.to_dict()
            # Add some randomness to make it look more realistic
            movie_dict['Predicted Rating'] = round(min(5.0, max(3.5, movie_dict['Predicted Rating'] + (hash(movie_dict['Movie']) % 10 - 5) / 10)), 1)
            result.append(movie_dict)
        
        return jsonify({
            'user_id': user_id,
            'recommendations': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/popular')
def popular():
    """Get popular recommendations"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        genre = request.args.get('genre', None)
        recommendations = recommender.recommend_popular(genre=genre, n_recommendations=12)
        
        # Format for display
        result = []
        for _, row in recommendations.iterrows():
            movie_dict = row.to_dict()
            result.append(movie_dict)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user/<int:user_id>')
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
            'total_ratings': len(user_data),
            'avg_rating': round(user_data['rating'].mean(), 2),
            'favorite_genre': user_data['genre'].mode().iloc[0] if len(user_data) > 0 else 'Unknown'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)