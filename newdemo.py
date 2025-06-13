import streamlit as st
import pandas as pd
import numpy as np
from CfilterClaude4 import UserUserCF
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from mf2 import MF2
import random

st.set_page_config(page_title="Movie Recommendation System", layout="wide")
# --- Load Data ---
@st.cache_data
def load_data():
    # Load ratings
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('data/ml-100k/ub.base', sep='\t', names=r_cols)
    
    # Load movies with all columns for content-based filtering
    i_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
    movies = pd.read_csv('data/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
    
    return ratings, movies

@st.cache_data
def load_content_based_data():
    try:
        from contentBased1 import Yhat, items, user_means, genre_name_to_col, filter_indices_by_genre
        return Yhat, items, user_means, genre_name_to_col, filter_indices_by_genre
    except Exception as e:
        print("Error importing content-based data:", e)
        return None, None, None, None, None

@st.cache_data
def initialize_models():
    ratings_df, movies_df = load_data()
    
    # Prepare data for CF model
    rate_train = ratings_df.copy()
    rate_train[['user_id', 'movie_id']] -= 1
    cf_model = UserUserCF(rate_train.to_numpy(), k=30)
    cf_model.fit()
    
    return cf_model, ratings_df, movies_df

# Initialize models
cf_model, ratings_df, movies_df = initialize_models()
Yhat, content_items, user_means, genre_name_to_col, filter_indices_by_genre = load_content_based_data()

# --- Main Dashboard ---

st.title("🎬 Movie Recommendation System - Main Dashboard")

# Sidebar for main navigation
st.sidebar.title("Navigation")
main_option = st.sidebar.selectbox(
    "Choose Recommendation Method:",
    ["User-Based CF", "Content-Based", "Item-Based CF", "Matrix Factorization"]
)

# --- User-Based Collaborative Filtering ---
if main_option == "User-Based CF":
    st.header("🤝 User-Based Collaborative Filtering")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Settings")
        user_id = st.number_input("Enter User ID:", min_value=0, max_value=cf_model.n_users - 1, step=1, value=10)
        n_recs = st.slider("Number of recommendations:", 1, 20, 10)
        k_neighbors = st.slider("Number of neighbors (K):", 5, 50, 30)
        
        if st.button("Get Recommendations", key="user_cf"):
            # Update model with new K if different
            if k_neighbors != cf_model.k:
                cf_model.k = k_neighbors
                cf_model.fit()
            
            rec_ids = cf_model.recommend(user_id, n_recommendations=n_recs)
            st.session_state.user_cf_recs = rec_ids
    
    with col2:
        st.subheader("Recommendations")
        if 'user_cf_recs' in st.session_state:
            for idx, movie_id in enumerate(st.session_state.user_cf_recs):
                title = movies_df[movies_df['movie_id'] == movie_id + 1]['movie_title'].values
                if len(title) > 0:
                    st.write(f"**{idx+1}.** {title[0]}")
                else:
                    st.write(f"**{idx+1}.** [Movie ID {movie_id}]")
        else:
            st.info("Click 'Get Recommendations' to see results")
    
    # User Profile Section
    with st.expander("👤 View User Profile"):
        if user_id is not None:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id + 1]
            if not user_ratings.empty:
                st.write(f"**User {user_id + 1} has rated the following movies:**")
                for _, row in user_ratings.head(10).iterrows():
                    movie_title = movies_df[movies_df['movie_id'] == row['movie_id']]['movie_title'].values
                    if len(movie_title) > 0:
                        st.write(f"• {movie_title[0]} - Rating: {row['rating']}")
            else:
                st.write("No ratings found for this user.")

# --- Content-Based Filtering ---
elif main_option == "Content-Based":
    st.header("🎭 Content-Based Filtering")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Settings")
        
        # Genre selection
        
        
        user_id_content = st.number_input("Enter User ID for Content-Based:", min_value=0, max_value=942, step=1, value=10)
        n_content_recs = st.slider("Number of content recommendations:", 1, 20, 10)
        
        if st.button("Get Content-Based Recommendations", key="content_based"):
            if Yhat is not None:
                # Get predictions for the user
                user_pred_scores = Yhat[:, user_id_content]
                
                # Get user's rated movies
                user_ratings = ratings_df[ratings_df['user_id'] == user_id_content + 1]['movie_id'].tolist()
                rated_movies = [mid - 1 for mid in user_ratings]  # Convert to 0-based
                
                # Filter unrated movies
                unrated_indices = [i for i in range(len(user_pred_scores)) if i not in rated_movies]
                unrated_scores = [(i, user_pred_scores[i]) for i in unrated_indices]
                unrated_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Get top recommendations
                top_movie_indices = [i for i, score in unrated_scores[:n_content_recs]]
                st.session_state.content_recs = top_movie_indices[:n_content_recs]
            else:
                st.error("Content-based model not available. Please check contentBased1.py")
    
    with col2:
        st.subheader("Content-Based Recommendations")
        if 'content_recs' in st.session_state:
            for idx, movie_idx in enumerate(st.session_state.content_recs):
                movie_info = movies_df[movies_df['movie_id'] == movie_idx + 1]
                if not movie_info.empty:
                    title = movie_info['movie_title'].values[0]
                    st.write(f"**{idx+1}.** {title}")
                else:
                    st.write(f"**{idx+1}.** [Movie Index {movie_idx}]")
        else:
            st.info("Configure settings and click 'Get Content-Based Recommendations'")

# --- Item-Based Collaborative Filtering ---
elif main_option == "Item-Based CF":
    st.header("🎬 Item-Based Collaborative Filtering")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Movie Similarity")
        
        # Movie dropdown selection
        movie_titles = movies_df['movie_title'].tolist()
        selected_movie = st.selectbox("Select a movie:", movie_titles, key="item_cf_movie")
        
        
        n_similar = st.slider("Number of similar movies:", 1, 20, 10)
        
        if st.button("Find Similar Movies", key="item_cf"):
            try:
                # Create movie-user matrix
                movie_user_matrix = ratings_df.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
                
                # Calculate similarity
                similarity_matrix = cosine_similarity(movie_user_matrix)
                
                # Find selected movie ID
                selected_movie_id = movies_df[movies_df['movie_title'] == selected_movie]['movie_id'].values[0]
                
                if selected_movie_id in movie_user_matrix.index:
                    movie_idx = list(movie_user_matrix.index).index(selected_movie_id)
                    sim_scores = list(enumerate(similarity_matrix[movie_idx]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                    
                    # Get top similar movies (excluding the movie itself)
                    similar_indices = [i for i, _ in sim_scores[1:n_similar+1]]
                    similar_movie_ids = [movie_user_matrix.index[i] for i in similar_indices]
                    
                    st.session_state.similar_movies = similar_movie_ids
                    st.session_state.selected_movie_name = selected_movie
                else:
                    st.error("Selected movie not found in rating matrix")
                    
            except Exception as e:
                st.error(f"Error calculating similarity: {e}")
    
    with col2:
        st.subheader("Similar Movies")
        if 'similar_movies' in st.session_state:
            st.write(f"**Movies similar to '{st.session_state.selected_movie_name}':**")
            for idx, movie_id in enumerate(st.session_state.similar_movies):
                movie_title = movies_df[movies_df['movie_id'] == movie_id]['movie_title'].values
                if len(movie_title) > 0:
                    st.write(f"**{idx+1}.** {movie_title[0]}")
        else:
            st.info("Select a movie and click 'Find Similar Movies'")
    
    # Item Neighborhoods
    with st.expander("🏘️ Item Neighborhoods"):
        st.write("This section shows the neighborhood structure of items based on collaborative filtering patterns.")
        if st.button("Analyze Item Neighborhoods"):
            st.info("Item neighborhood analysis would be implemented here using the P2.py algorithms")

# --- Matrix Factorization ---
elif main_option == "Matrix Factorization":
    st.header("🔢 Matrix Factorization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("New User Setup")
        st.write("Rate at least 5 movies to get personalized recommendations:")
        
        # Movie selection for rating
        movie_choices = movies_df['movie_title'].tolist()
        selected_movies_mf = st.multiselect("Select movies you've watched:", movie_choices, key="mf_movies")
        
        # Rating interface
        user_ratings = {}
        if selected_movies_mf:
            st.write("Rate the selected movies:")
            for movie in selected_movies_mf:
                rating = st.slider(f"Rate '{movie}':", 1.0, 5.0, 3.0, 0.5, key=f"rating_{movie}")
                user_ratings[movie] = rating
        
        if st.button("Get MF Recommendations", key="matrix_fact"):
            if len(selected_movies_mf) >= 5:
                try:
                    # Create movie title to ID mapping
                    title_to_id = dict(zip(movies_df['movie_title'], movies_df['movie_id']))
                    
                    # Prepare new user ratings
                    new_user_ratings = []
                    for movie_title, rating in user_ratings.items():
                        movie_id = title_to_id[movie_title]
                        new_user_ratings.append([movie_id, rating])
                    
                    # Load MF model and get recommendations
                    try:
                        mf_model = MF2.load("model/mf_model")
                        mf_model.add_new_user(new_user_ratings)
                        
                        predictions = mf_model.pred_for_user(mf_model.n_users - 1)
                        top_k_recs = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]
                        
                        # Filter out already rated movies
                        rated_ids = [title_to_id[movie] for movie in selected_movies_mf]
                        filtered_recs = [(mid, rating) for mid, rating in top_k_recs if mid not in rated_ids]
                        
                        st.session_state.mf_recs = filtered_recs[:10]
                        
                    except Exception as e:
                        st.error(f"Error loading MF model: {e}")
                        st.info("Make sure the MF model is trained and saved in the 'model' directory")
                        
                except Exception as e:
                    st.error(f"Error processing ratings: {e}")
            else:
                st.warning("Please select and rate at least 5 movies")
    
    with col2:
        st.subheader("Matrix Factorization Recommendations")
        if 'mf_recs' in st.session_state:
            st.write("**Recommended movies based on your ratings:**")
            movie_id_to_title = dict(zip(movies_df['movie_id'], movies_df['movie_title']))
            
            for idx, (movie_id, pred_rating) in enumerate(st.session_state.mf_recs):
                title = movie_id_to_title.get(movie_id, f"Movie ID {movie_id}")
                rating = min(5, max(1, round(pred_rating, 1)))
                st.write(f"**{idx+1}.** {title} — ⭐ {rating:.1f}")
        else:
            st.info("Rate some movies to get personalized recommendations")



# --- Footer ---
st.markdown("---")
st.markdown("**Movie Recommendation System** - Powered by Multiple ML Algorithms")
