import streamlit as st
import pandas as pd
import numpy as np
from CfilterClaude4 import UserUserCF
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# --- Load Data ---
@st.cache_data
def load_data():
    # Load ratings. We use the `ub.base` for training the CF model and for similarity calculation.
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('data/ml-100k/ub.base', sep='\t', names=r_cols)

    # Load movies
    movie_cols = ['movie_id', 'title']
    movies = pd.read_csv('data/ml-100k/u.item', sep='|', encoding='latin-1',
                         usecols=[0, 1], names=movie_cols, header=None)
    return ratings, movies

ratings_df, movies_df = load_data()

# Prepare data for CF model (0-indexed numpy array)
rate_train = ratings_df.copy()
rate_train[['user_id', 'movie_id']] -= 1
cf_model = UserUserCF(rate_train.to_numpy(), k=30)
cf_model.fit()

# --- Streamlit UI ---
st.title("🎬 Movie Recommender System")

tab1, tab2, tab3 = st.tabs([
    "🔎 Recommend for User", 
    "🎞️ Find Similar Movies", 
    "🎯 Recommend by Content"
])

with tab1:
    st.header("📌 Get Movie Recommendations")
    user_id = st.number_input("Enter user ID:", min_value=0, max_value=cf_model.n_users - 1, step=1)
    n_recs = st.slider("Number of recommendations:", 1, 20, 10)

    if st.button("Get Recommendations"):
        rec_ids = cf_model.recommend(user_id, n_recommendations=n_recs)
        st.subheader("🎯 Recommended Movies:")
        for idx, movie_id in enumerate(rec_ids):
            title = movies_df[movies_df['movie_id'] == movie_id + 1]['title'].values
            if len(title) > 0:
                st.write(f"{idx+1}. {title[0]}")
            else:
                st.write(f"{idx+1}. [Movie ID {movie_id}]")

with tab2:
    st.header("🎞️ Find Movies Similar to a Given Title")

    movie_name = st.text_input("Enter a movie title", placeholder="e.g. Toy Story (1995)")

    if st.button("Find Similar Movies"):
        try:
            # Use the global ratings_df
            movie_user_matrix = ratings_df.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
            similarity_matrix = cosine_similarity(movie_user_matrix)

            # Create mappers between movie_id and matrix index
            id_to_idx = {mid: i for i, mid in enumerate(movie_user_matrix.index)}
            idx_to_id = {i: mid for i, mid in enumerate(movie_user_matrix.index)}

            # Find movie in the global movies_df
            exact_match = movies_df[movies_df['title'].str.lower().str.strip() == movie_name.lower().strip()]

            if exact_match.empty:
                all_titles = movies_df['title'].tolist()
                close_matches = get_close_matches(movie_name, all_titles, n=5, cutoff=0.5)
                if close_matches:
                    st.warning("No exact match found. Did you mean:")
                    for title in close_matches:
                        st.write(f"👉 {title}")
                else:
                    st.error("❌ No movie found with that name.")
            else:
                movie_id = exact_match.iloc[0]['movie_id']
                if movie_id not in id_to_idx:
                    st.error(f"Sorry, no ratings found for '{movie_name}' to calculate similarity.")
                else:
                    matrix_idx = id_to_idx[movie_id]
                    sim_scores = list(enumerate(similarity_matrix[matrix_idx]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

                    # Get top 10 similar movie indices (excluding the movie itself)
                    top_similar_indices = [i for i, _ in sim_scores[1:11]]
                    
                    # Convert matrix indices back to movie_ids
                    top_similar_ids = [idx_to_id[i] for i in top_similar_indices]
                    
                    # Get titles
                    similar_titles = movies_df[movies_df['movie_id'].isin(top_similar_ids)]['title'].tolist()

                    st.success(f"Movies similar to **{exact_match.iloc[0]['title']}**:")
                    for i, title in enumerate(similar_titles, 1):
                        st.write(f"{i}. {title}")
        except Exception as e:
            st.error(f"🚨 Error occurred: {e}")

with tab3:
    st.header("🎯 Content-Based Movie Recommendations")

    @st.cache_data
    def load_cbf_model():
        # Load data
        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        ratings_base = pd.read_csv('data/ml-100k/ua.base', sep='\t', names=r_cols)
        ratings_test = pd.read_csv('data/ml-100k/ua.test', sep='\t', names=r_cols)
        ratings = pd.concat([ratings_base, ratings_test], ignore_index=True)

        i_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action',
                  'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        items = pd.read_csv('data/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

        # Convert to 0-based indexing for consistency
        ratings['user_id'] -= 1
        ratings['movie_id'] -= 1
        items['movie_id'] -= 1

        # TF-IDF from genres
        genre_matrix = items.iloc[:, -19:].values
        from sklearn.feature_extraction.text import TfidfTransformer
        tfidf = TfidfTransformer().fit_transform(genre_matrix).toarray()

        n_users = ratings['user_id'].nunique()
        from sklearn.linear_model import Ridge
        d = tfidf.shape[1]
        W = np.zeros((d, n_users))
        b = np.zeros((1, n_users))

        for n in range(n_users):
            user_ratings = ratings[ratings['user_id'] == n]
            ids = user_ratings['movie_id'].values
            scores = user_ratings['rating'].values
            if len(ids) == 0:
                continue
            clf = Ridge(alpha=0.01)
            clf.fit(tfidf[ids], scores)
            W[:, n] = clf.coef_
            b[0, n] = clf.intercept_

        Yhat = tfidf @ W + b
        return Yhat, items, ratings

    Yhat, items_df, ratings_df = load_cbf_model()
    n_users_cbf = Yhat.shape[1]

    user_id_cbf = st.number_input(
        "Enter user ID (for CBF):", 
        min_value=0, 
        max_value=n_users_cbf - 1, 
        step=1, 
        key="cbf_user"
    )
    n_recs_cbf = st.slider("Number of content-based recommendations:", 1, 20, 10, key="cbf_slider")

    def recommend_movies_cbf(user_id, Yhat, items, ratings, top_n=10):
        rated_movies = ratings[ratings['user_id'] == user_id]['movie_id'].tolist()
        user_pred_scores = Yhat[:, user_id]
        unseen_ids = [i for i in range(len(user_pred_scores)) if i not in rated_movies]
        unseen_scores = [(i, user_pred_scores[i]) for i in unseen_ids]
        unseen_scores.sort(key=lambda x: x[1], reverse=True)
        top_movies_idx = [i for i, score in unseen_scores[:top_n]]
        
        # Use .iloc to select by 0-based index
        recommended = items.iloc[top_movies_idx][['movie_id', 'title']]
        return recommended

    if st.button("Recommend (CBF)"):
        recs = recommend_movies_cbf(user_id_cbf, Yhat, items_df, ratings_df, top_n=n_recs_cbf)
        st.subheader("📚 Content-Based Recommendations:")
        for i, row in recs.iterrows():
            st.write(f"{row['title']}")
