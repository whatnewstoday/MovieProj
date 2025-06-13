import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Users ---
u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('data/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

# --- Load Ratings ---
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('data/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('data/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
rate_train = ratings_base.values
rate_test = ratings_test.values

# --- Load Items (Movies) ---
i_cols = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
          'Unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 
          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
          'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('data/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

# --- TF-IDF for genres (last 19 columns) ---
X_train_counts = items.iloc[:, -19:].values
transformer = TfidfTransformer(smooth_idf=True, norm='l2')
tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()

# --- Compute item-item similarity ---
item_sim = cosine_similarity(tfidf)

# --- Content-based prediction matrix Yhat ---
n_users = ratings_base['user_id'].max()
n_items = items.shape[0]
Yhat = np.zeros((n_items, n_users))

for user in range(n_users):
    user_ratings = rate_train[rate_train[:, 0] == user]
    rated_items = user_ratings[:, 1] - 1  # movie_id is 1-based
    ratings = user_ratings[:, 2]
    if len(rated_items) > 0:
        for i in range(n_items):
            sim_scores = item_sim[i, rated_items]
            if sim_scores.sum() > 0:
                Yhat[i, user] = np.dot(sim_scores, ratings) / sim_scores.sum()
            else:
                Yhat[i, user] = 0
    else:
        Yhat[:, user] = 0  # No ratings for this user

# --- Optionally, compute user means (can be None if not needed) ---
user_means = ratings_base.groupby('user_id')['rating'].mean().reindex(range(1, n_users+1)).values

# --- Make available for import ---
# Yhat: np.ndarray, items: pd.DataFrame, user_means: np.ndarray

# --- Finish Yhat computation ---
for user in range(n_users):
    user_ratings = rate_train[rate_train[:, 0] == user]
    rated_items = user_ratings[:, 1] - 1  # movie_id is 1-based
    ratings = user_ratings[:, 2]
    if len(rated_items) > 0:
        for i in range(n_items):
            sim_scores = item_sim[i, rated_items]
            if sim_scores.sum() > 0:
                Yhat[i, user] = np.dot(sim_scores, ratings) / sim_scores.sum()
            else:
                Yhat[i, user] = 0
    else:
        Yhat[:, user] = 0  # No ratings for this user

# --- Optionally, compute user means ---
user_means = ratings_base.groupby('user_id')['rating'].mean().reindex(range(1, n_users+1)).values

# --- Genre mapping for use in the app ---
genre_name_to_col = {
    'Unknown': 'Unknown',
    'Action': 'Action',
    'Adventure': 'Adventure',
    'Animation': 'Animation',
    "Children's": "Children's",
    'Comedy': 'Comedy',
    'Crime': 'Crime',
    'Documentary': 'Documentary',
    'Drama': 'Drama',
    'Fantasy': 'Fantasy',
    'Film-Noir': 'Film-Noir',
    'Horror': 'Horror',
    'Musical': 'Musical',
    'Mystery': 'Mystery',
    'Romance': 'Romance',
    'Sci-Fi': 'Sci-Fi',
    'Thriller': 'Thriller',
    'War': 'War',
    'Western': 'Western'
}

def filter_indices_by_genre(indices, selected_genres):
    """Filter a list of movie indices by selected genre names."""
    if not selected_genres:
        return indices
    selected_cols = [genre_name_to_col[g] for g in selected_genres]
    filtered = []
    for idx in indices:
        movie_row = items.iloc[idx]
        if any(movie_row[col] == 1 for col in selected_cols):
            filtered.append(idx)
    return filtered

# Yhat, items, user_means, genre_name_to_col, filter_indices_by_genre are now available for import