import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import os
import json  # Added for JSON output

# Configuration
DATA_DIR = "ml-100k"
RATING_FILE = "Rating.csv"
MOVIE_FILE = os.path.join(DATA_DIR, "u.item")
NUM_USERS = 943
NUM_ITEMS = 1682
K_NEIGHBORS = 30  # Number of nearest neighbors
TOP_N = 10  # Number of recommendations per user

# Implementation of Item-Item Collaborative Filtering using cosine similarity
class ItemItemCF:
    """Item-Item Collaborative Filtering"""
    def __init__(self, Y_data, k):
        self.Y_data = Y_data[:, [1, 0, 2]]  # Item-Item CF: [movie_id, user_id, rating]
        self.k = k
        self.Ybar_data = None
        self.n_users = int(np.max(self.Y_data[:, 1])) + 1
        self.n_items = int(np.max(self.Y_data[:, 0])) + 1
        self.S = None
        self.mu = None

    def normalize_Y(self):
        """Normalize rating matrix by subtracting user means."""
        users = self.Y_data[:, 1]
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros(self.n_users)
        for n in range(self.n_users):
            ids = np.where(users == n)[0].astype(np.int32)
            ratings = self.Ybar_data[ids, 2]
            self.mu[n] = np.mean(ratings) if ratings.size > 0 else 0
            self.Ybar_data[ids, 2] = ratings - self.mu[n]
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
                                       (self.Ybar_data[:, 0], self.Ybar_data[:, 1])),
                                      (self.n_items, self.n_users)).tocsr()

    def similarity(self):
        """Compute item-item cosine similarity."""
        self.S = cosine_similarity(self.Ybar.T)
        np.fill_diagonal(self.S, 0)
        np.savetxt('item_item_cossim.csv', self.S, delimiter=',', fmt='%10.6f')

    def fit(self):
        """Fit the model: normalize data and compute similarity."""
        self.normalize_Y()
        self.similarity()

    def pred(self, u, i, normalized=1):
        """Predict rating of user u for item i."""
        ids = np.where(self.Y_data[:, 1] == u)[0].astype(np.int32)
        items_rated_by_u = self.Y_data[ids, 0].astype(np.int32)
        if not items_rated_by_u.size:
            return 0
        sim = self.S[i, items_rated_by_u]
        top_k = np.argsort(sim)[-self.k:]
        nearest_s = sim[top_k]
        r = self.Ybar[items_rated_by_u[top_k], u].toarray().flatten()
        denom = np.abs(nearest_s).sum() + 1e-8
        pred_rating = (r * nearest_s).sum() / denom
        if not normalized:
            pred_rating += self.mu[u]
        return pred_rating

    def recommend(self, u, top_n=TOP_N):
        """Recommend top-N items for user u."""
        ids = np.where(self.Y_data[:, 1] == u)[0]
        items_rated_by_u = self.Y_data[ids, 0].tolist()
        ratings = []
        items = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.pred(u, i)
                items.append(i)
                ratings.append(rating)
        if len(ratings) == 0:
            return np.full(top_n, -1, dtype=int)
        top_indices = np.argsort(ratings)[::-1][:top_n]
        recommended_items = np.array(items)[top_indices]
        result = np.full(top_n, -1, dtype=int)
        result[:len(recommended_items)] = recommended_items
        return result

def load_data(rating_file, movie_file):
    """Load ratings and movie metadata."""
    ratings = pd.read_csv(rating_file, usecols=['user_id', 'movie_id', 'rating'])
    if ratings['user_id'].max() > NUM_USERS or ratings['movie_id'].max() > NUM_ITEMS:
        print(f"Error: Rating.csv contains invalid user_id (> {NUM_USERS}) or movie_id (> {NUM_ITEMS}).")
        exit()
    Y_data = ratings[['movie_id', 'user_id', 'rating']].values - [1, 1, 0]
    np.savetxt('item_item_R.csv', Y_data, delimiter=',', fmt='%d')
    try:
        movies = pd.read_csv(movie_file, sep='|', encoding='latin-1', header=None,
                             usecols=[0, 1], names=['movie_id', 'movie_name'])
        movie_id_to_name = dict(zip(movies['movie_id'] - 1, movies['movie_name']))
        name_to_movie_id = {v: k for k, v in movie_id_to_name.items()}
    except FileNotFoundError:
        print(f"Error: File '{movie_file}' not found. Please download MovieLens 100k.")
        exit()
    return Y_data, movie_id_to_name, name_to_movie_id

def split_train_test(Y_data):
    """Split data into train and test sets."""
    train = Y_data.copy()
    test = []
    for u in range(NUM_USERS):
        user_ratings = train[train[:, 1] == u]
        if user_ratings.size > 0:
            max_rating = user_ratings[:, 2].max()
            max_indices = np.where(user_ratings[:, 2] == max_rating)[0]
            test_idx = user_ratings[np.random.choice(max_indices)]
            test.append(test_idx)
            train = train[~((train[:, 1] == u) & (train[:, 0] == test_idx[0]) & (train[:, 2] == test_idx[2]))]
    test = np.array(test)
    np.savetxt('item_item_train.csv', train, delimiter=',', fmt='%d')
    np.savetxt('item_item_test.csv', test, delimiter=',', fmt='%d')
    return train, test

def compute_recall(recommendations, test):
    """Compute recall@N metric."""
    hits = 0
    for u in range(NUM_USERS):
        test_items = test[test[:, 1] == u, 0]
        if test_items.size > 0 and np.any(np.isin(recommendations[u], test_items)):
            hits += 1
    recall = hits / NUM_USERS
    np.savetxt('item_item_recall.csv', [recall], fmt='%f')
    return recall

# New function: Compute RMSE
def compute_rmse(cf_model, test):
    """Compute Root Mean Squared Error on test set."""
    se = 0  # Squared error
    n_tests = len(test)
    for row in test:
        u, i, r = row[1], row[0], row[2]  # user_id, movie_id, rating
        pred_r = cf_model.pred(u, i, normalized=0)  # Unnormalized rating
        se += (pred_r - r) ** 2
    rmse = np.sqrt(se / n_tests) if n_tests > 0 else 0
    np.savetxt('item_item_rmse.csv', [rmse], fmt='%f')
    return rmse

# New function: Compute MAE
def compute_mae(cf_model, test):
    """Compute Mean Absolute Error on test set."""
    ae = 0  # Absolute error
    n_tests = len(test)
    for row in test:
        u, i, r = row[1], row[0], row[2]
        pred_r = cf_model.pred(u, i, normalized=0)
        ae += abs(pred_r - r)
    mae = ae / n_tests if n_tests > 0 else 0
    np.savetxt('item_item_mae.csv', [mae], fmt='%f')
    return mae

# New function: Compute Coverage
def compute_coverage(recommendations):
    """Compute coverage of recommended items."""
    unique_items = np.unique(recommendations[recommendations != -1])
    coverage = len(unique_items) / NUM_ITEMS
    np.savetxt('item_item_coverage.csv', [coverage], fmt='%f')
    return coverage

# New function: Compute Density
def compute_density(Y_data):
    """Compute density of rating matrix."""
    num_ratings = len(Y_data)
    density = num_ratings / (NUM_USERS * NUM_ITEMS)
    np.savetxt('item_item_density.csv', [density], fmt='%f')
    return density

# New function: Extract metrics for reporting/UI
def extract_metrics(cf_model, recommendations, test, Y_data):
    """Extract all metrics and save to JSON/CSV for reporting/UI."""
    recall = compute_recall(recommendations, test)
    rmse = compute_rmse(cf_model, test)
    mae = compute_mae(cf_model, test)
    coverage = compute_coverage(recommendations)
    density = compute_density(Y_data)
    
    metrics = {
        'recall': recall,
        'rmse': rmse,
        'mae': mae,
        'coverage': coverage,
        'density': density
    }
    
    # Save to JSON
    with open('item_item_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('item_item_metrics.csv', index=False)
    
    return metrics

def get_top_n_similar_movies(movie_name, sim_matrix, name_to_movie_id, movie_id_to_name, n=5):
    """Get top-N similar movies for a given movie."""
    if movie_name not in name_to_movie_id:
        return f"Movie '{movie_name}' not found. Please check the movie name (e.g., 'Toy Story (1995)')."
    movie_idx = name_to_movie_id[movie_name]
    sim_scores = sim_matrix[movie_idx]
    top_indices = np.argsort(sim_scores)[::-1][1:n+1]
    return [movie_id_to_name[idx] for idx in top_indices]

def get_user_recommendations(user_id, cf_model, movie_id_to_name, n=5):
    """Get top-N recommendations for a user."""
    if user_id < 1 or user_id > NUM_USERS:
        return f"User ID {user_id} is invalid."
    recommendations = cf_model.recommend(user_id - 1, n)
    return [movie_id_to_name[idx] for idx in recommendations if idx in movie_id_to_name]

def main():
    """Main function to run the Item-Item CF recommendation pipeline."""
    np.random.seed(42)
    print("Loading data and training Item-Item CF model...")
    Y_data, movie_id_to_name, name_to_movie_id = load_data(RATING_FILE, MOVIE_FILE)
    train, test = split_train_test(Y_data)
    cf = ItemItemCF(train, k=K_NEIGHBORS)
    cf.fit()
    print("Generating recommendations for all users...")
    recommendations = np.zeros((NUM_USERS, TOP_N), dtype=int)
    for u in range(NUM_USERS):
        recommendations[u] = cf.recommend(u)
    np.savetxt('item_item_topn.csv', recommendations, delimiter=',', fmt='%d')
    
    # Compute and extract metrics
    print("Computing metrics for evaluation...")
    metrics = extract_metrics(cf, recommendations, test, Y_data)
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Coverage: {metrics['coverage']:.4f}")
    print(f"Density: {metrics['density']:.4f}")
    
    print("\nStep 1: Enter a movie name to find 5 similar movies (based on all users' ratings).")
    movie_name = input("Enter movie name (e.g., Toy Story (1995)): ")
    top_n_movies = get_top_n_similar_movies(movie_name, cf.S, name_to_movie_id, movie_id_to_name)
    print(f"Top 5 similar movies to '{movie_name}':")
    for i, movie in enumerate(top_n_movies, 1):
        print(f"{i}. {movie}")
    
    print("\nStep 2: Enter a user ID to get 5 personalized movie recommendations (based on their ratings).")
    user_id = int(input("Enter user ID (1-943): "))
    user_recs = get_user_recommendations(user_id, cf, movie_id_to_name)
    print(f"Top 5 recommendations for User {user_id}:")
    for i, movie in enumerate(user_recs, 1):
        print(f"{i}. {movie}")

if __name__ == "__main__":
    main()