import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class UserUserCF():
    """User-User Collaborative Filtering using lazy KNN
    Assume input schema is
    user_id, item_id, rating
    """
    def __init__(self, Y_data, k, dist_func = cosine_similarity):
        self.Y_data = Y_data
        self.k = k # number of neighbor points
        self.dist_func = dist_func
        self.Ybar_data = None
        # number of users and items. Add 1 since id starts from 0. [user_id, item_id, rating]
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
        
        
    def add(self, new_data):
        """
        Update Y_data matrix when new ratings come.
        For simplicity, suppose that there is no new user or item.
        Adding vertically
        """
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)
    
    def normalize_Y(self):
        users = self.Y_data[:, 0] # user column
        
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        
        for u in range(self.n_users):
            # row indices of ratings done by user u
            ids = np.where(users == u)[0].astype(np.int32)
            # ratings by user u
            ratings = self.Y_data[ids, 2]
            # take mean
            m = np.mean(ratings)
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[u] = m
            # normalize ratings by subtracting user mean
            self.Ybar_data[ids, 2] = ratings - self.mu[u]
    
        # Create sparse rating matrix (items x users)
        # This is more memory efficient for large datasets
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()
    
    def similarity(self):
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
    
    
    def refresh(self):
        """
        Normalize data and calculate similarity matrix again (after
        some few ratings added)
        """
        self.normalize_Y()
        self.similarity() 
        
    def fit(self):
        self.refresh()
        
        
    def pred(self, u, i, normalized = 1):
        """
        Predict the rating of user u for item i using user-user collaborative filtering
        """
        # Step 1: find all users who rated item i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)

        # Step 2: find similarity between the current user and others who rated i
        sim = self.S[u, users_rated_i]
        
        # Step 3: find the k most similar users
        a = np.argsort(sim)[-self.k:]
        nearest_s = sim[a]
        
        # Step 4: get normalized ratings from similar users for item i
        r = self.Ybar[i, users_rated_i[a]]
        
        if normalized:
            # Return normalized prediction
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)
        else:
            # Return denormalized prediction (add back user mean)
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]

    def recommend(self, u, normalized = 1, n_recommendations = 10, max_candidates=500):
        """
        Determine top n items to be recommended for user u using user-user CF.
        The decision is made based on predicted ratings for items not yet rated.
        Returns the top n_recommendations items with highest predicted ratings.
        
        max_candidates: Limit the number of items to consider for faster computation
        """
        # Find items already rated by user u
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = set(self.Y_data[ids, 1].tolist())
        
        # Get candidate items (not rated by user)
        candidate_items = [i for i in range(self.n_items) if i not in items_rated_by_u]
        
        # Limit candidates for performance if dataset is large
        if len(candidate_items) > max_candidates:
            candidate_items = np.random.choice(candidate_items, max_candidates, replace=False)
            
        # Store items with their predicted ratings
        item_ratings = []
        for i in candidate_items:
            try:
                rating = self.pred(u, i, normalized=0)  # Get actual rating prediction
                if rating > 0:
                    item_ratings.append((i, rating))
            except:
                continue
        
        # Sort by predicted rating (descending) and return top n
        item_ratings.sort(key=lambda x: x[1], reverse=True)
        recommended_items = [item[0] for item in item_ratings[:n_recommendations]]
        
        return recommended_items
    
    def print_recommendation(self, movie_titles=None, n_recommendations=10):
        """Print top n recommendations for all users"""
        print(f'Top {n_recommendations} Recommendations:')
        for u in range(self.n_users):
            recommended_items = self.recommend(u, n_recommendations=n_recommendations)
            if recommended_items:  # Only print if there are recommendations
                print(f'  User {u + 1}:')
                for i in recommended_items:
                    if movie_titles is not None:
                        title = movie_titles[movie_titles['movie_id'] == i + 1]['title'].values
                        if len(title) > 0:
                            print(f'    - {title[0]}')
                    else:
                        print(f'    - Item ID {i}')
                print()  # Empty line between users

    def evaluate_rmse(self, test_data):
        """
        Calculate Root Mean Square Error (RMSE) on test data
        """
        n_tests = test_data.shape[0]
        SE = 0  # sum of squared errors
        valid_predictions = 0
        
        for n in range(n_tests):
            try:
                pred = self.pred(test_data[n, 0], test_data[n, 1], normalized=0)
                if not np.isnan(pred):
                    SE += (pred - test_data[n, 2])**2
                    valid_predictions += 1
            except:
                continue
                
        if valid_predictions == 0:
            return float('inf')
        
        RMSE = np.sqrt(SE / valid_predictions)
        return RMSE
    
    def evaluate_mae(self, test_data):
        """
        Calculate Mean Absolute Error (MAE) on test data
        """
        n_tests = test_data.shape[0]
        AE = 0  # sum of absolute errors
        valid_predictions = 0
        
        for n in range(n_tests):
            try:
                pred = self.pred(test_data[n, 0], test_data[n, 1], normalized=0)
                if not np.isnan(pred):
                    AE += abs(pred - test_data[n, 2])
                    valid_predictions += 1
            except:
                continue
                
        if valid_predictions == 0:
            return float('inf')
        
        MAE = AE / valid_predictions
        return MAE
    
    def evaluate_precision_recall(self, test_data, threshold=4.0, n_recommendations=10):
        """
        Calculate Precision, Recall, and F1-score for recommendations
        Items with rating >= threshold are considered relevant
        """
        precisions = []
        recalls = []
        
        # Get unique users in test data
        test_users = np.unique(test_data[:, 0])
        
        for user in test_users:
            # Get actual relevant items for this user (rating >= threshold)
            user_test_data = test_data[test_data[:, 0] == user]
            relevant_items = set(user_test_data[user_test_data[:, 2] >= threshold, 1])
            
            if len(relevant_items) == 0:
                continue
                
            # Get recommended items
            try:
                recommended_items = set(self.recommend(user, n_recommendations=n_recommendations))
                
                if len(recommended_items) == 0:
                    continue
                
                # Calculate precision and recall
                relevant_recommended = relevant_items.intersection(recommended_items)
                
                precision = len(relevant_recommended) / len(recommended_items) if len(recommended_items) > 0 else 0
                recall = len(relevant_recommended) / len(relevant_items) if len(relevant_items) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
            except:
                continue
        
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return avg_precision, avg_recall, f1_score
    
    def evaluate_coverage(self, n_recommendations=10, sample_users=100):
        """
        Calculate catalog coverage - percentage of items that can be recommended
        Uses sampling for large datasets to improve performance
        """
        all_recommended_items = set()
        
        # Sample users if dataset is large
        if self.n_users > sample_users:
            sampled_users = np.random.choice(self.n_users, sample_users, replace=False)
        else:
            sampled_users = range(self.n_users)
        
        for user in sampled_users:
            try:
                recommended_items = self.recommend(user, n_recommendations=n_recommendations)
                all_recommended_items.update(recommended_items)
            except:
                continue
        
        coverage = len(all_recommended_items) / self.n_items if self.n_items > 0 else 0
        return coverage
    
    def evaluate_diversity(self, n_recommendations=10, sample_users=50):
        """
        Calculate intra-list diversity - average pairwise distance between recommended items
        Uses Jaccard distance based on users who rated the items
        Uses sampling for large datasets to improve performance
        """
        diversities = []
        
        # Sample users if dataset is large
        if self.n_users > sample_users:
            sampled_users = np.random.choice(self.n_users, sample_users, replace=False)
        else:
            sampled_users = range(self.n_users)
        
        for user in sampled_users:
            try:
                recommended_items = self.recommend(user, n_recommendations=n_recommendations)
                
                if len(recommended_items) < 2:
                    continue
                
                # Calculate pairwise diversity
                pairwise_distances = []
                for i in range(len(recommended_items)):
                    for j in range(i+1, len(recommended_items)):
                        item1, item2 = recommended_items[i], recommended_items[j]
                        
                        # Get users who rated each item
                        users_item1 = set(self.Y_data[self.Y_data[:, 1] == item1, 0])
                        users_item2 = set(self.Y_data[self.Y_data[:, 1] == item2, 0])
                        
                        # Calculate Jaccard distance
                        intersection = len(users_item1.intersection(users_item2))
                        union = len(users_item1.union(users_item2))
                        
                        jaccard_similarity = intersection / union if union > 0 else 0
                        jaccard_distance = 1 - jaccard_similarity
                        
                        pairwise_distances.append(jaccard_distance)
                
                if pairwise_distances:
                    diversities.append(np.mean(pairwise_distances))
            except:
                continue
        
        avg_diversity = np.mean(diversities) if diversities else 0
        return avg_diversity
    
    def comprehensive_evaluation(self, test_data, threshold=4.0, n_recommendations=10):
        """
        Perform comprehensive evaluation of the collaborative filtering system
        """
        print("=== User-User Collaborative Filtering Evaluation ===")
        print(f"Dataset: {self.n_users} users, {self.n_items} items")
        print(f"Training data: {len(self.Y_data)} ratings")
        print(f"Test data: {len(test_data)} ratings")
        print(f"K (neighbors): {self.k}")
        print()
        
        # Accuracy metrics
        print("Calculating accuracy metrics...")
        rmse = self.evaluate_rmse(test_data)
        mae = self.evaluate_mae(test_data)
        
        print("=== Accuracy Metrics ===")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print()
        
        # Coverage and diversity (with progress indicators)
        print("Calculating coverage metrics (sampling users for efficiency)...")
        coverage = self.evaluate_coverage(n_recommendations)
        
        print("Calculating diversity metrics (sampling users for efficiency)...")
        diversity = self.evaluate_diversity(n_recommendations)
        
        print("=== Coverage and Diversity Metrics ===")
        print(f"Catalog Coverage: {coverage:.4f} ({coverage*100:.1f}%)")
        print(f"Intra-list Diversity: {diversity:.4f}")
        print()
        
        return {
            'rmse': rmse,
            'mae': mae,
            'coverage': coverage,
            'diversity': diversity
        }


# Example usage with MovieLens dataset
if __name__ == "__main__":
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

    # Read training and test data from MovieLens dataset
    # Note: You'll need to have the ml-100k dataset files
    try:
        ratings_base = pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
        ratings_test = pd.read_csv('ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')

        # Convert DataFrame to numpy array (user, movie, rating data)
        rate_train = ratings_base.to_numpy()
        rate_test = ratings_test.to_numpy()

        # Convert user and movie indices to start from 0 instead of 1
        rate_train[:, :2] -= 1
        rate_test[:, :2] -= 1

        # Create user-user CF system with k=30 nearest neighbors
        rs = UserUserCF(rate_train, k=30)
        rs.fit()

        # Perform comprehensive evaluation
        evaluation_results = rs.comprehensive_evaluation(rate_test, threshold=4.0, n_recommendations=10)

        # Load movie titles for demonstration
        movie_cols = ['movie_id', 'title']
        movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                           usecols=[0, 1], names=movie_cols, header=None)
        
        # Get top 10 recommendations for user 1
        user_id = 0  # user 1 with 0-based indexing
        recommended_ids = rs.recommend(user_id, n_recommendations=10)

        print("=== Sample Recommendations ===")
        print(f"Top 10 movies recommended for user {user_id + 1}:")
        for i, movie_id in enumerate(recommended_ids, 1):
            title = movies[movies['movie_id'] == movie_id + 1]['title'].values
            if len(title) > 0:
                print(f"{i:2d}. {title[0]}")

        # Print top 5 recommendations for first 3 users
        print(f"\nTop 5 recommendations for first 3 users:")
        for u in range(min(3, rs.n_users)):
            recommended_items = rs.recommend(u, n_recommendations=5)
            if recommended_items:
                print(f'\n  User {u + 1}:')
                for i, item_id in enumerate(recommended_items, 1):
                    title = movies[movies['movie_id'] == item_id + 1]['title'].values
                    if len(title) > 0:
                        print(f'    {i:2d}. {title[0]}')
                        
    except FileNotFoundError:
        print("MovieLens dataset files not found. Please download the ml-100k dataset.")
        print("You can create sample data to test the UserUserCF class:")
        
        