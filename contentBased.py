
import pandas as pd
u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('data/ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

n_users = users.shape[0]
print ('Number of users:', n_users)
# users.head() #uncomment this to see some few examples

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('data/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('data/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.values
rate_test = ratings_test.values

print ('Number of traing rates:', rate_train.shape[0])
print ('Number of test rates:', rate_test.shape[0])

i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('data/ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')

n_items = items.shape[0]
print ('Number of items:', n_items)

X0 = items.values
X_train_counts = X0[:, -19:]

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=True, norm ='l2')
tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()

import numpy as np
def get_items_rated_by_user(rate_matrix, user_id):
    """
    in each line of rate_matrix, we have infor: user_id, item_id, rating (scores), time_stamp
    we care about the first three values
    return (item_ids, scores) rated by user user_id
    """
    y = rate_matrix[:,0] # all users
    # item indices rated by user_id
    # we need to +1 to user_id since in the rate_matrix, id starts from 1
    # while index in python starts from 0
    ids = np.where(y == user_id +1)[0]
    item_ids = rate_matrix[ids, 1] - 1 # index starts from 0
    scores = rate_matrix[ids, 2]
    return (item_ids, scores)

from sklearn.linear_model import Ridge
from sklearn import linear_model

d = tfidf.shape[1] # data dimension
W = np.zeros((d, n_users))
b = np.zeros((1, n_users))

for n in range(n_users):
    ids, scores = get_items_rated_by_user(rate_train, n)
    clf = Ridge(alpha=0.01, fit_intercept  = True)
    Xhat = tfidf[ids, :]

    clf.fit(Xhat, scores)
    W[:, n] = clf.coef_
    b[0, n] = clf.intercept_    

Yhat = tfidf.dot(W) + b


n = 10
np.set_printoptions(precision=2) # 2 digits after .
ids, scores = get_items_rated_by_user(rate_test, n)
Yhat[n, ids]
print('Rated movies ids :', ids )
print('True ratings     :', scores)
print('Predicted ratings:', Yhat[ids, n])

from math import sqrt

def evaluate(Yhat, rates, W, b):
    se = 0
    cnt = 0
    for n in range(n_users):
        ids, scores_truth = get_items_rated_by_user(rates, n)
        scores_pred = Yhat[ids, n]
        e = scores_truth - scores_pred
        se += (e*e).sum(axis = 0)
        cnt += e.size
    return sqrt(se/cnt)

print ('RMSE for training:', evaluate(Yhat, rate_train, W, b))
print ('RMSE for test    :', evaluate(Yhat, rate_test, W, b))

def recommend_movies(user_id, Yhat, movies, ratings, top_n=10):
    # Lấy danh sách movie_id user đã đánh giá
    rated_movies = ratings[ratings['user_id'] == user_id]['movie_id'].tolist()

    # Dự đoán điểm cho tất cả phim của user
    user_pred_scores = Yhat[:, user_id]

    # Lọc ra phim chưa đánh giá
    unseen_ids = [i for i in range(len(user_pred_scores)) if i not in rated_movies]

    # Lấy điểm dự đoán cho phim chưa xem
    unseen_scores = [(i, user_pred_scores[i]) for i in unseen_ids]

    # Sắp xếp theo điểm dự đoán giảm dần
    unseen_scores.sort(key=lambda x: x[1], reverse=True)

    # Lấy top N phim gợi ý
    top_movies_idx = [i for i, score in unseen_scores[:top_n]]

    # Trả về thông tin phim gợi ý
    recommended = movies[movies['movie id'].isin(top_movies_idx)][['movie id', 'movie title']]

    return recommended

import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('data/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('data/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

# Gộp dữ liệu train và test thành một bảng đánh giá đầy đủ
ratings = pd.concat([ratings_base, ratings_test], ignore_index=True)

user_id = 21  # ví dụ
top_recommendations = recommend_movies(user_id, Yhat, items, ratings, top_n=10)
print(top_recommendations)