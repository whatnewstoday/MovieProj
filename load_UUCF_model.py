import numpy as np
from scipy import sparse
from Cfilter import UserUserCF

# Load data
Y_data = np.loadtxt("model_data/Y_data.csv", delimiter=",")
mu = np.loadtxt("model_data/user_means.csv", delimiter=",")
S = np.loadtxt("model_data/user_similarity.csv", delimiter=",")
Ybar_dense = np.loadtxt("model_data/Ybar.csv", delimiter=",")

# Rebuild model
rs = UserUserCF(Y_data, k=30)
rs.mu = mu
rs.S = S
rs.Ybar = sparse.csr_matrix(Ybar_dense)

# Now you can call rs.recommend(), rs.pred() without retraining
