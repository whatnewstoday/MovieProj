import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from CfilterClaude4 import UserUserCF
from sklearn.model_selection import train_test_split
# === Sample Metrics (You can replace with dynamic values later) ===
metrics_data = {
    "Model": [
        "User-User CF",
        "Item-Item CF (Cosine)",
        "Item-Item CF (Norm Cosine)",
        "Item-Item CF (Probability)",
        "Item-Item CF (Norm Probability)"
    ],
    "RMSE": [0.9980, None, None, None, None],
    "MAE": [0.7886, None, None, None, None],
    "Recall@10": [None, 0.2471, 0.1633, 0.2428, 0.2534],
    "Coverage": [0.1819, None, None, None, None],
    "Diversity": [0.9545, None, None, None, None],
}

metrics_df = pd.DataFrame(metrics_data)

st.set_page_config(page_title="Movie Recommender Metrics Dashboard", layout="wide")
st.title("🎬 Movie Recommendation System: Model Comparison Dashboard")

st.markdown("""
This dashboard shows evaluation metrics (Accuracy, Recall, Coverage, Diversity) 
for different collaborative filtering models using the MovieLens 100k dataset.
""")

# === Show metrics table ===
st.subheader("📊 Metrics Summary Table")
st.dataframe(metrics_df.set_index("Model"), use_container_width=True)

# === Recall Chart ===
st.subheader("🎯 Recall@10 Comparison")
recall_data = metrics_df.dropna(subset=["Recall@10"])
fig1, ax1 = plt.subplots()
ax1.bar(recall_data["Model"], recall_data["Recall@10"], color="skyblue")
ax1.set_ylabel("Recall@10")
ax1.set_xticklabels(recall_data["Model"], rotation=45, ha="right")
fig1.tight_layout()
st.pyplot(fig1)

# === Coverage & Diversity ===
st.subheader("🌍 Coverage and Diversity")
cov_div_data = metrics_df.dropna(subset=["Coverage", "Diversity"])
fig2, ax2 = plt.subplots()
ax2.bar(["Coverage", "Diversity"], [cov_div_data.iloc[0]["Coverage"], cov_div_data.iloc[0]["Diversity"]], color=["green", "orange"])
ax2.set_ylim(0, 1.0)
fig2.tight_layout()
st.pyplot(fig2)

st.markdown("""
👉 Tip: Try combining models with high Recall and high Diversity for a better overall recommendation experience.
""")





st.title("KNN Recommender Elbow Method Dashboard")

# --- Load ratings data ---
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings_df = pd.read_csv('data/ml-100k/u.data', sep='\t', names=r_cols)

# --- Split into train/test ---
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
rate_train = train_df[['user_id', 'movie_id', 'rating']].values
rate_test = test_df[['user_id', 'movie_id', 'rating']].values

st.write(f"Train set size: {len(rate_train)}")
st.write(f"Test set size: {len(rate_test)}")

# --- Elbow method for k ---
def compute_rmse(preds, actuals):
    return np.sqrt(np.mean((preds - actuals) ** 2))

k_min, k_max = st.slider("Select k range for elbow plot", 1, 100, (1, 30))
k_values = list(range(k_min, k_max + 1))
rmse_scores = []

progress = st.progress(0)
for idx, k in enumerate(k_values):
    model = UserUserCF(rate_train, k=k)
    model.fit()
    preds = []
    actuals = []
    for user_id, item_id, actual_rating in rate_test:
        try:
            pred = model.pred(user_id, item_id)
        except Exception:
            pred = np.mean(rate_train[:, 2])  # fallback to global mean if prediction fails
        preds.append(pred)
        actuals.append(actual_rating)
    rmse = compute_rmse(np.array(preds), np.array(actuals))
    rmse_scores.append(rmse)
    progress.progress((idx + 1) / len(k_values))

# --- Plot ---
fig, ax = plt.subplots()
ax.plot(k_values, rmse_scores, marker='o')
ax.set_xlabel('Number of Neighbors (k)')
ax.set_ylabel('RMSE')
ax.set_title('Elbow Method for Optimal k')
fig.tight_layout()
st.pyplot(fig)

st.markdown("""
👉 **Tip:** The "elbow" in the curve is usually the best k.  
Choose the lowest k after which RMSE stops decreasing rapidly!
""")