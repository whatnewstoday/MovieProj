import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
st.pyplot(fig1)

# === Coverage & Diversity ===
st.subheader("🌍 Coverage and Diversity")
cov_div_data = metrics_df.dropna(subset=["Coverage", "Diversity"])
fig2, ax2 = plt.subplots()
ax2.bar(["Coverage", "Diversity"], [cov_div_data.iloc[0]["Coverage"], cov_div_data.iloc[0]["Diversity"]], color=["green", "orange"])
ax2.set_ylim(0, 1.0)
st.pyplot(fig2)

st.markdown("""
👉 Tip: Try combining models with high Recall and high Diversity for a better overall recommendation experience.
""")
