# day34_tsne.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Day 34 - t-SNE Visualization", layout="wide")
st.title("📌 Day 34 — t-SNE: Dimensionality Reduction & Visualization")

st.markdown("""
t-SNE (**t-distributed Stochastic Neighbor Embedding**) is a powerful algorithm for 
**visualizing high-dimensional data** in 2D or 3D space.  

In this project, we apply t-SNE on the **Digits dataset** (handwritten digits 0–9).
""")

# -------------------------------
# Load dataset
# -------------------------------
digits = load_digits()
X = digits.data
y = digits.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("🔧 Settings")

perplexity = st.sidebar.slider("Perplexity", 5, 50, 30)
learning_rate = st.sidebar.slider("Learning Rate", 10, 500, 200)
n_iter = st.sidebar.slider("Number of Iterations", 250, 2000, 1000)
dim = st.sidebar.radio("Dimensions", [2, 3], index=0)

# -------------------------------
# Run t-SNE
# -------------------------------
st.write("Running **t-SNE**... Please wait ⏳")

tsne = TSNE(
    n_components=dim,
    perplexity=perplexity,
    learning_rate=learning_rate,
    max_iter=n_iter,        # ✅ FIXED: changed from n_iter → max_iter
    random_state=42
)

X_embedded = tsne.fit_transform(X_scaled)

# -------------------------------
# Plot Results
# -------------------------------
st.subheader("🔮 t-SNE Visualization")

fig = plt.figure(figsize=(8, 6))
if dim == 2:
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=y, palette="tab10", s=40, alpha=0.8, edgecolor="none"
    )
    plt.title("t-SNE (2D) of Digits Dataset")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
else:
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2],
        c=y, cmap="tab10", s=40, alpha=0.8
    )
    legend = ax.legend(
        *scatter.legend_elements(), title="Digits", loc="best"
    )
    ax.add_artist(legend)
    ax.set_title("t-SNE (3D) of Digits Dataset")

st.pyplot(fig)

st.success("✅ Visualization complete!")
