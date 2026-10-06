# t-SNE: visualisasi data berdimensi tinggi ke 2D dengan menjaga tetangga lokal
import time

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE, trustworthiness
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True)

t = time.perf_counter()
tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0)
Z = tsne.fit_transform(X)
print(f"Waktu t-SNE            : {time.perf_counter() - t:.1f} detik untuk {len(X)} titik")
print(f"KL divergence akhir    : {tsne.kl_divergence_:.3f}")
print(f"Trustworthiness t-SNE  : {trustworthiness(X, Z, n_neighbors=10):.3f}")
print(f"Trustworthiness PCA-2D : {trustworthiness(X, PCA(2).fit_transform(X), n_neighbors=10):.3f}")
print("(trustworthiness 1.0 = tetangga terdekat di 2D sama persis dengan di 64D)")

# === VISUALISASI ===
from _plot import fig, save
f, ax = fig(5.6, 4)
ax.scatter(*Z.T, c=y, cmap="bwr", s=4)
for d in range(10):
    ax.text(*Z[y == d].mean(0), str(d), fontsize=12, weight="bold", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", lw=.6))
ax.set(title="t-SNE: 1.797 digit tulisan tangan", xticks=[], yticks=[])
save("tsne")
