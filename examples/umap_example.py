# UMAP: reduksi dimensi berbasis topologi, cepat & menjaga struktur global
import time
import warnings

import umap
from sklearn.datasets import load_digits
from sklearn.manifold import trustworthiness

warnings.filterwarnings("ignore")
X, y = load_digits(return_X_y=True)

t = time.perf_counter()
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
Z = reducer.fit_transform(X)
print(f"Waktu UMAP             : {time.perf_counter() - t:.1f} detik (termasuk kompilasi JIT numba)")
print(f"Trustworthiness UMAP   : {trustworthiness(X, Z, n_neighbors=10):.3f}")

baru = reducer.transform(X[:5])                 # berbeda dari t-SNE: bisa memetakan data baru
print("Data baru bisa diproyeksikan:", baru.astype(float).round(2).tolist()[:2], "...")

# === VISUALISASI ===
from _plot import fig, save
f, ax = fig(5.6, 4)
ax.scatter(*Z.T, c=y, cmap="bwr", s=4)
for d in range(10):
    ax.text(*Z[y == d].mean(0), str(d), fontsize=12, weight="bold", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", lw=.6))
ax.set(title="UMAP: digit tulisan tangan", xticks=[], yticks=[])
save("umap_example")
