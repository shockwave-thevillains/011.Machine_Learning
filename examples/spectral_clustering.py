# Spectral Clustering: clustering lewat eigenvector graf kemiripan
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import make_circles
from sklearn.metrics import adjusted_rand_score

X, y = make_circles(n_samples=500, factor=0.45, noise=0.05, random_state=0)  # 2 lingkaran sepusat

km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
sc = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_neighbors=10,
                        random_state=0).fit_predict(X)
print(f"K-Means              ARI = {adjusted_rand_score(y, km):.3f}")
print(f"Spectral Clustering  ARI = {adjusted_rand_score(y, sc):.3f}")

# langkah inti di balik layar: graf k-NN -> Laplacian -> eigenvector
import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.neighbors import kneighbors_graph
A = kneighbors_graph(X, 10, include_self=False)
A = 0.5 * (A + A.T)
eigval = np.linalg.eigvalsh(laplacian(A.toarray(), normed=True))
print("5 eigenvalue terkecil Laplacian:", np.round(eigval[:5], 4).tolist())
print("-> ada 2 eigenvalue ≈ 0, artinya graf punya 2 komponen terhubung = 2 cluster")

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
f, axes = fig(7.2, 3.2, ncols=2)
for ax, l, t in zip(axes, (km, sc), ("K-Means", "Spectral Clustering")):
    ax.scatter(*X[l == 0].T, s=7, color=BLUE)
    ax.scatter(*X[l == 1].T, s=7, color=RED)
    ax.set(title=t, xticks=[], yticks=[], aspect="equal")
save("spectral_clustering")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("spectral_clustering", {"x1": X[:, 0], "x2": X[:, 1], "lingkaran_asli": y, "cluster_spectral": sc}, {
    "x1, x2": "Koordinat titik (hanya dua kolom ini yang diolah).",
    "lingkaran_asli": "0 = lingkaran luar, 1 = lingkaran dalam. Hanya untuk evaluasi ARI.",
    "cluster_spectral": "HASIL Spectral Clustering. Nomor cluster boleh tertukar (0↔1); yang penting pengelompokannya sama.",
})
