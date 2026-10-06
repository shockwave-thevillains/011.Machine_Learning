# Hierarchical (Agglomerative) Clustering dengan Ward linkage
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)

for link in ("ward", "complete", "average", "single"):
    lab = AgglomerativeClustering(n_clusters=3, linkage=link).fit_predict(X)
    print(f"linkage={link:<8} ARI vs spesies asli = {adjusted_rand_score(iris.target, lab):.3f}")

Z = linkage(X, method="ward")                   # matriks penggabungan (n-1 baris)
print("\n5 penggabungan terakhir (cluster a, cluster b, jarak, ukuran):")
for a, b, d, n in Z[-5:]:
    print(f"  {int(a):>3} + {int(b):>3}  jarak = {d:6.2f}  anggota = {int(n)}")
for t in (2, 3, 4):
    print(f"Potong dendrogram jadi {t} cluster -> ukuran:",
          sorted(np.bincount(fcluster(Z, t, "maxclust"))[1:].tolist(), reverse=True))

# === VISUALISASI ===
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from _plot import BLUE, RED, fig, save
set_link_color_palette([BLUE, RED, "#000000"])
f, ax = fig(7, 3.4)
dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, above_threshold_color="#888", color_threshold=12)
ax.set(title="Dendrogram Ward (30 cabang terakhir)", ylabel="jarak penggabungan")
ax.grid(False)
save("hierarchical_clustering")
