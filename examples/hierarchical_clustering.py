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

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
from sklearn.preprocessing import StandardScaler
baru = pd.DataFrame({"sepal_pjg_cm": [5.0, 6.0, 6.9], "sepal_lbr_cm": [3.5, 2.7, 3.1],
                     "petal_pjg_cm": [1.4, 4.4, 5.6], "petal_lbr_cm": [0.2, 1.3, 2.2]}, index=["bunga_1", "bunga_2", "bunga_3"])
skala = StandardScaler().fit(iris.data)
label_ward = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X)
pusat = np.array([X[label_ward == k].mean(0) for k in range(3)])  # clustering hierarkis tidak punya predict()
for nama, z in zip(baru.index, skala.transform(baru.to_numpy())):
    d = np.linalg.norm(pusat - z, axis=1)
    k = int(d.argmin())
    mayoritas = iris.target_names[np.bincount(iris.target[label_ward == k]).argmax()]
    print(f"{nama} -> cluster {k} (jarak ke pusat {d[k]:.2f}; anggota cluster ini mayoritas {mayoritas})")

# === VISUALISASI ===
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from _plot import BLUE, RED, fig, save
set_link_color_palette([BLUE, RED, "#000000"])
f, ax = fig(7, 3.4)
dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, above_threshold_color="#888", color_threshold=12)
ax.set(title="Dendrogram Ward (30 cabang terakhir)", ylabel="jarak penggabungan")
ax.grid(False)
save("hierarchical_clustering")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import KET_IRIS, iris_df, simpan
simpan("hierarchical_clustering", iris_df(), KET_IRIS, idx=[0, 1, 50, 51, 100, 101],
       catatan="Dataset diurutkan per spesies (50 bunga per spesies), jadi baris contoh diambil dari ketiganya. Kolom spesies TIDAK dipakai untuk clustering; hanya untuk mengukur ARI. Keempat fitur distandarkan dulu.")
simpan("hierarchical_clustering_baru", baru.reset_index(names="bunga"), {"bunga_1 … bunga_3": "Tiga bunga baru yang diukur, spesiesnya belum diketahui."})
