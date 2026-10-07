# DBSCAN: clustering berbasis kepadatan, bisa menemukan bentuk sembarang & noise
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

X, y = make_moons(n_samples=400, noise=0.07, random_state=0)
rng = np.random.default_rng(0)
X = np.vstack([X, rng.uniform([-1.5, -1], [2.5, 1.5], (20, 2))])   # + 20 titik noise
y = np.concatenate([y, -np.ones(20, int)])

km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
db = DBSCAN(eps=0.15, min_samples=5).fit(X)
lab = db.labels_

print(f"K-Means  ARI (400 titik bulan) = {adjusted_rand_score(y[:400], km[:400]):.3f}")
print(f"DBSCAN   ARI (400 titik bulan) = {adjusted_rand_score(y[:400], lab[:400]):.3f}")
print(f"Cluster ditemukan DBSCAN       = {len(set(lab)) - (-1 in lab)}")
print(f"Titik ditandai noise (-1)      = {np.sum(lab == -1)} (20 di antaranya noise sungguhan: {np.sum(lab[400:] == -1)} tertangkap)")
print(f"Core points                    = {len(db.core_sample_indices_)}")

# === VISUALISASI ===
from _plot import BLUE, RED, BLACK, fig, save
f, axes = fig(7.2, 3, ncols=2)
for ax, l, t in zip(axes, (km, lab), ("K-Means (k=2)", "DBSCAN (eps=0.15)")):
    for v, c in ((0, BLUE), (1, RED)):
        ax.scatter(*X[l == v].T, s=7, color=c)
    ax.scatter(*X[l == -1].T, s=18, marker="x", color=BLACK, label="noise")
    ax.set(title=t, xticks=[], yticks=[])
axes[1].legend()
save("dbscan")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("dbscan", {"x1": X[:, 0], "x2": X[:, 1], "label_asli": y, "cluster_dbscan": lab}, {
    "x1, x2": "Koordinat titik (hanya dua kolom ini yang diolah).",
    "label_asli": "Bulan sabit asal (0/1) atau −1 untuk 20 titik noise yang ditambahkan. Hanya untuk evaluasi.",
    "cluster_dbscan": "HASIL DBSCAN: nomor cluster, atau −1 jika dianggap noise.",
}, idx=[0, 1, 2, 400, 401, 402], catatan="Tiga baris terakhir adalah titik noise yang sengaja ditambahkan.")
