# K-Means: segmentasi pelanggan berdasarkan pendapatan & skor belanja
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=400, centers=[(25, 20), (25, 80), (60, 50), (95, 20), (95, 80)],
                  cluster_std=7, random_state=42)
df = pd.DataFrame(X, columns=["pendapatan_juta_thn", "skor_belanja"])   # 400 pelanggan, tanpa label
X = df.to_numpy()

print(" k | inertia (SSE) | silhouette")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    print(f"{k:>2} | {km.inertia_:>13,.0f} | {silhouette_score(X, km.labels_):.3f}")

km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X)
print("\nPusat cluster (pendapatan, skor belanja) untuk k=5:")
for i, c in enumerate(km.cluster_centers_):
    print(f"  segmen {i}: ({c[0]:5.1f}, {c[1]:5.1f}) -> {np.sum(km.labels_ == i)} pelanggan")
print("Pelanggan baru (80 jt, skor 85) masuk segmen", km.predict([[80, 85]])[0])

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
baru = pd.DataFrame({"pendapatan_juta_thn": [30, 62, 100, 45], "skor_belanja": [85, 45, 15, 50]})
jarak = km.transform(baru.to_numpy())
for (pend, skor), seg, d in zip(baru.to_numpy(), km.predict(baru.to_numpy()), jarak.min(1)):
    c = km.cluster_centers_[seg]
    print(f"pelanggan ({pend} jt, skor {skor}) -> segmen {seg} (pusat {c[0]:.0f} jt / skor {c[1]:.0f}) | jarak ke pusat {d:.1f}")

# === VISUALISASI ===
from _plot import BLUE, RED, BLACK, fig, save
cols = [BLUE, RED, BLACK, "#7f7fff", "#ff8080"]
f, ax = fig(5.6, 3.6)
for i in range(5):
    ax.scatter(*X[km.labels_ == i].T, s=10, color=cols[i])
ax.scatter(*km.cluster_centers_.T, marker="X", s=140, color="white", edgecolors="black", lw=1.4)
ax.set(xlabel="pendapatan (juta/thn)", ylabel="skor belanja", title="K-Means: 5 segmen pelanggan")
save("kmeans")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("kmeans", df.assign(segmen_hasil=km.labels_), {
    "pendapatan_juta_thn": "Pendapatan pelanggan per tahun (juta Rp).",
    "skor_belanja": "Skor perilaku belanja 1–100.",
    "segmen_hasil": "HASIL K-Means (k=5): nomor segmen tiap pelanggan. Tidak ada di data awal.",
}, catatan="Data unsupervised: tidak ada kolom target. Hanya dua kolom pertama yang diolah.")
simpan("kmeans_baru", baru, {"pendapatan_juta_thn, skor_belanja": "Pelanggan baru yang belum punya segmen."})
