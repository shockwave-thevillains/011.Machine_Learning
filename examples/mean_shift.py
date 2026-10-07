# Mean Shift: mencari puncak kepadatan tanpa menentukan jumlah cluster
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

pusat_asli = [(1, 1), (-1, -1), (1, -1), (-1.5, 1.5)]
X, _ = make_blobs(n_samples=800, centers=pusat_asli, cluster_std=0.4, random_state=0)
df = pd.DataFrame(X, columns=["x1", "x2"])          # 800 titik, tanpa label
X = df.to_numpy()

for q in (0.1, 0.2, 0.3, 0.5):
    bw = estimate_bandwidth(X, quantile=q, random_state=0)
    ms = MeanShift(bandwidth=bw, bin_seeding=True).fit(X)
    print(f"quantile={q:<4} bandwidth={bw:.3f} -> {len(ms.cluster_centers_)} cluster")

bw = estimate_bandwidth(X, quantile=0.2, random_state=0)
ms = MeanShift(bandwidth=bw, bin_seeding=True).fit(X)
print("\nPusat yang ditemukan (tanpa memberi tahu jumlah cluster):")
for c in ms.cluster_centers_:
    print(f"  ({c[0]:+.2f}, {c[1]:+.2f})  anggota = {np.sum(ms.labels_ == ms.predict([c])[0])}")
print("Pusat asli:", pusat_asli)

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
baru = pd.DataFrame({"x1": [1.1, -1.4, 0.0, 3.0], "x2": [-0.9, 1.6, 0.0, 3.0]})
for xb, k in zip(baru.to_numpy(), ms.predict(baru.to_numpy())):
    c = ms.cluster_centers_[k]
    print(f"titik ({xb[0]:+.1f}, {xb[1]:+.1f}) -> cluster {k} (puncak di ({c[0]:+.2f}, {c[1]:+.2f}), jarak {np.linalg.norm(xb - c):.2f})")

# === VISUALISASI ===
from _plot import BLUE, RED, BLACK, fig, save
cols = [BLUE, RED, BLACK, "#7f7fff", "#ff8080"]
f, ax = fig(5.2, 3.6)
for k in range(len(ms.cluster_centers_)):
    ax.scatter(*X[ms.labels_ == k].T, s=6, color=cols[k % 5], alpha=.6)
ax.scatter(*ms.cluster_centers_.T, marker="X", s=140, color="white", edgecolors="black", lw=1.4)
ax.set(title=f"Mean Shift: {len(ms.cluster_centers_)} puncak kepadatan (bandwidth {bw:.2f})")
save("mean_shift")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("mean_shift", df.assign(cluster_hasil=ms.labels_), {
    "x1, x2": "Koordinat titik (hanya dua kolom ini yang diolah).",
    "cluster_hasil": "HASIL Mean Shift: puncak kepadatan tempat titik berakhir.",
})
simpan("mean_shift_baru", baru, {"x1, x2": "Koordinat titik baru."})
