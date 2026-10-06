# Mean Shift: mencari puncak kepadatan tanpa menentukan jumlah cluster
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

pusat_asli = [(1, 1), (-1, -1), (1, -1), (-1.5, 1.5)]
X, _ = make_blobs(n_samples=800, centers=pusat_asli, cluster_std=0.4, random_state=0)

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

# === VISUALISASI ===
from _plot import BLUE, RED, BLACK, fig, save
cols = [BLUE, RED, BLACK, "#7f7fff", "#ff8080"]
f, ax = fig(5.2, 3.6)
for k in range(len(ms.cluster_centers_)):
    ax.scatter(*X[ms.labels_ == k].T, s=6, color=cols[k % 5], alpha=.6)
ax.scatter(*ms.cluster_centers_.T, marker="X", s=140, color="white", edgecolors="black", lw=1.4)
ax.set(title=f"Mean Shift: {len(ms.cluster_centers_)} puncak kepadatan (bandwidth {bw:.2f})")
save("mean_shift")
