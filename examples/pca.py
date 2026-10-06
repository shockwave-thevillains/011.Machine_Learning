# Principal Component Analysis: memampatkan 64 piksel menjadi beberapa komponen utama
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True)             # 1797 x 64
pca = PCA().fit(X)
kum = np.cumsum(pca.explained_variance_ratio_)

for k in (2, 5, 10, 20, 30):
    print(f"{k:>2} komponen menjelaskan {kum[k - 1]:.1%} varians")
k95 = np.argmax(kum >= 0.95) + 1
print(f"Komponen yang dibutuhkan untuk 95% varians: {k95} dari 64 ({k95 / 64:.0%} ukuran asli)")

p = PCA(n_components=k95).fit(X)
X_rek = p.inverse_transform(p.transform(X))
print(f"Error rekonstruksi rata-rata (MSE piksel, skala 0-16): {np.mean((X - X_rek) ** 2):.3f}")

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
Z = PCA(2).fit_transform(X)
f, axes = fig(7.4, 3.2, ncols=2)
axes[0].plot(range(1, 65), kum, color=BLUE, lw=1.8)
axes[0].axhline(.95, color=RED, ls="--", lw=1)
axes[0].axvline(k95, color=RED, ls="--", lw=1)
axes[0].set(xlabel="jumlah komponen", ylabel="varians kumulatif", title="Scree kumulatif")
sc = axes[1].scatter(*Z.T, c=y, cmap="bwr", s=5)
axes[1].set(title="Proyeksi 2 komponen utama (warna = digit)", xticks=[], yticks=[])
save("pca")
