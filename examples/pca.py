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

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import numpy as np
GAMBAR = {                                       # angka baru yang digambar tangan: # = 16, + = 8, spasi = 0
    "nol":   ["  +##+  ", " +#++#+ ", " ##  ## ", " #+  +# ", " #+  +# ", " ##  ## ", " +#++#+ ", "  +##+  "],
    "satu":  ["   +#+  ", "  +##+  ", " +###+  ", "   ##+  ", "   ##+  ", "   ##+  ", "   ##+  ", "   +#+  "],
    "tujuh": ["  +#####", " ++++##+", "    +#+ ", "  +####+", "  ####+ ", "   +#+  ", "  +#+   ", "  +#    "],
}
GAMBAR["kotak-kotak"] = ["# # # # ", " # # # #"] * 4  # pola papan catur: BUKAN angka
def ke_piksel(pola):
    return np.array([[{"#": 16, "+": 8}.get(c, 0) for c in baris] for baris in pola], dtype=float).ravel()
X_baru = np.array([ke_piksel(p) for p in GAMBAR.values()])
Z_baru = p.transform(X_baru)
error = ((X_baru - p.inverse_transform(Z_baru)) ** 2).mean(1)
batas = np.percentile(((X - X_rek) ** 2).mean(1), 99)
for nama, e, z in zip(GAMBAR, error, Z_baru):
    print(f"gambar '{nama}': PC1 = {z[0]:+6.1f}, PC2 = {z[1]:+6.1f} | error rekonstruksi {e:5.2f} -> {'mirip angka' if e <= batas else 'TIDAK mirip angka'}")
print(f"(batas: 99% gambar di dataset punya error ≤ {batas:.2f})")

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

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import KET_DIGITS, digits_df, simpan
simpan("pca", digits_df(X, y), KET_DIGITS, catatan="Yang diolah adalah 64 kolom piksel.")
import pandas as pd
simpan("pca_baru", pd.DataFrame({n: [r.replace(" ", "·") for r in p] for n, p in GAMBAR.items()}),
       {"nol, satu, tujuh, kotak-kotak": "Gambar 8×8 baru yang digambar tangan, tidak ada di dataset. # = 16 (hitam), + = 8 (abu-abu), · = 0 (putih)."},
       catatan="Tiap kolom adalah satu gambar; tiap baris tabel adalah satu baris piksel. Sebelum masuk model, gambar diratakan menjadi 64 angka seperti px_00 … px_63.", idx=range(8))
