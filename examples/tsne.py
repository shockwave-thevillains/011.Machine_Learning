# t-SNE: visualisasi data berdimensi tinggi ke 2D dengan menjaga tetangga lokal
import time

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE, trustworthiness
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True)

t = time.perf_counter()
tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0)
Z = tsne.fit_transform(X)
print(f"Waktu t-SNE            : {time.perf_counter() - t:.1f} detik untuk {len(X)} titik")
print(f"KL divergence akhir    : {tsne.kl_divergence_:.3f}")
print(f"Trustworthiness t-SNE  : {trustworthiness(X, Z, n_neighbors=10):.3f}")
print(f"Trustworthiness PCA-2D : {trustworthiness(X, PCA(2).fit_transform(X), n_neighbors=10):.3f}")
print("(trustworthiness 1.0 = tetangga terdekat di 2D sama persis dengan di 64D)")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import numpy as np
from sklearn.neighbors import NearestNeighbors
GAMBAR = {                                       # angka baru yang digambar tangan: # = 16, + = 8, spasi = 0
    "nol":   ["  +##+  ", " +#++#+ ", " ##  ## ", " #+  +# ", " #+  +# ", " ##  ## ", " +#++#+ ", "  +##+  "],
    "satu":  ["   +#+  ", "  +##+  ", " +###+  ", "   ##+  ", "   ##+  ", "   ##+  ", "   ##+  ", "   +#+  "],
    "tujuh": ["  +#####", " ++++##+", "    +#+ ", "  +####+", "  ####+ ", "   +#+  ", "  +#+   ", "  +#    "],
}
def ke_piksel(pola):
    return np.array([[{"#": 16, "+": 8}.get(c, 0) for c in baris] for baris in pola], dtype=float).ravel()
X_baru = np.array([ke_piksel(p) for p in GAMBAR.values()])
# t-SNE tidak punya transform(): jalankan ulang pada data lama + data baru
Z_gabung = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(np.vstack([X, X_baru]))
_, idx = NearestNeighbors(n_neighbors=10).fit(Z_gabung[:len(X)]).kneighbors(Z_gabung[len(X):])
for nama, i in zip(GAMBAR, idx):
    hitung = np.bincount(y[i], minlength=10)
    print(f"gambar '{nama}' mendarat di kelompok digit {hitung.argmax()} ({hitung.max()} dari 10 tetangga terdekat di peta 2D)")

# === VISUALISASI ===
from _plot import fig, save
f, ax = fig(5.6, 4)
ax.scatter(*Z.T, c=y, cmap="bwr", s=4)
for d in range(10):
    ax.text(*Z[y == d].mean(0), str(d), fontsize=12, weight="bold", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", lw=.6))
ax.set(title="t-SNE: 1.797 digit tulisan tangan", xticks=[], yticks=[])
save("tsne")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import KET_DIGITS, digits_df, simpan
simpan("tsne", digits_df(X, y), KET_DIGITS, catatan="Yang diolah adalah 64 kolom piksel. t-SNE tidak memakai kolom digit; kolom ini hanya untuk mewarnai grafik.")
import pandas as pd
simpan("tsne_baru", pd.DataFrame({n: [r.replace(" ", "·") for r in p] for n, p in GAMBAR.items()}),
       {"nol, satu, tujuh": "Gambar 8×8 baru yang digambar tangan, tidak ada di dataset. # = 16 (hitam), + = 8 (abu-abu), · = 0 (putih)."},
       catatan="Tiap kolom adalah satu gambar; tiap baris tabel adalah satu baris piksel. Sebelum masuk model, gambar diratakan menjadi 64 angka seperti px_00 … px_63.", idx=range(8))
