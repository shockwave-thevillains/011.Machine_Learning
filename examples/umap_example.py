# UMAP: reduksi dimensi berbasis topologi, cepat & menjaga struktur global
import time
import warnings

import umap
from sklearn.datasets import load_digits
from sklearn.manifold import trustworthiness

warnings.filterwarnings("ignore")
df = load_digits(as_frame=True).frame.rename(columns={"target": "digit"})   # 1797 gambar: 64 kolom piksel + digit
X, y = df.drop(columns="digit").to_numpy(), df["digit"].to_numpy()

t = time.perf_counter()
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
Z = reducer.fit_transform(X)
print(f"Waktu UMAP             : {time.perf_counter() - t:.1f} detik (termasuk kompilasi JIT numba)")
print(f"Trustworthiness UMAP   : {trustworthiness(X, Z, n_neighbors=10):.3f}")

baru = reducer.transform(X[:5])                 # berbeda dari t-SNE: bisa memetakan data baru
print("Data baru bisa diproyeksikan:", baru.astype(float).round(2).tolist()[:2], "...")

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
Z_baru = reducer.transform(X_baru)                  # UMAP bisa langsung memetakan data baru
_, idx = NearestNeighbors(n_neighbors=10).fit(Z).kneighbors(Z_baru)
for nama, z, i in zip(GAMBAR, Z_baru, idx):
    hitung = np.bincount(y[i], minlength=10)
    print(f"gambar '{nama}' -> posisi ({z[0]:.2f}, {z[1]:.2f}) di kelompok digit {hitung.argmax()} ({hitung.max()} dari 10 tetangga)")

# === VISUALISASI ===
from _plot import fig, save
f, ax = fig(5.6, 4)
ax.scatter(*Z.T, c=y, cmap="bwr", s=4)
for d in range(10):
    ax.text(*Z[y == d].mean(0), str(d), fontsize=12, weight="bold", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", lw=.6))
ax.set(title="UMAP: digit tulisan tangan", xticks=[], yticks=[])
save("umap_example")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import KET_DIGITS, simpan
simpan("umap_example", df, KET_DIGITS, catatan="Yang diolah adalah 64 kolom piksel. UMAP tidak memakai kolom digit; kolom ini hanya untuk mewarnai grafik.")
import pandas as pd
simpan("umap_example_baru", pd.DataFrame({n: [r.replace(" ", "·") for r in p] for n, p in GAMBAR.items()}),
       {"nol, satu, tujuh": "Gambar 8×8 baru yang digambar tangan, tidak ada di dataset. # = 16 (hitam), + = 8 (abu-abu), · = 0 (putih)."},
       catatan="Tiap kolom adalah satu gambar; tiap baris tabel adalah satu baris piksel. Sebelum masuk model, gambar diratakan menjadi 64 angka seperti pixel_0_0 … pixel_7_7.", idx=range(8))
