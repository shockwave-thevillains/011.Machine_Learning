# Matrix Factorization (gaya Funk SVD, Netflix Prize): sistem rekomendasi film
import numpy as np

film = ["Laskar Pelangi", "Dilan 1990", "Pengabdi Setan", "The Raid", "KKN Desa Penari", "Habibie & Ainun"]
pengguna = ["Andi", "Budi", "Citra", "Dewi", "Eko"]
R = np.array([                                   # rating 1-5, 0 = belum menonton
    [5, 4, 0, 1, 0, 5],
    [4, 0, 1, 2, 1, 0],
    [1, 1, 5, 4, 5, 0],
    [0, 1, 4, 5, 4, 1],
    [5, 5, 0, 0, 1, 4],
], dtype=float)
ada = R > 0

rng = np.random.default_rng(0)
k, lr, reg = 2, 0.02, 0.02                       # 2 faktor laten (mis. "drama" vs "horor/aksi")
P = rng.normal(0, 0.1, (5, k))                   # vektor selera pengguna
Q = rng.normal(0, 0.1, (6, k))                   # vektor karakter film
mu = R[ada].mean()
for epoch in range(3000):
    for u, i in zip(*np.nonzero(ada)):
        err = R[u, i] - (mu + P[u] @ Q[i])
        P[u], Q[i] = P[u] + lr * (err * Q[i] - reg * P[u]), Q[i] + lr * (err * P[u] - reg * Q[i])

pred = np.clip(mu + P @ Q.T, 1, 5)
print(f"RMSE pada rating yang diketahui: {np.sqrt(np.mean((pred[ada] - R[ada]) ** 2)):.3f}")
print("Prediksi rating untuk film yang BELUM ditonton:")
for u, i in zip(*np.nonzero(~ada)):
    print(f"  {pengguna[u]:<6} x {film[i]:<16} -> {pred[u, i]:.1f}")
for u in range(5):
    belum = np.flatnonzero(~ada[u])
    terbaik = belum[pred[u, belum].argmax()]
    saran = film[terbaik] if pred[u, terbaik] >= 3.5 else "(tidak ada yang cocok, prediksi < 3.5)"
    print(f"Rekomendasi untuk {pengguna[u]:<6}: {saran}")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
import pandas as pd
from _sampel import simpan
tabel = pd.DataFrame(R.astype(int), columns=film).astype(object).where(R > 0, "–")
tabel.insert(0, "pengguna", pengguna)
simpan("matrix_factorization", tabel, {
    "pengguna": "Nama pengguna.",
    "Laskar Pelangi … Habibie & Ainun": "Rating pengguna untuk film itu (1–5). Tanda – berarti belum menonton; nilai inilah yang diprediksi.",
}, catatan="Matriks berisi 22 rating diketahui dan 8 sel kosong.")
