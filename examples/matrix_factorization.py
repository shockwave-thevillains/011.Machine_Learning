# Matrix Factorization (gaya Funk SVD, Netflix Prize): sistem rekomendasi film
import numpy as np
import pandas as pd

df = pd.DataFrame({                              # rating 1-5, 0 = belum menonton
    "pengguna":        ["Andi", "Budi", "Citra", "Dewi", "Eko"],
    "Laskar Pelangi":  [5, 4, 1, 0, 5],
    "Dilan 1990":      [4, 0, 1, 1, 5],
    "Pengabdi Setan":  [0, 1, 5, 4, 0],
    "The Raid":        [1, 2, 4, 5, 0],
    "KKN Desa Penari": [0, 1, 5, 4, 1],
    "Habibie & Ainun": [5, 0, 0, 1, 4],
}).set_index("pengguna")
film, pengguna = list(df.columns), list(df.index)
R = df.to_numpy(dtype=float)
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

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
baru = pd.DataFrame({"film": ["Laskar Pelangi", "The Raid", "Pengabdi Setan"], "rating_fajar": [5, 1, 2]})
Qf = Q[[film.index(f) for f in baru.film]]        # vektor film yang sudah dipelajari tetap dipakai
p_fajar = np.linalg.solve(Qf.T @ Qf + 0.1 * np.eye(k), Qf.T @ (baru.rating_fajar.to_numpy() - mu))
prediksi_fajar = np.clip(mu + Q @ p_fajar, 1, 5)
for f, r in zip(film, prediksi_fajar):
    tanda = "(sudah dinilai)" if f in set(baru.film) else ""
    print(f"Fajar x {f:<16} -> {r:.1f} {tanda}")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("matrix_factorization", df.astype(object).where(df > 0, "–").reset_index(), {
    "pengguna": "Nama pengguna.",
    "Laskar Pelangi … Habibie & Ainun": "Rating pengguna untuk film itu (1–5). Di kode, 0 berarti belum menonton (ditampilkan sebagai –); nilai inilah yang diprediksi.",
}, catatan="Matriks berisi 22 rating diketahui dan 8 sel kosong.")
simpan("matrix_factorization_baru", baru, {"film, rating_fajar": "Fajar adalah pengguna baru yang baru menilai 3 film."})
