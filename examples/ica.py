# Independent Component Analysis: memisahkan suara yang tercampur ("cocktail party problem")
import numpy as np
from sklearn.decomposition import FastICA, PCA

t = np.linspace(0, 8, 2000)
s1 = np.sin(2 * t)                              # sumber 1: nada sinus
s2 = np.sign(np.sin(3 * t))                     # sumber 2: gelombang kotak
s3 = 2 * ((t * 1.3) % 1) - 1                    # sumber 3: gigi gergaji
S = np.c_[s1, s2, s3] + 0.02 * np.random.default_rng(0).normal(size=(2000, 3))

A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])   # matriks pencampur (3 mikrofon)
X = S @ A.T

S_ica = FastICA(n_components=3, whiten="unit-variance", random_state=0).fit_transform(X)
S_pca = PCA(n_components=3).fit_transform(X)

def cocok(est):  # korelasi absolut terbaik antara sinyal asli dan hasil pemisahan
    C = np.abs(np.corrcoef(S.T, est.T)[:3, 3:])
    return C.max(axis=1)

print("Korelasi |r| sumber asli vs hasil pemisahan (1.0 = sempurna):")
for nama, est in (("FastICA", S_ica), ("PCA", S_pca)):
    print(f"  {nama:<8} sinus={cocok(est)[0]:.3f}  kotak={cocok(est)[1]:.3f}  gergaji={cocok(est)[2]:.3f}")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
ica = FastICA(n_components=3, whiten="unit-variance", random_state=0).fit(X)   # model yang sama seperti di atas
t_baru = np.linspace(8, 10, 500)                  # rekaman baru: 2 detik berikutnya
S_baru = np.c_[np.sin(2 * t_baru), np.sign(np.sin(3 * t_baru)), 2 * ((t_baru * 1.3) % 1) - 1]
X_rekaman = S_baru @ A.T
baru = pd.DataFrame({"waktu": t_baru, "mik_1": X_rekaman[:, 0], "mik_2": X_rekaman[:, 1], "mik_3": X_rekaman[:, 2]})
S_pisah = ica.transform(X_rekaman)                # pakai matriks pemisah yang sudah dipelajari
C = np.abs(np.corrcoef(S_baru.T, S_pisah.T)[:3, 3:])
for nama, r in zip(["sinus", "kotak", "gergaji"], C.max(1)):
    print(f"rekaman baru, sumber {nama:<8}: |r| = {r:.3f}")

# === VISUALISASI ===
from _plot import BLUE, RED, BLACK, fig, save
f, axes = fig(7.2, 4.2, nrows=3, sharex=True)
cols = [BLUE, RED, BLACK]
for ax, data, judul in zip(axes, (S, X, S_ica), ("Sumber asli", "Rekaman 3 mikrofon (tercampur)", "Hasil FastICA")):
    for i in range(3):
        ax.plot(t[:700], data[:700, i] / np.abs(data[:, i]).max() + 2.4 * i, color=cols[i], lw=1)
    ax.set(title=judul, yticks=[])
save("ica")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("ica", {"waktu": t, "mik_1": X[:, 0], "mik_2": X[:, 1], "mik_3": X[:, 2]}, {
    "waktu": "Waktu sampel (detik simulasi).",
    "mik_1, mik_2, mik_3": "Sinyal yang terekam tiap mikrofon, yaitu campuran ketiga sumber suara dengan proporsi berbeda.",
}, idx=[0, 1, 2, 500, 1000, 1500], catatan="ICA hanya menerima kolom mik_1–mik_3. Sumber asli (sinus, kotak, gergaji) disimpan terpisah untuk menghitung korelasi.")
simpan("ica_baru", baru, {"waktu": "Detik ke-8 sampai ke-10, setelah rekaman latih berakhir.",
                          "mik_1, mik_2, mik_3": "Rekaman baru dari tiga mikrofon yang sama."}, idx=[0, 1, 2, 250, 251, 499])
