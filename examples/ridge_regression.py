# Ridge Regression: regularisasi L2 menahan overfitting pada fitur yang banyak & berkorelasi
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import numpy as np

X, y = load_diabetes(return_X_y=True)           # 442 pasien, 10 fitur medis
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

def buat(model):  # 10 fitur -> 65 fitur (interaksi & kuadrat)
    return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), model)

ols = buat(LinearRegression()).fit(X_tr, y_tr)
print(f"OLS (tanpa regularisasi) R² test = {r2_score(y_te, ols.predict(X_te)):.3f}"
      f" | norma koefisien = {np.linalg.norm(ols[-1].coef_):,.0f}")
for alpha in (0.1, 1, 10, 100, 1000):
    r = buat(Ridge(alpha=alpha)).fit(X_tr, y_tr)
    print(f"Ridge alpha={alpha:<6} R² test = {r2_score(y_te, r.predict(X_te)):.3f}"
          f" | norma koefisien = {np.linalg.norm(r[-1].coef_):,.0f}")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
mentah = load_diabetes(scaled=False).data           # data asli dalam satuan klinis
mu, skala = mentah.mean(0), np.sqrt(((mentah - mentah.mean(0)) ** 2).sum(0))
baru = pd.DataFrame({
    "age": [30, 50, 65], "sex": [1, 2, 1], "bmi": [21.0, 27.5, 35.0], "bp": [78, 95, 115],
    "s1": [160, 190, 240], "s2": [90, 115, 160], "s3": [65, 48, 32], "s4": [2.5, 4.0, 7.0],
    "s5": [4.0, 4.6, 5.5], "s6": [80, 92, 115]}, index=["pasien_1", "pasien_2", "pasien_3"])
X_baru = (baru.to_numpy() - mu) / skala             # samakan skala dengan data latih
ridge = buat(Ridge(alpha=100)).fit(X_tr, y_tr)
for nama, p_r, p_o in zip(baru.index, ridge.predict(X_baru), ols.predict(X_baru)):
    print(f"{nama}: progresi menurut Ridge = {p_r:6.1f} | OLS = {p_o:7.1f}")
print(f"(target di data latih berkisar {y.min():.0f}–{y.max():.0f}, rata-rata {y.mean():.0f})")

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
alphas = np.logspace(-2, 4, 40)
coefs = [buat(Ridge(alpha=a)).fit(X_tr, y_tr)[-1].coef_ for a in alphas]
f, ax = fig()
ax.plot(alphas, coefs, lw=.8, color=BLUE, alpha=.55)
ax.axvline(100, color=RED, ls="--", lw=1.2)
ax.set_xscale("log")
ax.set(xlabel="alpha (kekuatan regularisasi)", ylabel="nilai koefisien", title="Ridge: koefisien menyusut saat alpha naik")
save("ridge_regression")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
import pandas as pd
from _sampel import simpan
simpan("ridge_regression", pd.DataFrame(X, columns=load_diabetes().feature_names).assign(progresi_penyakit=y), {
    "age, sex": "Usia dan jenis kelamin pasien.",
    "bmi, bp": "Indeks massa tubuh dan tekanan darah rata-rata.",
    "s1 … s6": "Enam hasil tes darah: kolesterol total, LDL, HDL, rasio kolesterol/HDL, log trigliserida, gula darah.",
    "progresi_penyakit": "Target. Ukuran perkembangan diabetes satu tahun setelah pemeriksaan awal (angka lebih besar = lebih parah).",
}, catatan="Sepuluh fitur sudah dinormalisasi oleh scikit-learn (dikurangi rata-rata lalu diskalakan), "
           "jadi nilainya kecil dan bisa negatif. Di pipeline, 10 fitur ini diperluas menjadi 65 fitur polinomial.")
simpan("ridge_regression_baru", baru.reset_index(names="pasien"), {
    "age, sex": "Usia (tahun) dan jenis kelamin (kode 1/2 dari dataset asli).",
    "bmi, bp": "Indeks massa tubuh dan tekanan darah rata-rata (mmHg).",
    "s1 … s6": "Kolesterol total, LDL, HDL (mg/dL), rasio kolesterol/HDL, log trigliserida, gula darah (mg/dL).",
}, catatan="Pasien baru ditulis dalam satuan klinis asli, lalu di kode diskalakan dengan cara yang sama seperti data latih.")
