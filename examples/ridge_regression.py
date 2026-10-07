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
