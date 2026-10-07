# Lasso Regression: regularisasi L1 memilih fitur penting secara otomatis
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 20 fitur, tetapi hanya 5 yang benar-benar berpengaruh
X, y, coef_asli = make_regression(n_samples=200, n_features=20, n_informative=5,
                                  noise=10, coef=True, random_state=7)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=7)

ols = LinearRegression().fit(X_tr, y_tr)
lasso = Lasso(alpha=3.0).fit(X_tr, y_tr)

print("Fitur yang benar-benar berpengaruh :", np.flatnonzero(coef_asli).tolist())
print("Fitur yang dipilih Lasso           :", np.flatnonzero(lasso.coef_).tolist())
print(f"Koefisien bukan-nol  OLS = {np.sum(ols.coef_ != 0)} | Lasso = {np.sum(lasso.coef_ != 0)}")
print(f"R² test              OLS = {r2_score(y_te, ols.predict(X_te)):.4f} | Lasso = {r2_score(y_te, lasso.predict(X_te)):.4f}")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
rng_baru = np.random.default_rng(99)
a = rng_baru.normal(size=20)
b, c = a.copy(), a.copy()
b[0] += 3                                         # B: fitur_00 (TIDAK dipilih Lasso) diubah besar
c[3] += 1                                         # C: fitur_03 (dipilih Lasso) diubah sedikit
baru = pd.DataFrame([a, b, c], columns=[f"fitur_{i:02d}" for i in range(20)], index=["A", "B", "C"])
benar = baru.to_numpy() @ coef_asli
for nama, p_ols, p_lasso, t in zip(baru.index, ols.predict(baru.to_numpy()), lasso.predict(baru.to_numpy()), benar):
    print(f"sampel {nama}: OLS = {p_ols:8.2f} | Lasso = {p_lasso:8.2f} | nilai sebenarnya (tanpa noise) = {t:8.2f}")

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
f, ax = fig()
idx = np.arange(20)
ax.bar(idx - .2, ols.coef_, .4, color=BLUE, label="OLS")
ax.bar(idx + .2, lasso.coef_, .4, color=RED, label="Lasso")
ax.set(xlabel="indeks fitur", ylabel="koefisien", title="Lasso membuat koefisien tak penting = 0", xticks=idx)
ax.legend()
save("lasso_regression")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
import pandas as pd
from _sampel import simpan
simpan("lasso_regression", pd.DataFrame(X, columns=[f"fitur_{i:02d}" for i in range(20)]).assign(y=y), {
    "fitur_00 … fitur_19": "20 fitur numerik sintetis berdistribusi normal. Hanya fitur_03, 04, 10, 11, dan 12 yang benar-benar memengaruhi y.",
    "y": "Target numerik (kombinasi linear 5 fitur informatif + noise).",
})
simpan("lasso_regression_baru", baru.reset_index(names="sampel"), {
    "sampel A": "Sampel baru acak.",
    "sampel B": "Sama dengan A, tetapi fitur_00 (fitur yang dibuang Lasso) dinaikkan 3.",
    "sampel C": "Sama dengan A, tetapi fitur_03 (fitur yang dipilih Lasso) dinaikkan 1.",
})
