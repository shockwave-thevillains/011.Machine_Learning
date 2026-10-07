# Polynomial Regression: hubungan non-linear suhu vs penjualan es teh (30 hari data)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.default_rng(4)
df = pd.DataFrame({"suhu_c": np.sort(rng.uniform(18, 38, 30))})                 # 1 baris = 1 hari
df["gelas_terjual"] = 0.9 * (df["suhu_c"] - 18) ** 2 - 4 * (df["suhu_c"] - 18) + 40 + rng.normal(0, 18, 30)
X, y = df[["suhu_c"]].to_numpy(), df["gelas_terjual"].to_numpy()

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1)
for derajat in (1, 2, 3, 15):
    model = make_pipeline(StandardScaler(), PolynomialFeatures(derajat), LinearRegression()).fit(X_tr, y_tr)
    print(f"derajat {derajat:>2} | R² train = {r2_score(y_tr, model.predict(X_tr)):.3f}"
          f" | R² test = {r2_score(y_te, model.predict(X_te)):.3f}")

terbaik = make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression()).fit(X_tr, y_tr)
print(f"Prediksi penjualan saat 35°C (derajat 2): {terbaik.predict([[35]])[0]:.0f} gelas")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
baru = pd.DataFrame({"suhu_c": [19.0, 27.5, 33.0, 37.5]})
m2 = make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression()).fit(X_tr, y_tr)
m15 = make_pipeline(StandardScaler(), PolynomialFeatures(15), LinearRegression()).fit(X_tr, y_tr)
for s, a, b in zip(baru.suhu_c, m2.predict(baru.to_numpy()), m15.predict(baru.to_numpy())):
    print(f"{s:4.1f}°C -> derajat 2: {a:5.0f} gelas | derajat 15: {b:8.0f} gelas")

# === VISUALISASI ===
from _plot import BLUE, RED, BLACK, fig, save
f, ax = fig()
ax.scatter(df["suhu_c"], df["gelas_terjual"], s=14, color=BLACK, alpha=.5, label="data")
xs = np.linspace(18, 38, 300).reshape(-1, 1)
for d, c, ls in ((1, BLUE, "--"), (2, RED, "-"), (15, BLUE, ":")):
    m = make_pipeline(StandardScaler(), PolynomialFeatures(d), LinearRegression()).fit(X_tr, y_tr)
    ax.plot(xs, np.clip(m.predict(xs), -50, 600), color=c, ls=ls, lw=1.8, label=f"derajat {d}")
ax.set(xlabel="Suhu (°C)", ylabel="Gelas terjual", title="Polynomial Regression", ylim=(0, 520))
ax.legend()
save("polynomial_regression")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("polynomial_regression", df, {
    "suhu_c": "Suhu harian (°C). Fitur input; di dalam pipeline diperluas menjadi suhu, suhu², suhu³, … sesuai derajat.",
    "gelas_terjual": "Jumlah gelas es teh terjual hari itu. Target.",
}, catatan="Baris sudah diurutkan berdasarkan suhu. 70% dipakai latih, 30% uji.")
simpan("polynomial_regression_baru", baru, {"suhu_c": "Ramalan suhu hari-hari berikutnya; penjualan belum diketahui."})
