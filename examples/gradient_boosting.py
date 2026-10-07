# Gradient Boosting: tiap pohon baru memperbaiki residual (gradien) pohon sebelumnya
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# benchmark Friedman #1 — dari paper Jerome Friedman sendiri
X, y = make_friedman1(n_samples=2000, noise=1.0, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

lin = LinearRegression().fit(X_tr, y_tr)
gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3,
                                subsample=0.8, random_state=0).fit(X_tr, y_tr)

print(f"Linear Regression  R² = {r2_score(y_te, lin.predict(X_te)):.3f}")
print(f"Gradient Boosting  R² = {r2_score(y_te, gbr.predict(X_te)):.3f}")
mse = [mean_squared_error(y_te, p) for p in gbr.staged_predict(X_te)]
for n in (1, 10, 50, 100, 250, 500):
    print(f"MSE test setelah {n:>3} pohon = {mse[n - 1]:.3f}")
print("Fitur 5-9 adalah noise murni; importance-nya:", [round(float(v), 3) for v in gbr.feature_importances_[5:]])

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import numpy as np
import pandas as pd
rng_baru = np.random.default_rng(5)
baru = pd.DataFrame(rng_baru.random((4, 10)).round(2), columns=[f"x{i:02d}" for i in range(1, 11)])
xb = baru.to_numpy()
benar = 10 * np.sin(np.pi * xb[:, 0] * xb[:, 1]) + 20 * (xb[:, 2] - 0.5) ** 2 + 10 * xb[:, 3] + 5 * xb[:, 4]
for i, (p_gb, p_lin, t) in enumerate(zip(gbr.predict(xb), lin.predict(xb), benar)):
    print(f"baris {i}: Gradient Boosting = {p_gb:6.2f} | Linear = {p_lin:6.2f} | nilai sebenarnya (tanpa noise) = {t:6.2f}")

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
f, ax = fig()
ax.plot(range(1, 501), mse, color=BLUE, lw=1.6, label="MSE test")
ax.axhline(1.0, color=RED, ls="--", label="batas noise (σ² = 1)")
ax.set_yscale("log")
ax.set(xlabel="jumlah pohon", ylabel="MSE (log)", title="Gradient Boosting: error turun tiap tahap")
ax.legend()
save("gradient_boosting")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
import pandas as pd
from _sampel import simpan
simpan("gradient_boosting", pd.DataFrame(X, columns=[f"x{i:02d}" for i in range(1, 11)]).assign(y=y), {
    "x01 … x05": "Fitur yang dipakai rumus target, nilainya 0–1.",
    "x06 … x10": "Fitur noise, tidak berpengaruh ke target.",
    "y": "Target = 10·sin(π·x01·x02) + 20·(x03 − 0,5)² + 10·x04 + 5·x05 + noise.",
})
simpan("gradient_boosting_baru", baru, {"x01 … x10": "Empat baris input baru, nilai acak 0–1. Target belum diketahui."})
