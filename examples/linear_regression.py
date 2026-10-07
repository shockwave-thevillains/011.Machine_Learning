# Linear Regression: memprediksi harga rumah dari luas bangunan
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
luas = rng.uniform(30, 200, 120).reshape(-1, 1)            # m²
harga = 12.5 * luas.ravel() + 150 + rng.normal(0, 90, 120)  # juta rupiah

X_train, X_test, y_train, y_test = train_test_split(luas, harga, test_size=0.25, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

print(f"Persamaan   : harga = {model.coef_[0]:.2f} x luas + {model.intercept_:.2f}")
print(f"R² (test)   : {r2_score(y_test, pred):.3f}")
print(f"MAE (test)  : Rp {mean_absolute_error(y_test, pred):.1f} juta")
for m2 in (45, 120, 180):
    print(f"Prediksi rumah {m2:>3} m² -> Rp {model.predict([[m2]])[0]:,.0f} juta")

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
f, ax = fig()
ax.scatter(luas, harga, s=12, color=BLUE, alpha=.6, label="data")
xs = np.linspace(30, 200, 2).reshape(-1, 1)
ax.plot(xs, model.predict(xs), color=RED, lw=2, label="garis regresi")
ax.set(xlabel="Luas bangunan (m²)", ylabel="Harga (juta Rp)", title="Linear Regression")
ax.legend()
save("linear_regression")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("linear_regression", {"luas_m2": luas.ravel(), "harga_juta_rp": harga}, {
    "luas_m2": "Luas bangunan dalam m². Fitur input (X).",
    "harga_juta_rp": "Harga rumah dalam juta rupiah. Target (y) yang diprediksi.",
}, catatan="75% baris dipakai untuk melatih model, 25% untuk menguji (train_test_split).")
