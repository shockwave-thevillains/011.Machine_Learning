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

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
f, ax = fig()
ax.plot(range(1, 501), mse, color=BLUE, lw=1.6, label="MSE test")
ax.axhline(1.0, color=RED, ls="--", label="batas noise (σ² = 1)")
ax.set_yscale("log")
ax.set(xlabel="jumlah pohon", ylabel="MSE (log)", title="Gradient Boosting: error turun tiap tahap")
ax.legend()
save("gradient_boosting")
