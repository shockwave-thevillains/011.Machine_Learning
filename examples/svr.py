# Support Vector Regression: regresi non-linear dengan "tabung" toleransi epsilon
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

rng = np.random.default_rng(1)
X = np.sort(rng.uniform(0, 6, 150)).reshape(-1, 1)
y = np.sin(X).ravel() + 0.3 * X.ravel() + rng.normal(0, .15, 150)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1)

for kernel in ("linear", "poly", "rbf"):
    m = SVR(kernel=kernel, C=10, epsilon=0.1, degree=3).fit(X_tr, y_tr)
    print(f"kernel={kernel:<6} R² test = {r2_score(y_te, m.predict(X_te)):.3f} | support vectors = {len(m.support_)}")

svr = SVR(kernel="rbf", C=10, epsilon=0.1).fit(X_tr, y_tr)
print(f"Prediksi x=2.5 -> {svr.predict([[2.5]])[0]:.3f} (nilai sebenarnya tanpa noise = {np.sin(2.5) + .75:.3f})")

# === VISUALISASI ===
from _plot import BLUE, RED, BLACK, fig, save
f, ax = fig()
xs = np.linspace(0, 6, 300).reshape(-1, 1)
p = svr.predict(xs)
ax.scatter(X_tr, y_tr, s=10, color=BLACK, alpha=.45, label="data latih")
ax.scatter(X_tr[svr.support_], y_tr[svr.support_], s=28, facecolors="none", edgecolors=RED, label="support vectors")
ax.plot(xs, p, color=BLUE, lw=2, label="SVR (RBF)")
ax.fill_between(xs.ravel(), p - .1, p + .1, color=BLUE, alpha=.12, label="tabung ε = 0.1")
ax.set(title="Support Vector Regression", xlabel="x", ylabel="y")
ax.legend(ncol=2)
save("svr")
