# Support Vector Machine: margin maksimum + kernel trick untuk data berbentuk bulan sabit
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

for kernel in ("linear", "poly", "rbf"):
    m = SVC(kernel=kernel, C=1.0, degree=3, gamma="scale").fit(X_tr, y_tr)
    print(f"kernel={kernel:<6} akurasi test = {m.score(X_te, y_te):.3f} | support vectors = {m.n_support_.sum()}")

for C in (0.1, 1, 10, 100):
    m = SVC(kernel="rbf", C=C).fit(X_tr, y_tr)
    print(f"RBF C={C:<5} akurasi train = {m.score(X_tr, y_tr):.3f} | test = {m.score(X_te, y_te):.3f}")

# === VISUALISASI ===
import numpy as np
from _plot import BLUE, RED, fig, save
m = SVC(kernel="rbf", C=10).fit(X_tr, y_tr)
xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 300), np.linspace(-1, 1.5, 300))
Z = m.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
f, ax = fig()
ax.contourf(xx, yy, Z > 0, colors=["#e6e6ff", "#ffe6e6"], alpha=.8)
ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors="black", linestyles=["--", "-", "--"], linewidths=1)
ax.scatter(*X_tr[y_tr == 0].T, s=12, color=BLUE)
ax.scatter(*X_tr[y_tr == 1].T, s=12, color=RED)
ax.scatter(*m.support_vectors_.T, s=40, facecolors="none", edgecolors="black", lw=.6)
ax.set(title="SVM (RBF): batas keputusan & margin")
ax.grid(False)
save("svm")
