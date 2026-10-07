# Support Vector Machine: margin maksimum + kernel trick untuk data berbentuk bulan sabit
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "kelas": y})   # 1 baris = 1 titik
X, y = df[["x1", "x2"]].to_numpy(), df["kelas"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

for kernel in ("linear", "poly", "rbf"):
    m = SVC(kernel=kernel, C=1.0, degree=3, gamma="scale").fit(X_tr, y_tr)
    print(f"kernel={kernel:<6} akurasi test = {m.score(X_te, y_te):.3f} | support vectors = {m.n_support_.sum()}")

for C in (0.1, 1, 10, 100):
    m = SVC(kernel="rbf", C=C).fit(X_tr, y_tr)
    print(f"RBF C={C:<5} akurasi train = {m.score(X_tr, y_tr):.3f} | test = {m.score(X_te, y_te):.3f}")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
svm_final = SVC(kernel="rbf", C=10).fit(X_tr, y_tr)
baru = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 0.5], "x2": [1.0, -0.5, 0.3, 0.25]})
for (x1, x2), d in zip(baru.to_numpy(), svm_final.decision_function(baru.to_numpy())):
    print(f"titik ({x1:+.2f}, {x2:+.2f}) -> kelas {int(d > 0)} | skor (jarak ke batas) = {d:+.2f}")

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

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("svm", df, {
    "x1, x2": "Koordinat titik pada bidang 2D.",
    "kelas": "Target: 0 = bulan sabit atas, 1 = bulan sabit bawah. Kedua kelas saling mengait sehingga tidak bisa dipisah garis lurus.",
})
simpan("svm_baru", baru, {"x1, x2": "Koordinat titik baru yang belum diketahui kelasnya."})
