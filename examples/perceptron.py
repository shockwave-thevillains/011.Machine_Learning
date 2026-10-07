# Perceptron Rosenblatt (1958) ditulis dari nol dengan NumPy
import numpy as np
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=[(-1.5, -1.5), (1.5, 1.5)], cluster_std=0.9, random_state=0)
y = np.where(y == 0, -1, 1)                      # label perceptron: -1 / +1

w, b, lr = np.zeros(2), 0.0, 0.1
for epoch in range(1, 51):
    salah = 0
    for xi, yi in zip(X, y):
        if yi * (xi @ w + b) <= 0:             # salah klasifikasi -> perbarui bobot
            w += lr * yi * xi
            b += lr * yi
            salah += 1
    print(f"epoch {epoch:>2}: {salah} kesalahan")
    if salah == 0:
        break

akurasi = np.mean(np.sign(X @ w + b) == y)
print(f"Bobot akhir w = {np.round(w, 3)}, bias b = {b:.2f}")
print(f"Akurasi data latih = {akurasi:.0%} (konvergen di epoch {epoch})")

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
f, ax = fig(5.2, 3.6)
ax.scatter(*X[y == -1].T, s=14, color=BLUE, label="kelas -1")
ax.scatter(*X[y == 1].T, s=14, color=RED, label="kelas +1")
xs = np.array([-5, 5])
ax.plot(xs, -(w[0] * xs + b) / w[1], color="black", lw=1.6, label="batas keputusan")
ax.set(xlim=(-5, 5), ylim=(-5, 5), title="Perceptron: garis pemisah linear")
ax.legend()
save("perceptron")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("perceptron", {"x1": X[:, 0], "x2": X[:, 1], "label": y}, {
    "x1, x2": "Dua koordinat titik (fitur input).",
    "label": "Target: −1 atau +1 (format label perceptron).",
})
