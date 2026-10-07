# Perceptron Rosenblatt (1958) ditulis dari nol dengan NumPy
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=[(-1.5, -1.5), (1.5, 1.5)], cluster_std=0.9, random_state=0)
df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": np.where(y == 0, -1, 1)})   # label perceptron: -1 / +1
X, y = df[["x1", "x2"]].to_numpy(), df["label"].to_numpy()

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

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
baru = pd.DataFrame({"x1": [2.0, -1.5, 0.5, -0.2], "x2": [1.5, -2.0, -0.3, 0.4]})
for x1, x2 in baru.to_numpy():
    skor = w @ [x1, x2] + b
    print(f"titik ({x1:+.1f}, {x2:+.1f}) -> skor w·x + b = {skor:+.3f} -> kelas {'+1' if skor > 0 else '-1'}")

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
simpan("perceptron", df, {
    "x1, x2": "Dua koordinat titik (fitur input).",
    "label": "Target: −1 atau +1 (format label perceptron).",
})
simpan("perceptron_baru", baru, {"x1, x2": "Koordinat titik baru yang belum diketahui kelasnya."})
