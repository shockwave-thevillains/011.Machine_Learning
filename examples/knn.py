# K-Nearest Neighbors: klasifikasi bunga iris berdasarkan tetangga terdekat
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
iris = load_iris()
hasil = {}
for k in range(1, 31):
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    hasil[k] = cross_val_score(model, X, y, cv=10).mean()
for k in (1, 3, 5, 10, 15, 20, 30):
    print(f"k = {k:>2} -> akurasi 10-fold CV = {hasil[k]:.3f}")
k_best = max(hasil, key=hasil.get)
print(f"k terbaik = {k_best} (akurasi {hasil[k_best]:.3f})")

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k_best)).fit(X, y)
bunga_baru = [[5.9, 3.0, 5.1, 1.8]]            # panjang/lebar sepal & petal (cm)
dist, idx = model[-1].kneighbors(model[0].transform(bunga_baru))
print("Bunga baru diprediksi:", iris.target_names[model.predict(bunga_baru)[0]])
print("Label tetangga terdekatnya:", [str(iris.target_names[y[i]]) for i in idx[0]])

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
f, ax = fig()
ks = list(hasil)
ax.plot(ks, list(hasil.values()), color=BLUE, marker="o", ms=3)
ax.scatter([k_best], [hasil[k_best]], color=RED, s=60, zorder=3, label=f"terbaik k={k_best}")
ax.set(xlabel="k (jumlah tetangga)", ylabel="akurasi CV", title="KNN: memilih k")
ax.legend()
save("knn")
