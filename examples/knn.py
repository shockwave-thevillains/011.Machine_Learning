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

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
baru = pd.DataFrame({"sepal_pjg_cm": [5.0, 6.0, 6.9], "sepal_lbr_cm": [3.5, 2.7, 3.1],
                     "petal_pjg_cm": [1.4, 4.4, 5.6], "petal_lbr_cm": [0.2, 1.3, 2.2]}, index=["bunga_1", "bunga_2", "bunga_3"])
for nama, pred, proba in zip(baru.index, model.predict(baru.to_numpy()), model.predict_proba(baru.to_numpy())):
    suara = ", ".join(f"{n} {p:.0%}" for n, p in zip(iris.target_names, proba))
    print(f"{nama}: {iris.target_names[pred]:<10} (suara {k_best} tetangga: {suara})")

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
f, ax = fig()
ks = list(hasil)
ax.plot(ks, list(hasil.values()), color=BLUE, marker="o", ms=3)
ax.scatter([k_best], [hasil[k_best]], color=RED, s=60, zorder=3, label=f"terbaik k={k_best}")
ax.set(xlabel="k (jumlah tetangga)", ylabel="akurasi CV", title="KNN: memilih k")
ax.legend()
save("knn")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import KET_IRIS, iris_df, simpan
simpan("knn", iris_df(), KET_IRIS, idx=[0, 1, 50, 51, 100, 101],
       catatan="Dataset diurutkan per spesies (50 bunga per spesies), jadi baris contoh diambil dari ketiganya. Fitur distandarkan (StandardScaler) sebelum jarak dihitung.")
simpan("knn_baru", baru.reset_index(names="bunga"), {"bunga_1 … bunga_3": "Tiga bunga baru yang diukur, spesiesnya belum diketahui."})
