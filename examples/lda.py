# Linear Discriminant Analysis (Fisher, 1936): klasifikasi + reduksi dimensi wine
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

wine = load_wine()                               # 178 sampel, 13 fitur kimia, 3 kultivar
X, y = wine.data, wine.target
lda = LinearDiscriminantAnalysis()
print(f"Akurasi 10-fold CV          : {cross_val_score(lda, X, y, cv=10).mean():.3f}")

Z = lda.fit(X, y).transform(X)                   # 13 dimensi -> 2 dimensi diskriminan
print(f"Dimensi sebelum -> sesudah  : {X.shape[1]} -> {Z.shape[1]}")
print(f"Rasio varians antar-kelas   : LD1 = {lda.explained_variance_ratio_[0]:.1%}, LD2 = {lda.explained_variance_ratio_[1]:.1%}")
fitur = sorted(zip(abs(lda.scalings_[:, 0]), wine.feature_names), reverse=True)[:3]
print("Fitur paling berpengaruh di LD1:", [f for _, f in fitur])

# === VISUALISASI ===
from _plot import BLUE, RED, BLACK, fig, save
f, ax = fig(5.6, 3.6)
for k, c in zip(range(3), (BLUE, RED, BLACK)):
    ax.scatter(*Z[y == k].T, s=14, color=c, label=f"kultivar {k}")
ax.set(xlabel="LD1", ylabel="LD2", title="LDA: 13 fitur kimia diproyeksikan ke 2D")
ax.legend()
save("lda")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
import pandas as pd
from _sampel import simpan
simpan("lda", pd.DataFrame(X, columns=wine.feature_names).assign(kultivar=y), {
    "alcohol … proline": "13 hasil analisis kimia anggur: kadar alkohol, asam malat, abu, magnesium, fenol, flavanoid, intensitas warna, prolin, dll.",
    "kultivar": "Target: jenis kultivar anggur (0, 1, atau 2) dari satu daerah di Italia.",
}, idx=[0, 1, 59, 60, 130, 131], catatan="Baris contoh diambil dari ketiga kultivar.")
