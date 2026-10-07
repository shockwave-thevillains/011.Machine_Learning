# Linear Discriminant Analysis (Fisher, 1936): klasifikasi + reduksi dimensi wine
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

wine = load_wine(as_frame=True)
df = wine.frame.rename(columns={"target": "kultivar"})   # 178 anggur: 13 kolom kimia + kultivar
X, y = df.drop(columns="kultivar").to_numpy(), df["kultivar"].to_numpy()
lda = LinearDiscriminantAnalysis()
print(f"Akurasi 10-fold CV          : {cross_val_score(lda, X, y, cv=10).mean():.3f}")

Z = lda.fit(X, y).transform(X)                   # 13 dimensi -> 2 dimensi diskriminan
print(f"Dimensi sebelum -> sesudah  : {X.shape[1]} -> {Z.shape[1]}")
print(f"Rasio varians antar-kelas   : LD1 = {lda.explained_variance_ratio_[0]:.1%}, LD2 = {lda.explained_variance_ratio_[1]:.1%}")
fitur = sorted(zip(abs(lda.scalings_[:, 0]), wine.feature_names), reverse=True)[:3]
print("Fitur paling berpengaruh di LD1:", [f for _, f in fitur])

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import numpy as np
import pandas as pd
median = [np.median(X[y == k], axis=0) for k in range(3)]
baru = pd.DataFrame([median[0], median[2], (median[1] + median[2]) / 2], columns=wine.feature_names,
                    index=["anggur_1", "anggur_2", "anggur_3"])
proba, koord = lda.predict_proba(baru.to_numpy()), lda.transform(baru.to_numpy())
for nama, pr, kd in zip(baru.index, proba, koord):
    print(f"{nama}: kultivar {pr.argmax()} | P = {np.round(pr, 3).tolist()} | posisi LD1 = {kd[0]:+.2f}, LD2 = {kd[1]:+.2f}")

# === VISUALISASI ===
from _plot import BLUE, RED, BLACK, fig, save
f, ax = fig(5.6, 3.6)
for k, c in zip(range(3), (BLUE, RED, BLACK)):
    ax.scatter(*Z[y == k].T, s=14, color=c, label=f"kultivar {k}")
ax.set(xlabel="LD1", ylabel="LD2", title="LDA: 13 fitur kimia diproyeksikan ke 2D")
ax.legend()
save("lda")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("lda", df, {
    "alcohol … proline": "13 hasil analisis kimia anggur: kadar alkohol, asam malat, abu, magnesium, fenol, flavanoid, intensitas warna, prolin, dll.",
    "kultivar": "Target: jenis kultivar anggur (0, 1, atau 2) dari satu daerah di Italia.",
}, idx=[0, 1, 59, 60, 130, 131], catatan="Baris contoh diambil dari ketiga kultivar.")
simpan("lda_baru", baru.reset_index(names="anggur"), {
    "anggur_1": "Hasil analisis kimia setara median kultivar 0.",
    "anggur_2": "Setara median kultivar 2.",
    "anggur_3": "Campuran: tepat di tengah median kultivar 1 dan 2 (kasus sulit).",
})
