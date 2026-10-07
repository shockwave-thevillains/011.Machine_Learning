# Multilayer Perceptron + backpropagation: mengenali angka tulisan tangan 8x8 piksel
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

df = load_digits(as_frame=True).frame.rename(columns={"target": "digit"})   # 1797 gambar: 64 kolom piksel + digit
X, y = df.drop(columns="digit").to_numpy(), df["digit"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
sc = StandardScaler().fit(X_tr)

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                    max_iter=300, random_state=42).fit(sc.transform(X_tr), y_tr)
pred = mlp.predict(sc.transform(X_te))

print(f"Arsitektur        : 64 -> 64 -> 32 -> 10 ({sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_):,} parameter)")
print(f"Iterasi pelatihan : {mlp.n_iter_} | loss akhir = {mlp.loss_:.4f}")
print(f"Akurasi test      : {accuracy_score(y_te, pred):.3f}")
cm = confusion_matrix(y_te, pred)
salah = [(int(cm[i, j]), i, j) for i in range(10) for j in range(10) if i != j and cm[i, j]]
print("Kesalahan terbanyak (jumlah, asli, prediksi):", sorted(salah, reverse=True)[:3])

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import numpy as np
GAMBAR = {                                       # angka baru yang digambar tangan: # = 16, + = 8, spasi = 0
    "nol":   ["  +##+  ", " +#++#+ ", " ##  ## ", " #+  +# ", " #+  +# ", " ##  ## ", " +#++#+ ", "  +##+  "],
    "satu":  ["   +#+  ", "  +##+  ", " +###+  ", "   ##+  ", "   ##+  ", "   ##+  ", "   ##+  ", "   +#+  "],
    "tujuh": ["  +#####", " ++++##+", "    +#+ ", "  +####+", "  ####+ ", "   +#+  ", "  +#+   ", "  +#    "],
}
def ke_piksel(pola):
    return np.array([[{"#": 16, "+": 8}.get(c, 0) for c in baris] for baris in pola], dtype=float).ravel()
X_baru = np.array([ke_piksel(p) for p in GAMBAR.values()])
for nama, pr in zip(GAMBAR, mlp.predict_proba(sc.transform(X_baru))):
    top = pr.argsort()[::-1][:2]
    print(f"gambar '{nama}' -> prediksi {top[0]} ({pr[top[0]]:.1%}) | kemungkinan kedua: {top[1]} ({pr[top[1]]:.1%})")

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
f, ax = fig()
ax.plot(mlp.loss_curve_, color=BLUE, lw=1.8)
ax.scatter([len(mlp.loss_curve_) - 1], [mlp.loss_curve_[-1]], color=RED, zorder=3)
ax.set_yscale("log")
ax.set(xlabel="iterasi (epoch)", ylabel="cross-entropy loss (log)", title="MLP: kurva loss selama backpropagation")
save("mlp")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import KET_DIGITS, simpan
simpan("mlp", df, KET_DIGITS, catatan="Sebelum masuk jaringan, tiap kolom piksel distandarkan (StandardScaler).")
import pandas as pd
simpan("mlp_baru", pd.DataFrame({n: [r.replace(" ", "·") for r in p] for n, p in GAMBAR.items()}),
       {"nol, satu, tujuh": "Gambar 8×8 baru yang digambar tangan, tidak ada di dataset. # = 16 (hitam), + = 8 (abu-abu), · = 0 (putih)."},
       catatan="Tiap kolom adalah satu gambar; tiap baris tabel adalah satu baris piksel. Sebelum masuk model, gambar diratakan menjadi 64 angka seperti pixel_0_0 … pixel_7_7.", idx=range(8))
