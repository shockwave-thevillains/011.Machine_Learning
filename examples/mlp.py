# Multilayer Perceptron + backpropagation: mengenali angka tulisan tangan 8x8 piksel
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)             # 1797 gambar, 64 piksel
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

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
f, ax = fig()
ax.plot(mlp.loss_curve_, color=BLUE, lw=1.8)
ax.scatter([len(mlp.loss_curve_) - 1], [mlp.loss_curve_[-1]], color=RED, zorder=3)
ax.set_yscale("log")
ax.set(xlabel="iterasi (epoch)", ylabel="cross-entropy loss (log)", title="MLP: kurva loss selama backpropagation")
save("mlp")
