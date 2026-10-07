# AdaBoost: menggabungkan "weak learner" (stump 1 level) secara berurutan
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# dataset klasik dari buku "Elements of Statistical Learning" (Hastie dkk.)
X, y = make_hastie_10_2(n_samples=12000, random_state=1)
X_tr, y_tr, X_te, y_te = X[:2000], y[:2000], X[2000:], y[2000:]

stump = DecisionTreeClassifier(max_depth=1).fit(X_tr, y_tr)
pohon = DecisionTreeClassifier(max_depth=9, random_state=0).fit(X_tr, y_tr)
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=400,
                         learning_rate=1.0, random_state=0).fit(X_tr, y_tr)

print(f"Error 1 stump (weak learner) : {1 - stump.score(X_te, y_te):.3f}")
print(f"Error pohon kedalaman 9      : {1 - pohon.score(X_te, y_te):.3f}")
err = [1 - (p == y_te).mean() for p in ada.staged_predict(X_te)]
for n in (1, 10, 50, 100, 200, 400):
    print(f"Error AdaBoost {n:>3} stump    : {err[n - 1]:.3f}")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import numpy as np
import pandas as pd
baru = pd.DataFrame(np.vstack([np.full(10, 0.3), np.full(10, 1.5), np.full(10, 0.97), np.r_[3.2, np.zeros(9)]]),
                    columns=[f"x{i:02d}" for i in range(1, 11)], index=["A", "B", "C", "D"])
for nama, xb, pred, skor in zip(baru.index, baru.to_numpy(), ada.predict(baru.to_numpy()), ada.decision_function(baru.to_numpy())):
    aturan = 1 if (xb ** 2).sum() > 9.34 else -1
    print(f"sampel {nama}: Σx² = {(xb ** 2).sum():5.2f} -> prediksi {pred:+.0f} (skor {skor:+.3f}) | jawaban sebenarnya {aturan:+d}")

# === VISUALISASI ===
from _plot import BLUE, RED, BLACK, fig, save
f, ax = fig()
ax.plot(range(1, 401), err, color=BLUE, lw=1.6, label="AdaBoost (stump)")
ax.axhline(1 - stump.score(X_te, y_te), color=RED, ls="--", label="1 stump")
ax.axhline(1 - pohon.score(X_te, y_te), color=BLACK, ls=":", label="pohon kedalaman 9")
ax.set(xlabel="jumlah iterasi boosting", ylabel="error test", title="AdaBoost: weak learner menjadi strong learner")
ax.legend()
save("adaboost")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
import pandas as pd
from _sampel import simpan
simpan("adaboost", pd.DataFrame(X, columns=[f"x{i:02d}" for i in range(1, 11)]).assign(y=y), {
    "x01 … x10": "10 fitur acak berdistribusi normal standar.",
    "y": "Target: +1 jika jumlah kuadrat ke-10 fitur > 9,34, selain itu −1.",
}, catatan="2.000 baris pertama untuk latih, 10.000 sisanya untuk uji.")
simpan("adaboost_baru", baru.reset_index(names="sampel"), {
    "sampel A, B, C": "Kesepuluh fitur bernilai sama: 0,3 (jelas di dalam), 1,5 (jelas di luar), 0,97 (tepat di dekat batas).",
    "sampel D": "Hanya x01 yang besar (3,2), fitur lain 0. Σx² = 10,24 sedikit di atas batas 9,34.",
})
