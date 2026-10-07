# Bagging (Bootstrap Aggregating): rata-rata banyak model yang dilatih di sampel bootstrap
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=500, noise=0.35, random_state=0)
pohon = DecisionTreeClassifier(random_state=0)
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, max_samples=0.8,
                        bootstrap=True, oob_score=True, random_state=0)

s1 = cross_val_score(pohon, X, y, cv=10)
s2 = cross_val_score(bag, X, y, cv=10)
print(f"Decision Tree tunggal : akurasi {s1.mean():.3f} ± {s1.std():.3f}")
print(f"Bagging 200 pohon     : akurasi {s2.mean():.3f} ± {s2.std():.3f}")
bag.fit(X, y)
print(f"OOB score bagging     : {bag.oob_score_:.3f}")
unik = np.mean([len(np.unique(i)) / len(X) for i in bag.estimators_samples_])
print(f"Rata-rata porsi data unik per sampel bootstrap: {unik:.1%}")

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
xx, yy = np.meshgrid(np.linspace(-2, 3, 250), np.linspace(-1.5, 2, 250))
g = np.c_[xx.ravel(), yy.ravel()]
f, axes = fig(7.2, 3.2, ncols=2)
for ax, m, t in zip(axes, (pohon.fit(X, y), bag), ("1 Decision Tree", "Bagging 200 pohon")):
    ax.contourf(xx, yy, m.predict(g).reshape(xx.shape), colors=["#e6e6ff", "#ffe6e6"])
    ax.scatter(*X[y == 0].T, s=6, color=BLUE)
    ax.scatter(*X[y == 1].T, s=6, color=RED)
    ax.set(title=t, xticks=[], yticks=[])
save("bagging")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("bagging", {"x1": X[:, 0], "x2": X[:, 1], "kelas": y}, {
    "x1, x2": "Koordinat titik.",
    "kelas": "Target 0/1 (dua bulan sabit dengan noise tinggi, sehingga sebagian titik tumpang tindih).",
})
