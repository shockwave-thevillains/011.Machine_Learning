# Local Outlier Factor: outlier relatif terhadap kepadatan tetangganya
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

rng = np.random.default_rng(1)
padat = rng.normal([0, 0], 0.3, (200, 2))       # cluster sangat padat
jarang = rng.normal([5, 5], 1.5, (200, 2))      # cluster renggang
outlier = np.array([[1.4, 1.4], [-1.2, 1.3], [0, -1.6]])   # dekat cluster padat, tapi "aneh" secara lokal
X = np.vstack([padat, jarang, outlier])

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
pred_lof = lof.fit_predict(X)
iso = IsolationForest(contamination=0.02, random_state=0).fit(X)
pred_iso = iso.predict(X)

print("Skor LOF (≈1 normal, >1.5 mencurigakan):")
for i, p in enumerate(outlier):
    print(f"  outlier lokal {p.tolist()} -> LOF = {-lof.negative_outlier_factor_[400 + i]:.2f}")
print(f"Outlier lokal tertangkap  LOF = {np.sum(pred_lof[400:] == -1)}/3 | Isolation Forest = {np.sum(pred_iso[400:] == -1)}/3")
print(f"Titik cluster renggang yang dicap outlier  LOF = {np.sum(pred_lof[200:400] == -1)} | IsoForest = {np.sum(pred_iso[200:400] == -1)}")

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
f, ax = fig()
ax.scatter(*X.T, s=7, color=BLUE, alpha=.5)
s = -lof.negative_outlier_factor_
ax.scatter(*X[pred_lof == -1].T, s=40 * s[pred_lof == -1], facecolors="none", edgecolors=RED, lw=1.2, label="outlier LOF (ukuran ~ skor)")
ax.set(title="Local Outlier Factor", aspect="equal")
ax.legend()
save("lof")
