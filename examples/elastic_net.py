# Elastic Net: gabungan L1 + L2, stabil saat fitur saling berkorelasi
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(3)
n = 300
dasar = rng.normal(size=(n, 3))
# tiap faktor dasar punya 4 "kembaran" yang hampir identik -> 12 fitur berkorelasi + 18 fitur noise
grup = np.hstack([dasar[:, [i]] + rng.normal(0, .01, (n, 4)) for i in range(3)])
X = np.hstack([grup, rng.normal(size=(n, 18))])
y = 3 * dasar[:, 0] - 2 * dasar[:, 1] + 1.5 * dasar[:, 2] + rng.normal(0, .5, n)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=3)

lasso = Lasso(alpha=0.1, max_iter=50_000).fit(X_tr, y_tr)
enet = ElasticNet(alpha=0.1, l1_ratio=0.3, max_iter=50_000).fit(X_tr, y_tr)

np.set_printoptions(precision=2, suppress=True)
print("Koefisien grup A (4 fitur kembar, bobot asli total = 3):")
print("  Lasso      :", lasso.coef_[:4], "-> total", round(lasso.coef_[:4].sum(), 2))
print("  ElasticNet :", enet.coef_[:4], "-> total", round(enet.coef_[:4].sum(), 2))
print(f"R² test  Lasso = {r2_score(y_te, lasso.predict(X_te)):.4f} | ElasticNet = {r2_score(y_te, enet.predict(X_te)):.4f}")
print("Lasso membuang", np.sum(lasso.coef_[:12] == 0), "fitur kembar secara acak; ElasticNet membagi bobot merata.")

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
f, ax = fig()
idx = np.arange(30)
ax.bar(idx - .2, lasso.coef_, .4, color=BLUE, label="Lasso")
ax.bar(idx + .2, enet.coef_, .4, color=RED, label="Elastic Net")
ax.axvspan(-.5, 11.5, color="#eef", zorder=0)
ax.set(xlabel="indeks fitur (area biru = 3 grup fitur kembar)", ylabel="koefisien", title="Elastic Net membagi bobot ke fitur yang berkorelasi")
ax.legend()
save("elastic_net")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
import pandas as pd
from _sampel import simpan
nama_kolom = [f"{g}{i}" for g in "ABC" for i in range(1, 5)] + [f"noise_{i:02d}" for i in range(1, 19)]
simpan("elastic_net", pd.DataFrame(X, columns=nama_kolom).assign(y=y), {
    "A1 … A4": "Empat fitur \"kembar\" yang hampir identik (berasal dari faktor dasar A). Begitu juga B1 … B4 dan C1 … C4.",
    "noise_01 … noise_18": "18 fitur acak yang sama sekali tidak berhubungan dengan y.",
    "y": "Target = 3·A − 2·B + 1,5·C + noise.",
})
