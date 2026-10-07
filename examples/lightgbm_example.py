# LightGBM: boosting berbasis histogram & leaf-wise growth, cepat untuk data besar
import time

import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=100_000, n_features=40, n_informative=15,
                           random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

t = time.perf_counter()
lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63,
                          random_state=0, verbose=-1).fit(X_tr, y_tr)
t_lgb = time.perf_counter() - t

t = time.perf_counter()
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0).fit(X_tr[:20_000], y_tr[:20_000])
t_gb = time.perf_counter() - t

print(f"Data latih: {X_tr.shape[0]:,} baris x {X_tr.shape[1]} fitur")
print(f"LightGBM (300 pohon, 80.000 baris)       : {t_lgb:5.1f} detik | AUC = {roc_auc_score(y_te, lgbm.predict_proba(X_te)[:, 1]):.4f}")
print(f"sklearn GBM (100 pohon, hanya 20.000 baris): {t_gb:5.1f} detik | AUC = {roc_auc_score(y_te, gb.predict_proba(X_te)[:, 1]):.4f}")
print(f"Jumlah daun per pohon LightGBM (maks)    : {lgbm.get_params()['num_leaves']}")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
baru = pd.DataFrame(X_te[:5], columns=[f"f{i:02d}" for i in range(40)])   # 5 baris yang tidak pernah dilihat saat latihan
t = time.perf_counter()
proba = lgbm.predict_proba(baru.to_numpy())[:, 1]
for i, (p, asli) in enumerate(zip(proba, y_te[:5])):
    print(f"baris {i}: P(kelas 1) = {p:.3f} -> prediksi {int(p >= 0.5)} | label asli {asli}")
t = time.perf_counter()
lgbm.predict(X_te)
print(f"Waktu memprediksi {len(X_te):,} baris sekaligus: {time.perf_counter() - t:.2f} detik")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
import pandas as pd
from _sampel import simpan
simpan("lightgbm_example", pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(40)]).assign(kelas=y), {
    "f00 … f39": "40 fitur numerik sintetis: 15 informatif, 2 kombinasi linear dari fitur informatif, sisanya noise.",
    "kelas": "Target biner 0/1.",
})
simpan("lightgbm_example_baru", baru, {"f00 … f39": "Lima baris dari data uji, yang tidak dipakai saat melatih model."})
