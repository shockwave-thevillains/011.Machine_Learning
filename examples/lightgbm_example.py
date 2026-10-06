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
