# XGBoost: gradient boosting teregularisasi & sangat dioptimalkan
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.25,
                                          stratify=data.target, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=4,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          early_stopping_rounds=50, eval_metric="logloss", random_state=42)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

proba = model.predict_proba(X_te)[:, 1]
print(f"Versi XGBoost       : {xgb.__version__}")
print(f"Iterasi terbaik     : {model.best_iteration} (early stopping dari maks 1000)")
print(f"Akurasi test        : {accuracy_score(y_te, proba > .5):.3f}")
print(f"ROC AUC test        : {roc_auc_score(y_te, proba):.4f}")
gain = model.get_booster().get_score(importance_type="gain")
top = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:3]
print("Top 3 fitur (gain)  :", [(str(data.feature_names[int(k[1:])]), round(v, 1)) for k, v in top])

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
h = model.evals_result()["validation_0"]["logloss"]
f, ax = fig()
ax.plot(h, color=BLUE, lw=1.6, label="logloss validasi")
ax.axvline(model.best_iteration, color=RED, ls="--", label=f"best iteration = {model.best_iteration}")
ax.set(xlabel="iterasi", ylabel="logloss", title="XGBoost dengan early stopping")
ax.legend()
save("xgboost_example")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
import pandas as pd
from _sampel import KET_BREAST_CANCER, simpan
simpan("xgboost_example", pd.DataFrame(data.data, columns=data.feature_names).assign(diagnosis=data.target), KET_BREAST_CANCER,
       catatan="569 pasien: 212 ganas dan 357 jinak. Hanya 11 kolom pertama yang ditampilkan.")
