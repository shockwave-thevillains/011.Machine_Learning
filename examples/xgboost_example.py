# XGBoost: gradient boosting teregularisasi & sangat dioptimalkan
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

data = load_breast_cancer(as_frame=True)
df = data.frame.rename(columns={"target": "diagnosis"})   # 569 baris: 30 kolom ukuran sel + diagnosis
X, y = df.drop(columns="diagnosis").to_numpy(), df["diagnosis"].to_numpy()   # fitur (X) dan target (y)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
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

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import numpy as np
import pandas as pd
fitur = df.drop(columns="diagnosis")
profil_ganas = fitur[df.diagnosis == 0].median()       # median tiap kolom pada pasien ganas
profil_jinak = fitur[df.diagnosis == 1].median()
baru = pd.DataFrame([profil_jinak, (profil_jinak + profil_ganas) / 2, profil_ganas],
                    index=["pasien_A", "pasien_B", "pasien_C"])
for nama, p in zip(baru.index, model.predict_proba(baru.to_numpy())[:, 1]):
    print(f"{nama}: P(jinak) = {p:.3f} -> {'jinak' if p >= 0.5 else 'GANAS'} (memakai {model.best_iteration + 1} pohon terbaik)")

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
from _sampel import KET_BREAST_CANCER, simpan
simpan("xgboost_example", df, KET_BREAST_CANCER, catatan="569 pasien: 212 ganas dan 357 jinak.")
simpan("xgboost_example_baru", baru.reset_index(names="pasien"), {
    "pasien_A": "Ukuran sel setara median pasien jinak.",
    "pasien_B": "Tepat di tengah antara profil jinak dan ganas (kasus sulit).",
    "pasien_C": "Ukuran sel setara median pasien ganas.",
}, catatan="Tiga pasien baru yang tidak ada di dataset. Kolomnya sama dengan data latih (30 ukuran sel).")
