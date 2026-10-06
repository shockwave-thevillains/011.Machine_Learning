# CatBoost: boosting yang menangani fitur kategorikal secara native (prediksi churn pelanggan)
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
n = 5000
df = pd.DataFrame({
    "kota": rng.choice(["Jakarta", "Bandung", "Surabaya", "Medan", "Makassar"], n),
    "paket": rng.choice(["Basic", "Standard", "Premium"], n, p=[.5, .3, .2]),
    "metode_bayar": rng.choice(["e-wallet", "transfer", "kartu kredit"], n),
    "lama_langganan_bln": rng.integers(1, 48, n),
    "keluhan_3bln": rng.poisson(1.0, n),
})
logit = (-1.0 + 0.8 * df.keluhan_3bln - 0.05 * df.lama_langganan_bln
         + df.paket.map({"Basic": .9, "Standard": 0, "Premium": -.8})
         + df.metode_bayar.map({"e-wallet": .4, "transfer": 0, "kartu kredit": -.5}))
df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

X, y = df.drop(columns="churn"), df.churn
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
cat_cols = ["kota", "paket", "metode_bayar"]               # TANPA one-hot encoding manual
model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=4, verbose=0,
                           random_seed=0).fit(X_tr, y_tr, cat_features=cat_cols)

proba = model.predict_proba(X_te)[:, 1]
print(f"Churn rate data     : {y.mean():.1%}")
print(f"Akurasi test        : {accuracy_score(y_te, proba > .5):.3f}")
print(f"ROC AUC test        : {roc_auc_score(y_te, proba):.4f}")
print("Feature importance  :")
for nama, imp in sorted(zip(X.columns, model.get_feature_importance()), key=lambda t: -t[1]):
    print(f"  {nama:<20} {imp:5.1f}")
baru = pd.DataFrame([{"kota": "Bandung", "paket": "Basic", "metode_bayar": "e-wallet",
                      "lama_langganan_bln": 3, "keluhan_3bln": 4}])
print(f"Pelanggan baru (Basic, 3 bln, 4 keluhan): P(churn) = {model.predict_proba(baru)[0, 1]:.2f}")
