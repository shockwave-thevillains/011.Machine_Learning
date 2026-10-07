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

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
baru = pd.DataFrame([
    {"kota": "Jakarta", "paket": "Premium", "metode_bayar": "kartu kredit", "lama_langganan_bln": 40, "keluhan_3bln": 0},
    {"kota": "Surabaya", "paket": "Standard", "metode_bayar": "transfer", "lama_langganan_bln": 12, "keluhan_3bln": 2},
    {"kota": "Medan", "paket": "Basic", "metode_bayar": "e-wallet", "lama_langganan_bln": 2, "keluhan_3bln": 3},
    {"kota": "Yogyakarta", "paket": "Basic", "metode_bayar": "transfer", "lama_langganan_bln": 6, "keluhan_3bln": 1},
], index=["pelanggan_1", "pelanggan_2", "pelanggan_3", "pelanggan_4"])
for nama, p in zip(baru.index, model.predict_proba(baru)[:, 1]):
    print(f"{nama}: P(churn) = {p:.2f} -> {'BERISIKO churn' if p >= 0.5 else 'kemungkinan bertahan'}")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("catboost_example", df, {
    "kota": "Kota pelanggan (kategorikal, 5 nilai).",
    "paket": "Paket langganan: Basic, Standard, Premium (kategorikal).",
    "metode_bayar": "e-wallet, transfer, atau kartu kredit (kategorikal).",
    "lama_langganan_bln": "Sudah berapa bulan berlangganan.",
    "keluhan_3bln": "Jumlah keluhan dalam 3 bulan terakhir.",
    "churn": "Target: 1 = berhenti berlangganan, 0 = tetap.",
}, catatan="Tiga kolom teks diberikan apa adanya ke CatBoost lewat cat_features, tanpa one-hot encoding.")
simpan("catboost_example_baru", baru.reset_index(names="pelanggan"), {
    "pelanggan_1 … pelanggan_4": "Pelanggan aktif yang ingin dinilai risikonya. Kolom churn belum diketahui.",
    "kota = Yogyakarta": "Kota ini tidak pernah muncul di data latih.",
})
