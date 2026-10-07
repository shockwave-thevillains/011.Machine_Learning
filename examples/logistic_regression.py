# Logistic Regression: mendeteksi tumor payudara ganas vs jinak
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()                    # 569 sampel, 30 fitur, 0 = ganas, 1 = jinak
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.25,
                                          stratify=data.target, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
pred = model.predict(X_te)
proba = model.predict_proba(X_te)[:, 1]

print(f"Akurasi : {accuracy_score(y_te, pred):.3f}")
print(f"ROC AUC : {roc_auc_score(y_te, proba):.4f}")
print("Confusion matrix [baris = asli, kolom = prediksi] (ganas, jinak):")
print(confusion_matrix(y_te, pred))
nama = ["ganas", "jinak"]
for i in range(3):
    print(f"Pasien #{i}: P(jinak) = {proba[i]:.3f} -> prediksi {nama[pred[i]]}, asli {nama[y_te[i]]}")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import numpy as np
import pandas as pd
profil_ganas = np.median(data.data[data.target == 0], axis=0)
profil_jinak = np.median(data.data[data.target == 1], axis=0)
baru = pd.DataFrame([profil_jinak, (profil_jinak + profil_ganas) / 2, profil_ganas],
                    columns=data.feature_names, index=["pasien_A", "pasien_B", "pasien_C"])
for nama, p in zip(baru.index, model.predict_proba(baru.to_numpy())[:, 1]):
    print(f"{nama}: P(jinak) = {p:.3f} -> {'jinak' if p >= 0.5 else 'GANAS'}")

# === VISUALISASI ===
import numpy as np
from _plot import BLUE, RED, fig, save
z = model.decision_function(X_te)
f, ax = fig()
zs = np.linspace(-15, 15, 300)
ax.plot(zs, 1 / (1 + np.exp(-zs)), color="black", lw=1.2, label="sigmoid σ(z)")
ax.scatter(z[y_te == 0], proba[y_te == 0], s=14, color=RED, label="asli: ganas")
ax.scatter(z[y_te == 1], proba[y_te == 1], s=14, color=BLUE, label="asli: jinak")
ax.axhline(.5, ls="--", color="grey", lw=.8)
ax.set(xlabel="z = w·x + b", ylabel="P(jinak)", title="Logistic Regression: skor linear -> probabilitas", xlim=(-15, 15))
ax.legend()
save("logistic_regression")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
import pandas as pd
from _sampel import KET_BREAST_CANCER, simpan
simpan("logistic_regression", pd.DataFrame(data.data, columns=data.feature_names).assign(diagnosis=data.target), KET_BREAST_CANCER,
       catatan="569 pasien: 212 ganas dan 357 jinak. Hanya 11 kolom pertama yang ditampilkan.")
simpan("logistic_regression_baru", baru.reset_index(names="pasien"), {
    "pasien_A": "Ukuran sel setara median pasien jinak.",
    "pasien_B": "Tepat di tengah antara profil jinak dan ganas (kasus sulit).",
    "pasien_C": "Ukuran sel setara median pasien ganas.",
}, catatan="Tiga pasien baru yang tidak ada di dataset. Kolomnya sama dengan data latih (30 ukuran sel).")
