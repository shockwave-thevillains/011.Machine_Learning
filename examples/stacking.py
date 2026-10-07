# Stacking (Stacked Generalization): meta-model belajar menggabungkan beberapa model
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
base = [
    ("rf", RandomForestClassifier(n_estimators=200, random_state=0)),
    ("svm", make_pipeline(StandardScaler(), SVC(random_state=0))),
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))),
]
stack = StackingClassifier(estimators=base, final_estimator=LogisticRegression(), cv=5)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
for nama, m in base + [("STACKING", stack)]:
    s = cross_val_score(m, X, y, cv=cv)
    print(f"{nama:<9} akurasi 10-fold = {s.mean():.4f} ± {s.std():.4f}")

stack.fit(X, y)
print("Bobot meta-model (RF, SVM-skor, KNN):", stack.final_estimator_.coef_.round(2).tolist()[0])

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import numpy as np
data = load_breast_cancer()                       # nama kolom & data asli untuk menyusun pasien baru
import pandas as pd
profil_ganas = np.median(data.data[data.target == 0], axis=0)
profil_jinak = np.median(data.data[data.target == 1], axis=0)
baru = pd.DataFrame([profil_jinak, (profil_jinak + profil_ganas) / 2, profil_ganas],
                    columns=data.feature_names, index=["pasien_A", "pasien_B", "pasien_C"])
for nama, x in zip(baru.index, baru.to_numpy()):
    x = x.reshape(1, -1)
    rf_p = stack.named_estimators_["rf"].predict_proba(x)[0, 1]
    svm_s = stack.named_estimators_["svm"].decision_function(x)[0]
    knn_p = stack.named_estimators_["knn"].predict_proba(x)[0, 1]
    akhir = stack.predict_proba(x)[0, 1]
    print(f"{nama}: RF P(jinak)={rf_p:.2f} | SVM skor={svm_s:+.2f} | KNN P(jinak)={knn_p:.2f} -> STACKING P(jinak)={akhir:.3f}")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
import pandas as pd
from _sampel import KET_BREAST_CANCER, simpan
simpan("stacking", pd.DataFrame(X, columns=load_breast_cancer().feature_names).assign(diagnosis=y), KET_BREAST_CANCER,
       catatan="569 pasien: 212 ganas dan 357 jinak. Hanya 11 kolom pertama yang ditampilkan.")
simpan("stacking_baru", baru.reset_index(names="pasien"), {
    "pasien_A": "Ukuran sel setara median pasien jinak.",
    "pasien_B": "Tepat di tengah antara profil jinak dan ganas (kasus sulit).",
    "pasien_C": "Ukuran sel setara median pasien ganas.",
}, catatan="Tiga pasien baru yang tidak ada di dataset. Kolomnya sama dengan data latih (30 ukuran sel).")
