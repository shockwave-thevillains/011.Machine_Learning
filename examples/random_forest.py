# Random Forest: ratusan pohon keputusan yang "memilih bersama"
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.25,
                                          stratify=data.target, random_state=42)
pohon = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", oob_score=True,
                            n_jobs=-1, random_state=42).fit(X_tr, y_tr)

print(f"1 Decision Tree   akurasi test = {pohon.score(X_te, y_te):.3f}")
print(f"Random Forest     akurasi test = {rf.score(X_te, y_te):.3f}")
print(f"Out-of-bag score  (validasi gratis tanpa data test) = {rf.oob_score_:.3f}")
print("5 fitur terpenting:")
for imp, nama in sorted(zip(rf.feature_importances_, data.feature_names), reverse=True)[:5]:
    print(f"  {nama:<22} {imp:.3f}")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import numpy as np
import pandas as pd
profil_ganas = np.median(data.data[data.target == 0], axis=0)
profil_jinak = np.median(data.data[data.target == 1], axis=0)
baru = pd.DataFrame([profil_jinak, (profil_jinak + profil_ganas) / 2, profil_ganas],
                    columns=data.feature_names, index=["pasien_A", "pasien_B", "pasien_C"])
suara = np.array([pohon.predict(baru.to_numpy()) for pohon in rf.estimators_])   # pilihan tiap pohon
for nama, p, v in zip(baru.index, rf.predict_proba(baru.to_numpy())[:, 1], suara.mean(0)):
    print(f"{nama}: P(jinak) = {p:.3f} | {v:.0%} dari 300 pohon memilih jinak -> {'jinak' if p >= 0.5 else 'GANAS'}")

# === VISUALISASI ===
import numpy as np
from _plot import BLUE, RED, fig, save
ns = [1, 2, 5, 10, 20, 50, 100, 200, 300]
acc = [RandomForestClassifier(n_estimators=n, random_state=42).fit(X_tr, y_tr).score(X_te, y_te) for n in ns]
f, ax = fig()
ax.plot(ns, acc, color=BLUE, marker="o", ms=4, label="Random Forest")
ax.axhline(pohon.score(X_te, y_te), color=RED, ls="--", label="1 Decision Tree")
ax.set_xscale("log")
ax.set(xlabel="jumlah pohon", ylabel="akurasi test", title="Random Forest: makin banyak pohon, makin stabil")
ax.legend()
save("random_forest")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
import pandas as pd
from _sampel import KET_BREAST_CANCER, simpan
simpan("random_forest", pd.DataFrame(data.data, columns=data.feature_names).assign(diagnosis=data.target), KET_BREAST_CANCER,
       catatan="569 pasien: 212 ganas dan 357 jinak. Hanya 11 kolom pertama yang ditampilkan.")
simpan("random_forest_baru", baru.reset_index(names="pasien"), {
    "pasien_A": "Ukuran sel setara median pasien jinak.",
    "pasien_B": "Tepat di tengah antara profil jinak dan ganas (kasus sulit).",
    "pasien_C": "Ukuran sel setara median pasien ganas.",
}, catatan="Tiga pasien baru yang tidak ada di dataset. Kolomnya sama dengan data latih (30 ukuran sel).")
