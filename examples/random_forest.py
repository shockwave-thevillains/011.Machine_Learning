# Random Forest: ratusan pohon keputusan yang "memilih bersama"
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer(as_frame=True)
df = data.frame.rename(columns={"target": "diagnosis"})   # 569 baris: 30 kolom ukuran sel + diagnosis
X, y = df.drop(columns="diagnosis").to_numpy(), df["diagnosis"].to_numpy()   # fitur (X) dan target (y)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
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
fitur = df.drop(columns="diagnosis")
profil_ganas = fitur[df.diagnosis == 0].median()       # median tiap kolom pada pasien ganas
profil_jinak = fitur[df.diagnosis == 1].median()
baru = pd.DataFrame([profil_jinak, (profil_jinak + profil_ganas) / 2, profil_ganas],
                    index=["pasien_A", "pasien_B", "pasien_C"])
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
from _sampel import KET_BREAST_CANCER, simpan
simpan("random_forest", df, KET_BREAST_CANCER, catatan="569 pasien: 212 ganas dan 357 jinak.")
simpan("random_forest_baru", baru.reset_index(names="pasien"), {
    "pasien_A": "Ukuran sel setara median pasien jinak.",
    "pasien_B": "Tepat di tengah antara profil jinak dan ganas (kasus sulit).",
    "pasien_C": "Ukuran sel setara median pasien ganas.",
}, catatan="Tiga pasien baru yang tidak ada di dataset. Kolomnya sama dengan data latih (30 ukuran sel).")
