# Isolation Forest: deteksi transaksi anomali (fraud) dengan "mengisolasi" titik
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

rng = np.random.default_rng(42)
normal = rng.normal([300, 14], [120, 3], (1000, 2))          # [nominal ribu Rp, jam transaksi]
fraud = np.c_[rng.uniform(2000, 5000, 15), rng.uniform(0, 5, 15)]   # nominal besar, dini hari
df = pd.DataFrame(np.vstack([normal, fraud]), columns=["nominal_ribu_rp", "jam_transaksi"])
df["fraud_asli"] = np.r_[np.zeros(1000), np.ones(15)].astype(int)   # hanya untuk evaluasi
X, y = df[["nominal_ribu_rp", "jam_transaksi"]].to_numpy(), df["fraud_asli"].to_numpy()

iso = IsolationForest(n_estimators=200, contamination=0.015, random_state=0).fit(X)
pred = (iso.predict(X) == -1).astype(int)       # -1 = anomali
skor = -iso.score_samples(X)

print(classification_report(y, pred, target_names=["normal", "anomali"], digits=3))
print(f"Rata-rata skor anomali  normal = {skor[:1000].mean():.3f} | fraud = {skor[1000:].mean():.3f}")
print("Transaksi Rp 4,2 jt jam 03.00 ->", "ANOMALI" if iso.predict([[4200, 3]])[0] == -1 else "normal")
print("Transaksi Rp 250 rb jam 13.00 ->", "ANOMALI" if iso.predict([[250, 13]])[0] == -1 else "normal")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
baru = pd.DataFrame({"nominal_ribu_rp": [150, 450, 3500, 900, 2800], "jam_transaksi": [12, 19, 2, 23, 14]})
for (nom, jam), lbl, s in zip(baru.to_numpy(), iso.predict(baru.to_numpy()), -iso.score_samples(baru.to_numpy())):
    print(f"Rp {nom:>5,.0f} ribu pukul {jam:02.0f}.00 -> skor {s:.3f} -> {'ANOMALI' if lbl == -1 else 'normal'}")

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
f, ax = fig()
ax.scatter(*X[pred == 0].T, s=7, color=BLUE, alpha=.5, label="normal")
ax.scatter(*X[pred == 1].T, s=40, color=RED, marker="x", label="terdeteksi anomali")
ax.set(xlabel="nominal (ribu Rp)", ylabel="jam transaksi", title="Isolation Forest: deteksi fraud")
ax.legend()
save("isolation_forest")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("isolation_forest", df.assign(terdeteksi_anomali=pred), {
    "nominal_ribu_rp": "Nilai transaksi dalam ribu rupiah.",
    "jam_transaksi": "Jam terjadinya transaksi (0–24).",
    "fraud_asli": "1 = transaksi fraud yang sengaja disisipkan. Hanya untuk evaluasi.",
    "terdeteksi_anomali": "HASIL Isolation Forest: 1 = dianggap anomali.",
}, idx=[0, 1, 2, 1000, 1001, 1002], catatan="Tiga baris terakhir adalah transaksi fraud.")
simpan("isolation_forest_baru", baru, {"nominal_ribu_rp, jam_transaksi": "Transaksi baru yang masuk hari ini."})
