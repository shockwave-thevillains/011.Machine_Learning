# Isolation Forest: deteksi transaksi anomali (fraud) dengan "mengisolasi" titik
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

rng = np.random.default_rng(42)
normal = rng.normal([300, 14], [120, 3], (1000, 2))          # [nominal ribu Rp, jam transaksi]
fraud = np.c_[rng.uniform(2000, 5000, 15), rng.uniform(0, 5, 15)]   # nominal besar, dini hari
X = np.vstack([normal, fraud])
y = np.r_[np.zeros(1000), np.ones(15)]

iso = IsolationForest(n_estimators=200, contamination=0.015, random_state=0).fit(X)
pred = (iso.predict(X) == -1).astype(int)       # -1 = anomali
skor = -iso.score_samples(X)

print(classification_report(y, pred, target_names=["normal", "anomali"], digits=3))
print(f"Rata-rata skor anomali  normal = {skor[:1000].mean():.3f} | fraud = {skor[1000:].mean():.3f}")
print("Transaksi Rp 4,2 jt jam 03.00 ->", "ANOMALI" if iso.predict([[4200, 3]])[0] == -1 else "normal")
print("Transaksi Rp 250 rb jam 13.00 ->", "ANOMALI" if iso.predict([[250, 13]])[0] == -1 else "normal")

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
simpan("isolation_forest", {"nominal_ribu_rp": X[:, 0], "jam_transaksi": X[:, 1],
                            "fraud_asli": y.astype(int), "terdeteksi_anomali": pred}, {
    "nominal_ribu_rp": "Nilai transaksi dalam ribu rupiah.",
    "jam_transaksi": "Jam terjadinya transaksi (0–24).",
    "fraud_asli": "1 = transaksi fraud yang sengaja disisipkan. Hanya untuk evaluasi.",
    "terdeteksi_anomali": "HASIL Isolation Forest: 1 = dianggap anomali.",
}, idx=[0, 1, 2, 1000, 1001, 1002], catatan="Tiga baris terakhir adalah transaksi fraud.")
