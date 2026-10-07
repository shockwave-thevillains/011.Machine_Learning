# Gaussian Mixture Model + algoritma EM: clustering probabilistik (soft clustering)
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

X, y = make_blobs(n_samples=600, centers=3, cluster_std=[1.0, 1.5, 0.6], random_state=2)
X = X @ np.array([[0.6, -0.6], [-0.4, 0.8]])   # buat cluster lonjong (anisotropik)

print("Memilih jumlah komponen dengan BIC (lebih kecil lebih baik):")
for k in range(1, 7):
    g = GaussianMixture(n_components=k, covariance_type="full", random_state=0).fit(X)
    print(f"  k={k}  BIC = {g.bic(X):8.1f}")

gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0).fit(X)
print(f"\nEM konvergen setelah {gmm.n_iter_} iterasi | bobot komponen = {np.round(gmm.weights_, 3).tolist()}")
titik = np.array([[0.0, 0.0], [X[:, 0].mean(), X[:, 1].mean()]])
for t, p in zip(titik, gmm.predict_proba(titik)):
    print(f"Titik {np.round(t, 2).tolist()} -> probabilitas keanggotaan {np.round(p, 3).tolist()}")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
m = gmm.means_
baru = pd.DataFrame(np.vstack([m[0], (m[0] + m[2]) / 2, m[1] + 0.5, m.max(0) + 8]).round(2), columns=["x1", "x2"])
batas = np.percentile(gmm.score_samples(X), 1)           # 1% titik latih dengan kepadatan terendah
for xb, pr, ll in zip(baru.to_numpy(), gmm.predict_proba(baru.to_numpy()), gmm.score_samples(baru.to_numpy())):
    status = "TIDAK WAJAR (kepadatan sangat rendah)" if ll < batas else "wajar"
    print(f"titik {xb.tolist()} -> P komponen = {np.round(pr, 3).tolist()} | log-kepadatan {ll:6.1f} -> {status}")
print(f"(batas wajar: log-kepadatan ≥ {batas:.1f})")

# === VISUALISASI ===
from matplotlib.patches import Ellipse
from _plot import BLUE, RED, BLACK, fig, save
cols = [BLUE, RED, BLACK]
lab = gmm.predict(X)
f, ax = fig(5.6, 3.6)
for k in range(3):
    ax.scatter(*X[lab == k].T, s=7, color=cols[k], alpha=.6)
    val, vec = np.linalg.eigh(gmm.covariances_[k])
    ang = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
    for s in (1, 2):
        ax.add_patch(Ellipse(gmm.means_[k], *(2 * s * np.sqrt(val[::-1])), angle=ang,
                             fill=False, color=cols[k], lw=1.2))
ax.set(title="GMM: tiap cluster adalah distribusi Gaussian (elips 1σ, 2σ)")
save("gmm")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("gmm", {"x1": X[:, 0], "x2": X[:, 1], "komponen_hasil": gmm.predict(X),
               "p_keanggotaan_maks": gmm.predict_proba(X).max(1)}, {
    "x1, x2": "Koordinat titik (hanya dua kolom ini yang diolah).",
    "komponen_hasil": "HASIL: komponen Gaussian yang paling mungkin.",
    "p_keanggotaan_maks": "HASIL: seberapa yakin model (probabilitas komponen terpilih).",
})
simpan("gmm_baru", baru, {"x1, x2": "Titik baru: di pusat komponen 0, di antara komponen 0 dan 2, dekat komponen 1, dan jauh dari semuanya."})
