# Autoencoder: memampatkan gambar digit 64 piksel ke 8 angka lalu merekonstruksinya
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

torch.manual_seed(0)
df = load_digits(as_frame=True).frame.rename(columns={"target": "digit"})   # 1797 gambar: 64 kolom piksel + digit
X = torch.tensor(df.drop(columns="digit").to_numpy() / 16.0, dtype=torch.float32)   # kolom digit tidak dipakai
X_tr, X_te = X[:1400], X[1400:]

encoder = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))           # bottleneck 8
decoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid())
ae = nn.Sequential(encoder, decoder)
opt = torch.optim.Adam(ae.parameters(), lr=3e-3)

for epoch in range(1, 201):
    for idx in torch.randperm(1400).split(128):
        opt.zero_grad()
        loss = nn.functional.mse_loss(ae(X_tr[idx]), X_tr[idx])
        loss.backward()
        opt.step()
    if epoch in (1, 50, 100, 200):
        with torch.no_grad():
            print(f"epoch {epoch:>3} | MSE rekonstruksi test = {nn.functional.mse_loss(ae(X_te), X_te).item():.4f}")

pca = PCA(n_components=8).fit(X_tr.numpy())
rek_pca = pca.inverse_transform(pca.transform(X_te.numpy()))
print(f"Pembanding PCA 8 komponen     = {((rek_pca - X_te.numpy()) ** 2).mean():.4f}")
print(f"Rasio kompresi: 64 -> 8 angka ({8 / 64:.0%} ukuran asli)")
with torch.no_grad():
    kode = encoder(X_te[:1])
print("Kode laten 8 angka untuk 1 gambar:", [round(v, 2) for v in kode[0].tolist()])

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import numpy as np
GAMBAR = {                                       # angka baru yang digambar tangan: # = 16, + = 8, spasi = 0
    "nol":   ["  +##+  ", " +#++#+ ", " ##  ## ", " #+  +# ", " #+  +# ", " ##  ## ", " +#++#+ ", "  +##+  "],
    "satu":  ["   +#+  ", "  +##+  ", " +###+  ", "   ##+  ", "   ##+  ", "   ##+  ", "   ##+  ", "   +#+  "],
    "tujuh": ["  +#####", " ++++##+", "    +#+ ", "  +####+", "  ####+ ", "   +#+  ", "  +#+   ", "  +#    "],
}
GAMBAR["kotak-kotak"] = ["# # # # ", " # # # #"] * 4  # pola papan catur: BUKAN angka
def ke_piksel(pola):
    return np.array([[{"#": 16, "+": 8}.get(c, 0) for c in baris] for baris in pola], dtype=float).ravel()
X_baru = np.array([ke_piksel(p) for p in GAMBAR.values()])
with torch.no_grad():
    x_b = torch.tensor(X_baru / 16.0, dtype=torch.float32)
    error = ((ae(x_b) - x_b) ** 2).mean(1).numpy()
    batas = np.percentile(((ae(X_te) - X_te) ** 2).mean(1).numpy(), 99)
for nama, e in zip(GAMBAR, error):
    print(f"gambar '{nama}': error rekonstruksi {e:.4f} -> {'mirip angka' if e <= batas else 'TIDAK mirip angka (anomali)'}")
print(f"(batas: 99% gambar uji punya error ≤ {batas:.4f})")

# === VISUALISASI ===
from _plot import fig, save
with torch.no_grad():
    rek = ae(X_te[:8])
f, axes = fig(7.4, 2.2, nrows=2, ncols=8)
for i in range(8):
    axes[0, i].imshow(X_te[i].reshape(8, 8), cmap="Greys")
    axes[1, i].imshow(rek[i].reshape(8, 8), cmap="Blues")
    axes[0, i].axis("off")
    axes[1, i].axis("off")
axes[0, 0].set_title("asli", loc="left")
axes[1, 0].set_title("rekonstruksi", loc="left")
save("autoencoder")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import KET_DIGITS, simpan
simpan("autoencoder", df, KET_DIGITS, catatan="Di kode, nilai piksel dibagi 16 (skala 0–1). Kolom digit TIDAK dipakai: autoencoder belajar tanpa label, targetnya adalah input itu sendiri.")
import pandas as pd
simpan("autoencoder_baru", pd.DataFrame({n: [r.replace(" ", "·") for r in p] for n, p in GAMBAR.items()}),
       {"nol, satu, tujuh, kotak-kotak": "Gambar 8×8 baru yang digambar tangan, tidak ada di dataset. # = 16 (hitam), + = 8 (abu-abu), · = 0 (putih)."},
       catatan="Tiap kolom adalah satu gambar; tiap baris tabel adalah satu baris piksel. Sebelum masuk model, gambar diratakan menjadi 64 angka seperti pixel_0_0 … pixel_7_7.", idx=range(8))
