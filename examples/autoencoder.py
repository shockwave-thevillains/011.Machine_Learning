# Autoencoder: memampatkan gambar digit 64 piksel ke 8 angka lalu merekonstruksinya
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32)
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
from _sampel import KET_DIGITS, digits_df, simpan
simpan("autoencoder", digits_df(X.numpy(), y), KET_DIGITS,
       catatan="Nilai piksel sudah dibagi 16 (skala 0–1). Kolom digit TIDAK dipakai: autoencoder belajar tanpa label, targetnya adalah input itu sendiri.")
