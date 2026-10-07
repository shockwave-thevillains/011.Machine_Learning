# Convolutional Neural Network (gaya LeNet) dengan PyTorch: klasifikasi digit 8x8
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
df = load_digits(as_frame=True).frame.rename(columns={"target": "digit"})   # 1797 gambar: 64 kolom piksel + digit
X = torch.tensor(df.drop(columns="digit").to_numpy() / 16.0, dtype=torch.float32).reshape(-1, 1, 8, 8)   # (N, channel, H, W)
y = torch.tensor(df["digit"].to_numpy())
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),            # 16 filter 3x3
    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),                                                  # 8x8 -> 4x4
    nn.Flatten(), nn.Dropout(0.3),
    nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, 10),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
print(f"Jumlah parameter: {sum(p.numel() for p in model.parameters()):,}")

riwayat = []
for epoch in range(1, 31):
    model.train()
    for idx in torch.randperm(len(X_tr)).split(64):                   # mini-batch 64
        opt.zero_grad()
        loss = loss_fn(model(X_tr[idx]), y_tr[idx])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(X_te).argmax(1) == y_te).float().mean().item()
    riwayat.append(acc)
    if epoch in (1, 5, 10, 20, 30):
        print(f"epoch {epoch:>2} | loss latih = {loss.item():.4f} | akurasi test = {acc:.3f}")

with torch.no_grad():
    peta = model[0](X_te[:1]).relu()
print(f"Bentuk feature map lapisan konvolusi pertama: {tuple(peta.shape)}")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import numpy as np
GAMBAR = {                                       # angka baru yang digambar tangan: # = 16, + = 8, spasi = 0
    "nol":   ["  +##+  ", " +#++#+ ", " ##  ## ", " #+  +# ", " #+  +# ", " ##  ## ", " +#++#+ ", "  +##+  "],
    "satu":  ["   +#+  ", "  +##+  ", " +###+  ", "   ##+  ", "   ##+  ", "   ##+  ", "   ##+  ", "   +#+  "],
    "tujuh": ["  +#####", " ++++##+", "    +#+ ", "  +####+", "  ####+ ", "   +#+  ", "  +#+   ", "  +#    "],
}
def ke_piksel(pola):
    return np.array([[{"#": 16, "+": 8}.get(c, 0) for c in baris] for baris in pola], dtype=float).ravel()
X_baru = np.array([ke_piksel(p) for p in GAMBAR.values()])
model.eval()
with torch.no_grad():
    proba = torch.softmax(model(torch.tensor(X_baru / 16.0, dtype=torch.float32).reshape(-1, 1, 8, 8)), dim=1)
for nama, pr in zip(GAMBAR, proba):
    top = pr.argsort(descending=True)[:2]
    print(f"gambar '{nama}' -> prediksi {top[0].item()} ({pr[top[0]]:.1%}) | kemungkinan kedua: {top[1].item()} ({pr[top[1]]:.1%})")

# === VISUALISASI ===
from _plot import BLUE, fig, save
f, axes = fig(7.4, 2.4, ncols=8)
axes[0].imshow(X_te[0, 0], cmap="Greys")
axes[0].set_title(f"input ({y_te[0].item()})")
for i, ax in enumerate(axes[1:]):
    ax.imshow(peta[0, i], cmap="bwr")
    ax.set_title(f"filter {i}")
for ax in axes:
    ax.axis("off")
save("cnn")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import KET_DIGITS, simpan
simpan("cnn", df, KET_DIGITS, catatan="Di kode, nilai piksel dibagi 16 (skala 0–1) lalu 64 kolom ini dibentuk ulang menjadi gambar 1 × 8 × 8.")
import pandas as pd
simpan("cnn_baru", pd.DataFrame({n: [r.replace(" ", "·") for r in p] for n, p in GAMBAR.items()}),
       {"nol, satu, tujuh": "Gambar 8×8 baru yang digambar tangan, tidak ada di dataset. # = 16 (hitam), + = 8 (abu-abu), · = 0 (putih)."},
       catatan="Tiap kolom adalah satu gambar; tiap baris tabel adalah satu baris piksel. Sebelum masuk model, gambar diratakan menjadi 64 angka seperti pixel_0_0 … pixel_7_7.", idx=range(8))
