# LSTM: memprediksi deret waktu (beban listrik harian sintetis) satu langkah ke depan
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
rng = np.random.default_rng(0)
t = np.arange(1200)
seri = 10 + 3 * np.sin(2 * np.pi * t / 24) + 1.5 * np.sin(2 * np.pi * t / 168) + rng.normal(0, .3, 1200)
mu, sd = seri.mean(), seri.std()
z = (seri - mu) / sd

W = 48                                           # gunakan 48 jam terakhir untuk menebak jam berikutnya
X = np.stack([z[i:i + W] for i in range(len(z) - W)])[..., None]
Y = z[W:]
X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
X_tr, Y_tr, X_te, Y_te = X[:900], Y[:900], X[900:], Y[900:]

class Peramal(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.out = nn.Linear(32, 1)
    def forward(self, x):
        h, _ = self.lstm(x)                      # h: (batch, waktu, 32)
        return self.out(h[:, -1]).squeeze(-1)    # pakai hidden state terakhir

model = Peramal()
opt = torch.optim.Adam(model.parameters(), lr=5e-3)
for epoch in range(1, 41):
    for idx in torch.randperm(900).split(64):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X_tr[idx]), Y_tr[idx])
        loss.backward()
        opt.step()
    if epoch in (1, 10, 20, 40):
        with torch.no_grad():
            print(f"epoch {epoch:>2} | MSE test (skala z) = {nn.functional.mse_loss(model(X_te), Y_te).item():.4f}")

with torch.no_grad():
    pred = model(X_te).numpy() * sd + mu
asli = Y_te.numpy() * sd + mu
naif = seri[900 + W - 1:-1]                      # baseline: "sama seperti jam sebelumnya"
print(f"MAE LSTM            : {np.abs(pred - asli).mean():.3f} MW")
print(f"MAE baseline naif   : {np.abs(naif - asli).mean():.3f} MW")

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
f, ax = fig()
ax.plot(asli[:168], color=BLUE, lw=1.4, label="aktual")
ax.plot(pred[:168], color=RED, lw=1.2, ls="--", label="prediksi LSTM")
ax.set(xlabel="jam ke-", ylabel="beban (MW)", title="LSTM: prediksi 1 jam ke depan (1 minggu data test)")
ax.legend()
save("lstm")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("lstm", {"jam_ke": t, "beban_mw": seri}, {
    "jam_ke": "Urutan jam sejak awal data.",
    "beban_mw": "Beban listrik pada jam tersebut (MW).",
}, catatan="Model membaca 48 jam berturut-turut (jendela) untuk menebak jam ke-49. Jam 0–947 untuk latih, sisanya untuk uji.")
