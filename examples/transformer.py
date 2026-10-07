# Transformer (self-attention): model kecil belajar MENGURUTKAN deret angka
import torch
import torch.nn as nn

torch.manual_seed(0)
L, V, D = 8, 10, 64                               # panjang deret, kosakata (digit 0-9), dimensi model

class PengurutTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.randn(L, D) * 0.02)          # positional embedding
        lapis = nn.TransformerEncoderLayer(d_model=D, nhead=4, dim_feedforward=128,
                                           dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(lapis, num_layers=3)
        self.out = nn.Linear(D, V)
    def forward(self, x):
        return self.out(self.enc(self.tok(x) + self.pos))         # (batch, L, V)

def batch(n):
    x = torch.randint(0, V, (n, L))
    return x, x.sort(dim=1).values

model = PengurutTransformer()
print(f"Parameter: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(1, 1501):
    x, y = batch(128)
    loss = nn.functional.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step in (1, 100, 250, 500, 1500):
        with torch.no_grad():
            xt, yt = batch(2000)
            p = model(xt).argmax(-1)
            print(f"langkah {step:>4} | loss = {loss.item():.4f} | akurasi per-digit = {(p == yt).float().mean():.3f}"
                  f" | deret benar 100% = {(p == yt).all(1).float().mean():.3f}")

contoh = torch.tensor([[7, 2, 9, 2, 0, 5, 1, 8]])
with torch.no_grad():
    print("Input :", contoh[0].tolist())
    print("Output:", model(contoh).argmax(-1)[0].tolist())

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
deret_baru = [[9, 8, 7, 6, 5, 4, 3, 2], [0, 1, 2, 3, 4, 5, 6, 7], [5, 5, 5, 5, 5, 5, 5, 5],
              [3, 0, 3, 0, 3, 0, 3, 0], [9, 0, 9, 0, 1, 1, 8, 8]]
baru = pd.DataFrame({"input_deret": [" ".join(map(str, d)) for d in deret_baru]})
with torch.no_grad():
    keluaran = model(torch.tensor(deret_baru)).argmax(-1).tolist()
for d, o in zip(deret_baru, keluaran):
    print(f"{d} -> {o} {'benar' if o == sorted(d) else 'SALAH, seharusnya ' + str(sorted(d))}")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
xs, ys = batch(6)
simpan("transformer", {"input_deret": [" ".join(map(str, r)) for r in xs.tolist()],
                       "target_terurut": [" ".join(map(str, r)) for r in ys.tolist()]}, {
    "input_deret": "8 digit acak (token 0–9). Input model.",
    "target_terurut": "Digit yang sama setelah diurutkan. Model memprediksi digit di setiap posisi.",
}, total=1500 * 128, catatan="Data dibangkitkan baru di setiap langkah: 1.500 langkah × 128 deret = 192.000 deret latih.")
simpan("transformer_baru", baru, {"input_deret": "Deret khusus: terurut terbalik, sudah terurut, semua sama, pola berulang, banyak kembar."})
