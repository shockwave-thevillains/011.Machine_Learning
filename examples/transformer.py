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
