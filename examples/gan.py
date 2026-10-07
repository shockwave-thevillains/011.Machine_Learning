# Generative Adversarial Network: generator belajar meniru distribusi tinggi badan
import torch
import torch.nn as nn

torch.manual_seed(0)
def data_asli(n):                                 # data asli: kolom tinggi_cm ~ N(165 cm, 7 cm)
    tinggi_cm = torch.randn(n, 1) * 7 + 165
    return (tinggi_cm - 165) / 10                  # dinormalisasi agar mudah dipelajari

G = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
D = nn.Sequential(nn.Linear(1, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))
opt_G = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()
satu, nol = torch.ones(256, 1), torch.zeros(256, 1)

def ringkas(langkah):
    with torch.no_grad():
        s = G(torch.randn(10_000, 4)) * 10 + 165
    print(f"langkah {langkah:>5} | sampel generator: mean = {s.mean():6.2f} cm, std = {s.std():5.2f} cm")

for langkah in range(1, 6001):
    # 1) latih Discriminator: bedakan asli (1) vs palsu (0)
    palsu = G(torch.randn(256, 4)).detach()
    loss_D = bce(D(data_asli(256)), satu) + bce(D(palsu), nol)
    opt_D.zero_grad(); loss_D.backward(); opt_D.step()
    # 2) latih Generator: tipu Discriminator agar menilai palsu sebagai asli
    loss_G = bce(D(G(torch.randn(256, 4))), satu)
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()
    if langkah in (1, 500, 2000, 6000):
        ringkas(langkah)
print("Target distribusi asli            : mean = 165.00 cm, std =  7.00 cm")
print(f"Loss akhir D = {loss_D.item():.3f} (≈ 2·ln2 = 1.386 berarti D tak bisa membedakan) | G = {loss_G.item():.3f}")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
with torch.no_grad():
    sampel_baru = (G(torch.randn(8, 4)) * 10 + 165).ravel()
print("8 tinggi badan baru dari generator (cm):", [round(v, 1) for v in sampel_baru.tolist()])
baru = pd.DataFrame({"tinggi_cm": [120.0, 150.0, 165.0, 180.0, 210.0]})
with torch.no_grad():
    p_asli = torch.sigmoid(D(torch.tensor((baru.tinggi_cm.to_numpy() - 165) / 10, dtype=torch.float32).reshape(-1, 1))).ravel()
for tinggi, p in zip(baru.tinggi_cm, p_asli.tolist()):
    print(f"Discriminator menilai {tinggi:.0f} cm: P(asli) = {p:.2f}")

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
with torch.no_grad():
    s = (G(torch.randn(20_000, 4)) * 10 + 165).numpy().ravel()
a = (data_asli(20_000) * 10 + 165).numpy().ravel()
f, ax = fig()
ax.hist(a, bins=60, density=True, color=BLUE, alpha=.45, label="data asli")
ax.hist(s, bins=60, density=True, histtype="step", color=RED, lw=1.8, label="hasil generator")
ax.set(xlabel="tinggi badan (cm)", ylabel="kepadatan", title="GAN: distribusi palsu vs asli")
ax.legend()
save("gan")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
import pandas as pd
from _sampel import simpan
torch.manual_seed(1)
simpan("gan", pd.DataFrame({"tinggi_cm": (data_asli(6) * 10 + 165).numpy().ravel()}), {
    "tinggi_cm": "Tinggi badan dari distribusi asli N(165 cm, 7 cm). Discriminator melihat data ini (dinormalisasi) sebagai contoh \"asli\".",
}, total=6 * 256 * 1000, catatan="Contoh keluaran data_asli(). Data asli tidak disimpan sebagai tabel; tiap langkah pelatihan mengambil 256 sampel baru, sehingga 6.000 langkah memakai ±1,5 juta sampel.")
simpan("gan_baru", baru, {"tinggi_cm": "Nilai tinggi badan yang ingin dinilai Discriminator: wajar atau tidak."})
