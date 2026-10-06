# Multi-Armed Bandit: A/B testing 3 desain tombol "Beli" — ε-greedy vs UCB1 vs Thompson Sampling
import numpy as np

ctr_asli = np.array([0.040, 0.050, 0.065])      # tidak diketahui algoritma
T, ulang = 20_000, 30

def jalankan(strategi, rng):
    n, s = np.zeros(3), np.zeros(3)              # jumlah tampil, jumlah klik
    regret = np.zeros(T)
    for t in range(T):
        if strategi == "eps-greedy":
            a = rng.integers(3) if rng.random() < 0.1 else int(np.argmax(s / np.maximum(n, 1)))
        elif strategi == "UCB1":
            a = t if t < 3 else int(np.argmax(s / n + np.sqrt(2 * np.log(t) / n)))
        else:                                    # Thompson Sampling dengan prior Beta(1,1)
            a = int(np.argmax(rng.beta(1 + s, 1 + n - s)))
        klik = rng.random() < ctr_asli[a]
        n[a] += 1
        s[a] += klik
        regret[t] = ctr_asli.max() - ctr_asli[a]
    return np.cumsum(regret), n

hasil = {}
for st in ("eps-greedy", "UCB1", "Thompson"):
    runs = [jalankan(st, np.random.default_rng(i)) for i in range(ulang)]
    hasil[st] = np.mean([r for r, _ in runs], axis=0)
    porsi = np.mean([n / T for _, n in runs], axis=0)
    print(f"{st:<10} | regret kumulatif = {hasil[st][-1]:6.1f} klik hilang | "
          f"porsi tampil desain A/B/C = {np.round(porsi * 100, 1).tolist()} %")
print(f"(Uji A/B klasik 33/33/33 akan kehilangan ≈ {T * (ctr_asli.max() - ctr_asli.mean()):.0f} klik)")

# === VISUALISASI ===
from _plot import BLUE, RED, BLACK, fig, save
f, ax = fig()
for (st, r), c in zip(hasil.items(), (BLACK, BLUE, RED)):
    ax.plot(r, color=c, lw=1.6, label=st)
ax.set(xlabel="jumlah pengunjung", ylabel="regret kumulatif", title="Bandit: Thompson Sampling paling cepat fokus ke desain terbaik")
ax.legend()
save("bandit")
