# Q-Learning: agen belajar keluar dari labirin es (gridworld 4x4) lewat trial & error
import numpy as np

peta = ["SFFF",
        "FHFH",
        "FFFH",
        "HFFG"]                                    # S start, F es, H lubang, G tujuan
n, aksi = 4, [(0, -1), (1, 0), (0, 1), (-1, 0)]    # kiri, bawah, kanan, atas
panah = "←↓→↑"

def langkah(s, a, rng):
    if rng.random() < 0.1:                         # 10% licin: aksi acak
        a = rng.integers(4)
    r, c = divmod(s, n)
    r, c = min(max(r + aksi[a][0], 0), n - 1), min(max(c + aksi[a][1], 0), n - 1)
    sel = peta[r][c]
    return r * n + c, (1.0 if sel == "G" else 0.0), sel in "GH"

rng = np.random.default_rng(0)
Q = np.zeros((16, 4))
alpha, gamma, eps = 0.1, 0.95, 1.0
riwayat = []
for ep in range(5000):
    s, selesai, total = 0, False, 0
    for _ in range(100):
        a = rng.integers(4) if rng.random() < eps else int(Q[s].argmax())
        s2, r, selesai = langkah(s, a, rng)
        Q[s, a] += alpha * (r + gamma * Q[s2].max() * (not selesai) - Q[s, a])   # update Bellman
        s, total = s2, total + r
        if selesai:
            break
    eps = max(0.05, eps * 0.999)
    riwayat.append(total)

for i in (500, 1000, 2500, 5000):
    print(f"episode {i - 499:>4}-{i:<4}: tingkat sukses = {np.mean(riwayat[i - 500:i]):.1%} (ε akhir = {max(0.05, 0.999 ** i):.2f})")

print("\nKebijakan yang dipelajari (aksi terbaik tiap petak):")
for r in range(n):
    print("  " + " ".join(peta[r][c] if peta[r][c] in "HG" else panah[Q[r * n + c].argmax()] for c in range(n)))

uji = np.random.default_rng(1)
sukses = 0
for _ in range(1000):
    s = 0
    for _ in range(100):
        s, rew, done = langkah(s, int(Q[s].argmax()), uji)
        if done:
            sukses += rew
            break
print(f"Uji 1000 episode dengan kebijakan greedy: sukses {sukses / 1000:.1%}")

# === VISUALISASI ===
from _plot import BLUE, RED, fig, save
f, ax = fig()
w = 100
ax.plot(np.convolve(riwayat, np.ones(w) / w, "valid"), color=BLUE, lw=1.4)
ax.axhline(sukses / 1000, color=RED, ls="--", label="kebijakan akhir (greedy)")
ax.set(xlabel="episode", ylabel="tingkat sukses (rata-rata 100 ep.)", title="Q-Learning: agen makin sering mencapai tujuan")
ax.legend()
save("q_learning")
