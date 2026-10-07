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
    return np.cumsum(regret), n, s

hasil = {}
for st in ("eps-greedy", "UCB1", "Thompson"):
    runs = [jalankan(st, np.random.default_rng(i)) for i in range(ulang)]
    hasil[st] = np.mean([r for r, _, _ in runs], axis=0)
    porsi = np.mean([n / T for _, n, _ in runs], axis=0)
    print(f"{st:<10} | regret kumulatif = {hasil[st][-1]:6.1f} klik hilang | "
          f"porsi tampil desain A/B/C = {np.round(porsi * 100, 1).tolist()} %")
print(f"(Uji A/B klasik 33/33/33 akan kehilangan ≈ {T * (ctr_asli.max() - ctr_asli.mean()):.0f} klik)")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
_, n_akhir, s_akhir = jalankan("Thompson", np.random.default_rng(100))    # satu kampanye 20.000 pengunjung
baru = pd.DataFrame({"desain": list("ABC"), "tampil": n_akhir.astype(int), "klik": s_akhir.astype(int)})
rng_p = np.random.default_rng(1)
post = rng_p.beta(1 + s_akhir, 1 + n_akhir - s_akhir, size=(10_000, 3))   # keyakinan akhir tentang CTR
for i, d in enumerate("ABC"):
    lo, hi = np.percentile(post[:, i], [2.5, 97.5])
    print(f"desain {d}: CTR perkiraan {post[:, i].mean():.2%} (95%: {lo:.2%}–{hi:.2%}) | P(terbaik) = {(post.argmax(1) == i).mean():.1%}")
pilihan = ["ABC"[int(np.argmax(rng_p.beta(1 + s_akhir, 1 + n_akhir - s_akhir)))] for _ in range(10)]
print("Desain untuk 10 pengunjung berikutnya:", " ".join(pilihan))

# === VISUALISASI ===
from _plot import BLUE, RED, BLACK, fig, save
f, ax = fig()
for (st, r), c in zip(hasil.items(), (BLACK, BLUE, RED)):
    ax.plot(r, color=c, lw=1.6, label=st)
ax.set(xlabel="jumlah pengunjung", ylabel="regret kumulatif", title="Bandit: Thompson Sampling paling cepat fokus ke desain terbaik")
ax.legend()
save("bandit")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
rng_c, n_c, s_c, log = np.random.default_rng(0), np.zeros(3), np.zeros(3), []
for t in range(8):
    a = int(np.argmax(rng_c.beta(1 + s_c, 1 + n_c - s_c)))
    klik = int(rng_c.random() < ctr_asli[a])
    n_c[a] += 1
    s_c[a] += klik
    log.append((t + 1, "ABC"[a], klik))
simpan("bandit", dict(zip(["pengunjung_ke", "desain_ditampilkan", "klik"], zip(*log))), {
    "pengunjung_ke": "Urutan pengunjung halaman.",
    "desain_ditampilkan": "Desain tombol yang dipilih algoritma (Thompson Sampling) untuk pengunjung itu.",
    "klik": "1 jika pengunjung mengklik tombol, 0 jika tidak. Ini satu-satunya umpan balik yang diterima algoritma.",
}, total=T, catatan="Contoh 8 pengunjung pertama. Klik dibangkitkan dari CTR asli A = 4,0%, B = 5,0%, C = 6,5%, yang tidak diketahui algoritma.")
simpan("bandit_baru", baru, {"tampil, klik": "Hasil satu kampanye: berapa kali tiap desain ditampilkan dan diklik.",
                             "desain": "Data inilah yang dipakai untuk memutuskan desain bagi pengunjung berikutnya."})
