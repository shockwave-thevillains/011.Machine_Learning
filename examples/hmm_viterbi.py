# Hidden Markov Model: menebak cuaca (tersembunyi) dari aktivitas teman (terlihat)
import numpy as np
import pandas as pd

state = ["Hujan", "Cerah"]
obs_nama = ["jalan-jalan", "belanja", "bersih-bersih"]
pi = np.array([0.6, 0.4])                        # peluang awal
A = np.array([[0.7, 0.3],                        # transisi: Hujan->Hujan, Hujan->Cerah
              [0.4, 0.6]])                       #           Cerah->Hujan, Cerah->Cerah
B = np.array([[0.1, 0.4, 0.5],                   # emisi saat Hujan
              [0.6, 0.3, 0.1]])                  # emisi saat Cerah

def forward(obs):                                # P(urutan observasi)
    alpha = pi * B[:, obs[0]]
    for o in obs[1:]:
        alpha = (alpha @ A) * B[:, o]
    return alpha.sum()

def viterbi(obs):                                # urutan state paling mungkin
    delta = np.log(pi * B[:, obs[0]])
    jejak = []
    for o in obs[1:]:
        skor = delta[:, None] + np.log(A)
        jejak.append(skor.argmax(0))
        delta = skor.max(0) + np.log(B[:, o])
    jalur = [int(delta.argmax())]
    for j in reversed(jejak):
        jalur.insert(0, int(j[jalur[0]]))
    return jalur, np.exp(delta.max())

df = pd.DataFrame({"hari_ke": [1, 2, 3, 4, 5],
                   "aktivitas": ["jalan-jalan", "belanja", "bersih-bersih", "bersih-bersih", "jalan-jalan"]})
amatan = [obs_nama.index(a) for a in df["aktivitas"]]   # ubah teks aktivitas menjadi nomor 0/1/2
print("Aktivitas teman      :", [obs_nama[o] for o in amatan])
print(f"P(urutan aktivitas)  : {forward(amatan):.6f}  (algoritma Forward)")
jalur, p = viterbi(amatan)
print("Cuaca paling mungkin :", [state[s] for s in jalur], f" (P jalur = {p:.6f}, algoritma Viterbi)")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
amatan_baru = [2, 2, 1, 0, 0, 0, 2]              # 7 hari aktivitas yang baru diamati
jalur_baru, _ = viterbi(amatan_baru)
print("Aktivitas 7 hari  :", [obs_nama[o] for o in amatan_baru])
print("Cuaca (Viterbi)   :", [state[s] for s in jalur_baru])
alpha = pi * B[:, amatan_baru[0]]                 # filtering: peluang cuaca hari terakhir
for o in amatan_baru[1:]:
    alpha = (alpha @ A) * B[:, o]
hari_ini = alpha / alpha.sum()
besok = hari_ini @ A                              # ramalan 1 hari ke depan
print(f"P(cuaca hari ke-7): Hujan {hari_ini[0]:.2f}, Cerah {hari_ini[1]:.2f}")
print(f"Ramalan cuaca besok: Hujan {besok[0]:.2f}, Cerah {besok[1]:.2f}")
print("Ramalan aktivitas besok:", {obs_nama[i]: round(float(p), 2) for i, p in enumerate(besok @ B)})

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("hmm_viterbi", df.assign(cuaca_hasil_viterbi=[state[s] for s in jalur]), {
    "hari_ke": "Urutan hari.",
    "aktivitas": "Data input: aktivitas teman yang bisa kita lihat.",
    "cuaca_hasil_viterbi": "HASIL: cuaca tersembunyi yang paling mungkin menurut algoritma Viterbi.",
}, catatan="Parameter model (peluang awal π, transisi A, emisi B) ditulis langsung di kode, bukan dipelajari dari data.")
simpan("hmm_viterbi_baru", {"hari_ke": range(1, 8), "aktivitas_teramati": [obs_nama[o] for o in amatan_baru]},
       {"aktivitas_teramati": "Aktivitas teman selama 7 hari terakhir. Cuacanya tidak diketahui."}, total=7, idx=range(7))
