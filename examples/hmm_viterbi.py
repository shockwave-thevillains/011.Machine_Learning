# Hidden Markov Model: menebak cuaca (tersembunyi) dari aktivitas teman (terlihat)
import numpy as np

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

amatan = [0, 1, 2, 2, 0]                          # 5 hari aktivitas teman
print("Aktivitas teman      :", [obs_nama[o] for o in amatan])
print(f"P(urutan aktivitas)  : {forward(amatan):.6f}  (algoritma Forward)")
jalur, p = viterbi(amatan)
print("Cuaca paling mungkin :", [state[s] for s in jalur], f" (P jalur = {p:.6f}, algoritma Viterbi)")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("hmm_viterbi", {"hari_ke": range(1, len(amatan) + 1), "aktivitas_teramati": [obs_nama[o] for o in amatan],
                       "cuaca_hasil_viterbi": [state[s] for s in jalur]}, {
    "hari_ke": "Urutan hari.",
    "aktivitas_teramati": "Data input: aktivitas teman yang bisa kita lihat.",
    "cuaca_hasil_viterbi": "HASIL: cuaca tersembunyi yang paling mungkin menurut algoritma Viterbi.",
}, catatan="Parameter model (peluang awal π, transisi A, emisi B) ditulis langsung di kode, bukan dipelajari dari data.")
