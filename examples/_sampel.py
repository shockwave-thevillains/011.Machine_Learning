"""Menyimpan potongan data yang benar-benar diolah tiap contoh, untuk ditampilkan di halaman.

Dipanggil dari bagian `# === SAMPEL DATA ===` di akhir tiap skrip, sehingga sampel
selalu berasal dari variabel yang sama dengan yang dipakai model.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "outputs"
MAKS_KOLOM = 12

KET_BREAST_CANCER = {
    "mean radius … worst fractal dimension": "30 ukuran inti sel dari citra biopsi jarum halus (FNA) jaringan payudara. "
    "Ada 10 ciri (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, "
    "fractal dimension), masing-masing dalam 3 versi: mean (rata-rata), error (standar error), dan worst (nilai terburuk).",
    "diagnosis": "Target. 0 = ganas (malignant), 1 = jinak (benign).",
}
KET_IRIS = {
    "sepal_pjg_cm, sepal_lbr_cm": "Panjang dan lebar kelopak luar (sepal) bunga, dalam cm.",
    "petal_pjg_cm, petal_lbr_cm": "Panjang dan lebar mahkota bunga (petal), dalam cm.",
    "spesies": "Target / label: setosa, versicolor, atau virginica.",
}
KET_DIGITS = {
    "px_00 … px_63": "Kecerahan 64 piksel gambar angka 8×8, dibaca baris demi baris (px_00 kiri atas, px_63 kanan bawah). "
    "Nilai 0 = putih sampai 16 = hitam.",
    "digit": "Angka sebenarnya pada gambar (0–9).",
}
IRIS_KOLOM = ["sepal_pjg_cm", "sepal_lbr_cm", "petal_pjg_cm", "petal_lbr_cm"]


def iris_df():
    from sklearn.datasets import load_iris
    iris = load_iris()
    return pd.DataFrame(iris.data, columns=IRIS_KOLOM).assign(spesies=iris.target_names[iris.target])


def digits_df(X, y):
    return pd.DataFrame(X, columns=[f"px_{i:02d}" for i in range(X.shape[1])]).assign(digit=y)


def _nilai(v):
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}".replace(",", ".")
    if isinstance(v, (float, np.floating)):
        if float(v).is_integer() and abs(v) < 1e6:
            return str(int(v))
        return f"{v:.1f}" if abs(v) >= 100 else f"{v:.3f}"
    return str(v)


def simpan(nama, data, keterangan, idx=None, total=None, catatan=""):
    """Simpan 6 baris (atau baris `idx`) dari data yang diolah + arti tiap kolom."""
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    n_baris, n_kolom = (total or len(df)), df.shape[1]
    kolom = list(df.columns)
    tersembunyi = []
    if n_kolom > MAKS_KOLOM:
        kolom = kolom[:MAKS_KOLOM - 1] + [kolom[-1]]
        tersembunyi = [c for c in df.columns if c not in kolom]
    potong = df.iloc[list(idx)] if idx is not None else df.head(6)
    baris = [[_nilai(v) for v in r] for r in potong[kolom].itertuples(index=False)]
    OUT.mkdir(exist_ok=True)
    (OUT / f"{nama}_sampel.json").write_text(json.dumps({
        "kolom": [str(k) for k in kolom],
        "baris": baris,
        "nomor_baris": [int(i) for i in (idx if idx is not None else range(len(potong)))],
        "total_baris": int(n_baris),
        "total_kolom": int(n_kolom),
        "tersembunyi": [str(tersembunyi[0]), str(tersembunyi[-1]), len(tersembunyi)] if tersembunyi else None,
        "keterangan": keterangan,
        "catatan": catatan,
    }, ensure_ascii=False, indent=1))
