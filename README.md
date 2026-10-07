# Atlas Algoritma Machine Learning

Katalog 45 algoritma machine learning populer. Tiap algoritma punya sejarah singkat, fungsi dan cara kerja, rumus inti, kegunaan, kelebihan dan kekurangan, **contoh data yang diolah beserta nama dan arti kolomnya**, contoh kode Python yang bisa dijalankan, **output asli** dari eksekusi kode tersebut (plus grafik untuk sebagian besar algoritma), dan **panduan apa yang bisa dicek dari hasilnya**.

Buka `index.html` di browser untuk melihat halamannya (latar putih, teks hitam, aksen biru `#0000ff` dan merah `#ff0000`).

## Isi

| Kategori | Algoritma |
|---|---|
| Regresi | Linear, Polynomial, Ridge, Lasso, Elastic Net, SVR |
| Klasifikasi | Logistic Regression, Perceptron, KNN, Naive Bayes, LDA, Decision Tree, SVM, MLP |
| Ensemble | Random Forest, Bagging, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost, Stacking |
| Clustering | K-Means, Hierarchical, DBSCAN, GMM (EM), Mean Shift, Spectral Clustering |
| Reduksi Dimensi | PCA, t-SNE, UMAP, ICA |
| Deteksi Anomali | Isolation Forest, Local Outlier Factor |
| Aturan Asosiasi | Apriori, FP-Growth |
| Probabilistik & Rekomendasi | Hidden Markov Model (Viterbi), Matrix Factorization |
| Reinforcement Learning | Q-Learning, Thompson Sampling (multi-armed bandit) |
| Deep Learning | CNN, LSTM, Autoencoder, GAN, Transformer |

## Struktur

```
algorithms.py   # teks: sejarah, fungsi, rumus, kegunaan, kelebihan/kekurangan, panduan cek hasil (CEK)
examples/       # 45 skrip Python yang bisa dijalankan sendiri-sendiri
outputs/        # output asli (.txt), grafik (.png), sampel data (*_sampel.json), waktu eksekusi, versi library
build.py        # menjalankan semua contoh lalu menulis index.html
index.html      # halaman hasil build
```

## Menjalankan

```bash
pip install -r requirements.txt

cd examples && python kmeans.py     # jalankan satu contoh
cd .. && python build.py            # jalankan semua contoh + bangun ulang index.html
python build.py --skip-run          # bangun ulang halaman dari output yang ada
python build.py --only knn svm      # jalankan ulang sebagian contoh
```

Bagian di bawah penanda `# === SAMPEL DATA ===` dan `# === VISUALISASI ===` pada tiap skrip hanya menyimpan potongan data dan membuat grafik; keduanya tidak ditampilkan di halaman. Karena sampel diambil dari variabel yang sama dengan yang diolah model, tabel "Data yang diolah" di halaman selalu sesuai dengan kodenya. Angka bisa sedikit berbeda di mesin lain karena versi library, jumlah thread, dan perangkat keras.

## GitHub Pages

Workflow `.github/workflows/pages.yml` menerbitkan `index.html` dan grafik di `outputs/` ke GitHub Pages setiap kali ada push ke `main`. Alamatnya: https://shockwave-thevillains.github.io/011.Machine_Learning/

Jika workflow gagal di langkah `configure-pages`, aktifkan Pages sekali secara manual: **Settings → Pages → Build and deployment → Source: GitHub Actions**, lalu jalankan ulang workflow dari tab Actions.
