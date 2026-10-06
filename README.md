# Atlas Algoritma Machine Learning

Katalog 45 algoritma machine learning populer. Tiap algoritma punya sejarah singkat, fungsi dan cara kerja, rumus inti, kegunaan, kelebihan dan kekurangan, contoh kode Python yang bisa dijalankan, dan **output asli** dari eksekusi kode tersebut (plus grafik untuk sebagian besar algoritma).

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
algorithms.py   # teks: sejarah, fungsi, rumus, kegunaan, kelebihan/kekurangan
examples/       # 45 skrip Python yang bisa dijalankan sendiri-sendiri
outputs/        # output asli (.txt), grafik (.png), waktu eksekusi, versi library
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

Bagian di bawah penanda `# === VISUALISASI ===` pada tiap skrip hanya membuat grafik dan tidak ditampilkan di halaman. Angka bisa sedikit berbeda di mesin lain karena versi library, jumlah thread, dan perangkat keras.
