# FP-Growth: frequent itemset mining tanpa candidate generation (lebih cepat dari Apriori)
import time

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth

rng = np.random.default_rng(0)
produk = [f"P{i:02d}" for i in range(60)]
popularitas = rng.dirichlet(np.ones(60) * 0.3)        # beberapa produk sangat laris
bundel = [("P01", "P02"), ("P10", "P11", "P12"), ("P30", "P31")]
baris = []
for _ in range(20_000):                                # 20.000 struk belanja
    isi = set(rng.choice(produk, size=rng.integers(2, 9), p=popularitas))
    for b in bundel:
        if rng.random() < 0.15:
            isi.update(b)
    baris.append({p: (p in isi) for p in produk})
df = pd.DataFrame(baris)

for nama, fungsi in (("Apriori", apriori), ("FP-Growth", fpgrowth)):
    t = time.perf_counter()
    hasil = fungsi(df, min_support=0.02, use_colnames=True)
    print(f"{nama:<9}: {len(hasil):>4} itemset sering ditemukan dalam {time.perf_counter() - t:6.2f} detik")

hasil = fpgrowth(df, min_support=0.02, use_colnames=True)
besar = hasil[hasil.itemsets.apply(len) >= 3].sort_values("support", ascending=False)
print("Itemset 3-produk teratas:", [(sorted(s), round(v, 3)) for s, v in zip(besar.itemsets, besar.support)][:3])
