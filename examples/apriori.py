# Apriori: menemukan aturan asosiasi "yang membeli X juga membeli Y" (market basket analysis)
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

transaksi = [
    ["beras", "telur", "minyak goreng"],
    ["kopi", "gula", "susu kental manis"],
    ["mie instan", "telur", "kopi"],
    ["beras", "telur", "minyak goreng", "kecap"],
    ["kopi", "gula", "roti"],
    ["mie instan", "telur", "saus sambal"],
    ["beras", "minyak goreng", "gula"],
    ["kopi", "gula", "susu kental manis", "roti"],
    ["mie instan", "telur", "kopi", "gula"],
    ["beras", "telur", "minyak goreng", "mie instan"],
    ["roti", "susu kental manis", "kopi"],
    ["beras", "telur", "kecap"],
]
te = TransactionEncoder()
df = pd.DataFrame(te.fit_transform(transaksi), columns=te.columns_)

itemset = apriori(df, min_support=0.25, use_colnames=True)
print(f"{len(itemset)} itemset sering muncul (support ≥ 25%), contoh 2-item:")
for _, r in itemset[itemset.itemsets.apply(len) == 2].iterrows():
    print(f"  {sorted(r.itemsets)}  support = {r.support:.2f}")

rules = association_rules(itemset, metric="confidence", min_threshold=0.7)
rules = rules.sort_values("lift", ascending=False)
print("\nAturan asosiasi (confidence ≥ 70%):")
for _, r in rules.iterrows():
    print(f"  {sorted(r.antecedents)} -> {sorted(r.consequents)}  "
          f"support={r.support:.2f} conf={r.confidence:.2f} lift={r.lift:.2f}")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("apriori", {"id_struk": range(1, len(transaksi) + 1), "barang": [", ".join(t) for t in transaksi],
                   "jumlah_barang": [len(t) for t in transaksi]}, {
    "id_struk": "Nomor struk belanja.",
    "barang": "Barang yang dibeli bersamaan dalam satu struk.",
    "jumlah_barang": "Banyaknya barang di struk itu.",
}, catatan=f"Sebelum diolah, TransactionEncoder mengubah tiap struk menjadi satu baris True/False dengan {df.shape[1]} kolom (satu per jenis barang).")
