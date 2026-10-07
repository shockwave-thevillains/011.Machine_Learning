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
rules = rules.sort_values("lift", ascending=False, kind="stable")
print("\nAturan asosiasi (confidence ≥ 70%):")
for _, r in rules.iterrows():
    print(f"  {sorted(r.antecedents)} -> {sorted(r.consequents)}  "
          f"support={r.support:.2f} conf={r.confidence:.2f} lift={r.lift:.2f}")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
keranjang_baru = [["minyak goreng"], ["kopi", "roti"], ["mie instan", "beras"], ["kecap"]]
for k in keranjang_baru:
    isi = set(k)
    cocok = rules[rules.antecedents.apply(lambda a: a <= isi) & rules.consequents.apply(lambda c: not (c & isi))]
    if cocok.empty:
        print(f"keranjang {k} -> belum ada aturan yang cocok")
        continue
    r = cocok.sort_values(["lift", "confidence"], ascending=False).iloc[0]
    print(f"keranjang {k} -> tawarkan {sorted(r.consequents)} (aturan {sorted(r.antecedents)} -> {sorted(r.consequents)}, conf {r.confidence:.2f}, lift {r.lift:.2f})")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import simpan
simpan("apriori", df, {
    "beras … telur": f"Satu kolom per jenis barang ({df.shape[1]} barang). True = barang itu ada di struk tersebut.",
}, catatan="Tabel ini dibuat TransactionEncoder dari daftar transaksi di kode; 1 baris = 1 struk belanja. Inilah yang diolah apriori().")
simpan("apriori_baru", {"pembeli": [f"pembeli_{i + 1}" for i in range(len(keranjang_baru))],
                         "isi_keranjang": [", ".join(k) for k in keranjang_baru]},
       {"isi_keranjang": "Barang yang sedang ada di keranjang pembeli baru (belum dibayar)."})
