# Decision Tree (CART): aturan if-else yang bisa dibaca manusia
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

iris = load_iris(as_frame=True)
df = iris.frame.rename(columns={"sepal length (cm)": "sepal_pjg_cm", "sepal width (cm)": "sepal_lbr_cm",
                                "petal length (cm)": "petal_pjg_cm", "petal width (cm)": "petal_lbr_cm",
                                "target": "spesies"})    # 150 bunga; spesies 0 = setosa, 1 = versicolor, 2 = virginica
kolom_fitur = ["sepal_pjg_cm", "sepal_lbr_cm", "petal_pjg_cm", "petal_lbr_cm"]
X, y = df[kolom_fitur].to_numpy(), df["spesies"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
tree = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42).fit(X_tr, y_tr)

print(f"Akurasi test: {accuracy_score(y_te, tree.predict(X_te)):.3f}")
print("Aturan yang dipelajari:")
print(export_text(tree, feature_names=kolom_fitur,
                  class_names=list(iris.target_names)))
for nama, imp in zip(kolom_fitur, tree.feature_importances_):
    print(f"importance {nama:<18}: {imp:.3f}")

# === PREDIKSI DATA BARU ===
print("\n--- prediksi data baru ---")
import pandas as pd
baru = pd.DataFrame({"sepal_pjg_cm": [5.0, 6.0, 6.9], "sepal_lbr_cm": [3.5, 2.7, 3.1],
                     "petal_pjg_cm": [1.4, 4.4, 5.6], "petal_lbr_cm": [0.2, 1.3, 2.2]}, index=["bunga_1", "bunga_2", "bunga_3"])
nama_f, t = kolom_fitur, tree.tree_
for nama, x in zip(baru.index, baru.to_numpy()):
    simpul, jalur = 0, []
    while t.children_left[simpul] != -1:                # telusuri pohon sampai daun
        f, batas = t.feature[simpul], t.threshold[simpul]
        kiri = x[f] <= batas
        jalur.append(f"{nama_f[f]} {x[f]} {'≤' if kiri else '>'} {batas:.2f}")
        simpul = t.children_left[simpul] if kiri else t.children_right[simpul]
    print(f"{nama}: {' → '.join(jalur)} → {iris.target_names[t.value[simpul].argmax()]}")

# === VISUALISASI ===
from sklearn.tree import plot_tree
from _plot import fig, save
f, ax = fig(7.6, 4.2)
ax.grid(False)
plot_tree(tree, feature_names=kolom_fitur,
          class_names=list(iris.target_names), filled=False, fontsize=7, ax=ax, impurity=False)
save("decision_tree")

# === SAMPEL DATA (disimpan ke outputs/ untuk halaman) ===
from _sampel import KET_IRIS, simpan
simpan("decision_tree", df, KET_IRIS, idx=[0, 1, 50, 51, 100, 101],
       catatan="Dataset diurutkan per spesies (50 bunga per spesies), jadi baris contoh diambil dari ketiganya.")
simpan("decision_tree_baru", baru.reset_index(names="bunga"), {"bunga_1 … bunga_3": "Tiga bunga baru yang diukur, spesiesnya belum diketahui."})
