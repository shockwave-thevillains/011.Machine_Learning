# Decision Tree (CART): aturan if-else yang bisa dibaca manusia
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

iris = load_iris()
X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target, test_size=0.3,
                                          stratify=iris.target, random_state=42)
tree = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42).fit(X_tr, y_tr)

print(f"Akurasi test: {accuracy_score(y_te, tree.predict(X_te)):.3f}")
print("Aturan yang dipelajari:")
print(export_text(tree, feature_names=["sepal_pjg", "sepal_lbr", "petal_pjg", "petal_lbr"],
                  class_names=list(iris.target_names)))
for nama, imp in zip(iris.feature_names, tree.feature_importances_):
    print(f"importance {nama:<18}: {imp:.3f}")

# === VISUALISASI ===
from sklearn.tree import plot_tree
from _plot import fig, save
f, ax = fig(7.6, 4.2)
ax.grid(False)
plot_tree(tree, feature_names=["sepal_pjg", "sepal_lbr", "petal_pjg", "petal_lbr"],
          class_names=list(iris.target_names), filled=False, fontsize=7, ax=ax, impurity=False)
save("decision_tree")
