# Random Forest: ratusan pohon keputusan yang "memilih bersama"
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.25,
                                          stratify=data.target, random_state=42)
pohon = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", oob_score=True,
                            n_jobs=-1, random_state=42).fit(X_tr, y_tr)

print(f"1 Decision Tree   akurasi test = {pohon.score(X_te, y_te):.3f}")
print(f"Random Forest     akurasi test = {rf.score(X_te, y_te):.3f}")
print(f"Out-of-bag score  (validasi gratis tanpa data test) = {rf.oob_score_:.3f}")
print("5 fitur terpenting:")
for imp, nama in sorted(zip(rf.feature_importances_, data.feature_names), reverse=True)[:5]:
    print(f"  {nama:<22} {imp:.3f}")

# === VISUALISASI ===
import numpy as np
from _plot import BLUE, RED, fig, save
ns = [1, 2, 5, 10, 20, 50, 100, 200, 300]
acc = [RandomForestClassifier(n_estimators=n, random_state=42).fit(X_tr, y_tr).score(X_te, y_te) for n in ns]
f, ax = fig()
ax.plot(ns, acc, color=BLUE, marker="o", ms=4, label="Random Forest")
ax.axhline(pohon.score(X_te, y_te), color=RED, ls="--", label="1 Decision Tree")
ax.set_xscale("log")
ax.set(xlabel="jumlah pohon", ylabel="akurasi test", title="Random Forest: makin banyak pohon, makin stabil")
ax.legend()
save("random_forest")
