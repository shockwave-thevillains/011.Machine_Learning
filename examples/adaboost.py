# AdaBoost: menggabungkan "weak learner" (stump 1 level) secara berurutan
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# dataset klasik dari buku "Elements of Statistical Learning" (Hastie dkk.)
X, y = make_hastie_10_2(n_samples=12000, random_state=1)
X_tr, y_tr, X_te, y_te = X[:2000], y[:2000], X[2000:], y[2000:]

stump = DecisionTreeClassifier(max_depth=1).fit(X_tr, y_tr)
pohon = DecisionTreeClassifier(max_depth=9, random_state=0).fit(X_tr, y_tr)
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=400,
                         learning_rate=1.0, random_state=0).fit(X_tr, y_tr)

print(f"Error 1 stump (weak learner) : {1 - stump.score(X_te, y_te):.3f}")
print(f"Error pohon kedalaman 9      : {1 - pohon.score(X_te, y_te):.3f}")
err = [1 - (p == y_te).mean() for p in ada.staged_predict(X_te)]
for n in (1, 10, 50, 100, 200, 400):
    print(f"Error AdaBoost {n:>3} stump    : {err[n - 1]:.3f}")

# === VISUALISASI ===
from _plot import BLUE, RED, BLACK, fig, save
f, ax = fig()
ax.plot(range(1, 401), err, color=BLUE, lw=1.6, label="AdaBoost (stump)")
ax.axhline(1 - stump.score(X_te, y_te), color=RED, ls="--", label="1 stump")
ax.axhline(1 - pohon.score(X_te, y_te), color=BLACK, ls=":", label="pohon kedalaman 9")
ax.set(xlabel="jumlah iterasi boosting", ylabel="error test", title="AdaBoost: weak learner menjadi strong learner")
ax.legend()
save("adaboost")
