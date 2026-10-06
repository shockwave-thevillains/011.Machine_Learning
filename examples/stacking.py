# Stacking (Stacked Generalization): meta-model belajar menggabungkan beberapa model
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
base = [
    ("rf", RandomForestClassifier(n_estimators=200, random_state=0)),
    ("svm", make_pipeline(StandardScaler(), SVC(random_state=0))),
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))),
]
stack = StackingClassifier(estimators=base, final_estimator=LogisticRegression(), cv=5)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
for nama, m in base + [("STACKING", stack)]:
    s = cross_val_score(m, X, y, cv=cv)
    print(f"{nama:<9} akurasi 10-fold = {s.mean():.4f} ± {s.std():.4f}")

stack.fit(X, y)
print("Bobot meta-model (RF, SVM-skor, KNN):", stack.final_estimator_.coef_.round(2).tolist()[0])
