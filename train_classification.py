"""Task 1 - ML Classification Project (script version of the notebook).
Run from the repo root:  python src/train_classification.py
"""

import matplotlib
matplotlib.use('Agg')


import os, json, warnings, random
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn, joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, classification_report)
from sklearn.inspection import permutation_importance

SEED = 42
random.seed(SEED); np.random.seed(SEED)
sns.set_theme(style="whitegrid", context="notebook")

# Project root works whether run from notebooks/, src/ or the repo root
cwd = Path.cwd()
ROOT = cwd.parent if cwd.name in ("notebooks", "src") else cwd
for d in ("images", "results", "models", "data"):
    (ROOT / d).mkdir(exist_ok=True)

print("scikit-learn", sklearn.__version__, "| pandas", pd.__version__, "| numpy", np.__version__)

# ============================================================
# 1. Load data and inspect
# ============================================================

raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()

# In scikit-learn, target 0 = malignant, 1 = benign. Flip so that 1 = malignant (positive class).
df["target"] = (df["target"] == 0).astype(int)
df["diagnosis"] = df["target"].map({1: "malignant", 0: "benign"})

print("Shape:", df.shape)
print("\nClass balance:")
print(df["diagnosis"].value_counts())
print(df["diagnosis"].value_counts(normalize=True).round(3))
print(df.head())

# ============================================================
# 2. Data preprocessing and cleaning
# ============================================================

print("Missing values (total):", int(df.isna().sum().sum()))
print("Duplicate rows        :", int(df.drop(columns='diagnosis').duplicated().sum()))
print("Non-numeric features  :", [c for c in df.columns if c not in ('diagnosis',) and not np.issubdtype(df[c].dtype, np.number)])

# Cleaning step (no-ops on this dataset but kept so the pipeline is reusable on messier data)
df = df.drop_duplicates().reset_index(drop=True)
df = df.dropna().reset_index(drop=True)

df.drop(columns="diagnosis").to_csv(ROOT / "data" / "breast_cancer.csv", index=False)
print(df.drop(columns=["diagnosis", "target"]).describe().T.head(10).round(3))

# ============================================================
# 3. Exploratory data analysis
# ============================================================

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
order = ["benign", "malignant"]
sns.countplot(data=df, x="diagnosis", order=order, palette=["#4C9F70", "#C8553D"], ax=ax[0])
ax[0].set_title("Class distribution")
for p in ax[0].patches:
    ax[0].annotate(int(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")

feats = ["mean radius", "mean texture", "mean concave points", "mean area"]
melted = df.melt(id_vars="diagnosis", value_vars=feats)
sns.boxplot(data=melted, x="variable", y="value", hue="diagnosis", hue_order=order,
            palette=["#4C9F70", "#C8553D"], ax=ax[1])
ax[1].set_yscale("log"); ax[1].set_xlabel(""); ax[1].set_title("Selected features by class (log scale)")
ax[1].tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(ROOT / "images" / "01_class_and_features.png", dpi=150)
plt.close('all')

fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
for a, f in zip(axes, feats):
    sns.histplot(data=df, x=f, hue="diagnosis", hue_order=order, palette=["#4C9F70", "#C8553D"],
                 bins=30, element="step", ax=a, legend=(a is axes[-1]))
    a.set_title(f)
plt.tight_layout()
plt.savefig(ROOT / "images" / "02_histograms.png", dpi=150)
plt.close('all')

mean_cols = [c for c in df.columns if c.startswith("mean ")]
corr = df[mean_cols + ["target"]].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, cbar_kws={"shrink": .8})
plt.title("Correlation heatmap (mean features + target)")
plt.tight_layout()
plt.savefig(ROOT / "images" / "03_correlation_heatmap.png", dpi=150)
plt.close('all')

print("Top 5 features most correlated with malignancy:")
print(df.drop(columns="diagnosis").corr()["target"].drop("target").abs().sort_values(ascending=False).head(5).round(3))

# ============================================================
# 4. Train / test split
# ============================================================

X = df.drop(columns=["target", "diagnosis"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

print(f"Train: {X_train.shape}  malignant share = {y_train.mean():.3f}")
print(f"Test : {X_test.shape}  malignant share = {y_test.mean():.3f}")

# ============================================================
# 5. Compare algorithms with 5-fold stratified cross-validation
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                                     ("clf", LogisticRegression(max_iter=5000, random_state=SEED))]),
    "Random Forest":       Pipeline([("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]),
    "SVM (RBF)":           Pipeline([("scaler", StandardScaler()),
                                     ("clf", SVC(probability=True, random_state=SEED))]),
    "Gradient Boosting":   Pipeline([("clf", GradientBoostingClassifier(random_state=SEED))]),
}

scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_rows, cv_raw = [], {}
for name, pipe in models.items():
    res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_raw[name] = res
    row = {"Model": name}
    for m in scoring:
        row[m] = f"{res['test_' + m].mean():.3f} +/- {res['test_' + m].std():.3f}"
    cv_rows.append(row)

cv_table = pd.DataFrame(cv_rows).set_index("Model")
cv_table.to_csv(ROOT / "results" / "cv_comparison.csv")
print(cv_table)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, metric in zip(ax, ["f1", "roc_auc"]):
    data = pd.DataFrame({n: r["test_" + metric] for n, r in cv_raw.items()})
    sns.boxplot(data=data, ax=a, palette="Set2")
    sns.stripplot(data=data, ax=a, color="black", size=4)
    a.set_title(f"5-fold CV: {metric.upper() if metric=='f1' else 'ROC-AUC'}")
    a.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(ROOT / "images" / "04_cv_comparison.png", dpi=150)
plt.close('all')

# ============================================================
# 6. Hyper-parameter tuning (GridSearchCV)
# ============================================================

param_grids = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "Random Forest":       {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 6, 12],
                            "clf__min_samples_leaf": [1, 3]},
    "SVM (RBF)":           {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]},
    "Gradient Boosting":   {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                            "clf__max_depth": [2, 3]},
}

tuned, tune_rows = {}, []
for name, pipe in models.items():
    gs = GridSearchCV(pipe, param_grids[name], cv=cv, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    tuned[name] = gs.best_estimator_
    tune_rows.append({"Model": name, "Best CV F1": round(gs.best_score_, 4),
                      "Best params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}})

tune_table = pd.DataFrame(tune_rows).set_index("Model")
tune_table.to_csv(ROOT / "results" / "tuning_results.csv")
print(tune_table)

# ============================================================
# 7. Final evaluation on the held-out test set
# ============================================================

def evaluate(model, X_te, y_te):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    return {"Accuracy": accuracy_score(y_te, pred), "Precision": precision_score(y_te, pred),
            "Recall": recall_score(y_te, pred), "F1": f1_score(y_te, pred),
            "ROC-AUC": roc_auc_score(y_te, proba)}

test_table = pd.DataFrame({n: evaluate(m, X_test, y_test) for n, m in tuned.items()}).T.round(4)
test_table.to_csv(ROOT / "results" / "test_metrics.csv")
print(test_table)

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
for a, (name, m) in zip(axes, tuned.items()):
    ConfusionMatrixDisplay.from_estimator(m, X_test, y_test, display_labels=["benign", "malignant"],
                                          cmap="Blues", colorbar=False, ax=a)
    a.set_title(name, fontsize=10); a.grid(False)
plt.tight_layout()
plt.savefig(ROOT / "images" / "05_confusion_matrices.png", dpi=150)
plt.close('all')

plt.figure(figsize=(6.5, 5.5))
for name, m in tuned.items():
    proba = m.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(y_test, proba):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves on the test set"); plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(ROOT / "images" / "06_roc_curves.png", dpi=150)
plt.close('all')

# ============================================================
# 8. Model selection
# ============================================================

sel = []
for name, m in tuned.items():
    r = cross_validate(m, X_train, y_train, cv=cv, scoring=["f1", "roc_auc"], n_jobs=-1)
    sel.append({"Model": name, "CV F1": r["test_f1"].mean(), "CV ROC-AUC": r["test_roc_auc"].mean()})
sel = pd.DataFrame(sel).set_index("Model").round(4).sort_values(["CV F1", "CV ROC-AUC"], ascending=False)

best_name = sel.index[0]
best_model = tuned[best_name]
print("Selected model:", best_name)
print(sel)

best_pred = best_model.predict(X_test)
print(f"=== {best_name}: test-set report ===")
print(classification_report(y_test, best_pred, target_names=["benign", "malignant"], digits=3))

tn, fp, fn, tp = confusion_matrix(y_test, best_pred).ravel()
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

final = evaluate(best_model, X_test, y_test)
with open(ROOT / "results" / "final_metrics.json", "w") as f:
    json.dump({"model": best_name, "metrics": {k: round(v, 4) for k, v in final.items()},
               "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}, f, indent=2)
print(final)

# ============================================================
# 9. What drives the predictions? (permutation importance)
# ============================================================

pi = permutation_importance(best_model, X_test, y_test, scoring="f1", n_repeats=30,
                            random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False).head(10)
err = pd.Series(pi.importances_std, index=X.columns)[imp.index]

plt.figure(figsize=(8, 4.8))
plt.barh(imp.index[::-1], imp.values[::-1], xerr=err.values[::-1], color="#3B6EA5")
plt.xlabel("Mean drop in F1 when feature is shuffled")
plt.title(f"Top 10 features - {best_name}")
plt.tight_layout()
plt.savefig(ROOT / "images" / "07_feature_importance.png", dpi=150)
plt.close('all')
print(imp.round(4))

# ============================================================
# 10. Save the model and run a sample prediction
# ============================================================

joblib.dump(best_model, ROOT / "models" / "best_model.joblib")

loaded = joblib.load(ROOT / "models" / "best_model.joblib")
sample = X_test.iloc[[0]]
p = loaded.predict_proba(sample)[0, 1]
print("True label       :", "malignant" if y_test.iloc[0] == 1 else "benign")
print("Predicted        :", "malignant" if p >= 0.5 else "benign")
print(f"P(malignant)     : {p:.3f}")

# ============================================================
# 11. Conclusion
# ============================================================

print('\nDone. Outputs saved in images/, results/, models/.')