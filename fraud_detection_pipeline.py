

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  FRAUD DETECTION PIPELINE — DecodeLabs Project 2")
print("=" * 60)

# ── STEP 1: Load Dataset ────────────────────────────────────
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv("creditcard.csv")
print(f"  Total transactions : {len(df):,}")
print(f"  Legitimate (0)     : {df['Class'].value_counts()[0]:,}  (99.83%)")
print(f"  Fraudulent  (1)    : {df['Class'].value_counts()[1]:,}  (0.17%)")
print(f"  Features           : {df.shape[1] - 1}")

# ── STEP 2: EDA Chart ───────────────────────────────────────
print("\n[STEP 2] Class distribution chart...")
plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df, palette=["steelblue", "tomato"])
plt.title("Class Distribution (0=Legitimate, 1=Fraud)")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.close()
print("  → Saved: class_distribution.png")

# ── STEP 3: Feature / Target Split ──────────────────────────
print("\n[STEP 3] Splitting features and target...")
X = df.drop(columns=["Class"])
y = df["Class"]

# ── STEP 4: Train/Test Split (Stratified) ───────────────────
print("\n[STEP 4] Stratified Train/Test Split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,} | Fraud in train: {y_train.sum()}")
print(f"  Test : {len(X_test):,}  | Fraud in test : {y_test.sum()}")

# ── STEP 5: SMOTE (only on training data) ───────────────────
print("\n[STEP 5] Applying SMOTE on training data only...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"  Before → Fraud: {y_train.sum()} | Legit: {(y_train==0).sum()}")
print(f"  After  → Fraud: {y_train_res.sum()} | Legit: {(y_train_res==0).sum()}")

# ── STEP 6: Scale (for Logistic Regression) ─────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════
#  MODEL A — Logistic Regression
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  MODEL A: Logistic Regression")
print("=" * 60)

lr = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train_res)

y_pred_lr  = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

print(f"  Precision : {precision_score(y_test, y_pred_lr):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred_lr):.4f}")
print(f"  F1 Score  : {f1_score(y_test, y_pred_lr):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba_lr):.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=["Legitimate", "Fraud"]))

# ══════════════════════════════════════════════════════════════
#  MODEL B — Random Forest (fast: 10 trees)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  MODEL B: Random Forest")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=10,   # fast but effective
    max_depth=10,
    random_state=42,
    n_jobs=1
)
rf.fit(X_train_res, y_train_res)

y_pred_rf  = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print(f"  Precision : {precision_score(y_test, y_pred_rf):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred_rf):.4f}")
print(f"  F1 Score  : {f1_score(y_test, y_pred_rf):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba_rf):.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=["Legitimate", "Fraud"]))

# ── STEP 7: Confusion Matrices ──────────────────────────────
print("\n[STEP 7] Saving charts...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit","Fraud"],
            yticklabels=["Legit","Fraud"], ax=axes[0])
axes[0].set_title("Confusion Matrix — Logistic Regression")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["Legit","Fraud"],
            yticklabels=["Legit","Fraud"], ax=axes[1])
axes[1].set_title("Confusion Matrix — Random Forest")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()
print("  → Saved: confusion_matrices.png")

# ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, color="steelblue", lw=2,
         label=f"Logistic Regression (AUC={roc_auc_score(y_test, y_proba_lr):.4f})")
plt.plot(fpr_rf, tpr_rf, color="darkorange", lw=2,
         label=f"Random Forest       (AUC={roc_auc_score(y_test, y_proba_rf):.4f})")
plt.plot([0,1],[0,1],"k--", lw=1, label="Random (AUC=0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.close()
print("  → Saved: roc_curves.png")

# Feature Importance
importances = rf.feature_importances_
feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=False).head(15)
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis")
plt.title("Top 15 Feature Importances — Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("  → Saved: feature_importance.png")

# ── STEP 8: Final Comparison Table ──────────────────────────
print("\n" + "=" * 60)
print("  FINAL MODEL COMPARISON")
print("=" * 60)

results = pd.DataFrame({
    "Model":     ["Logistic Regression", "Random Forest"],
    "Precision": [round(precision_score(y_test, y_pred_lr),4), round(precision_score(y_test, y_pred_rf),4)],
    "Recall":    [round(recall_score(y_test, y_pred_lr),4),    round(recall_score(y_test, y_pred_rf),4)],
    "F1 Score":  [round(f1_score(y_test, y_pred_lr),4),        round(f1_score(y_test, y_pred_rf),4)],
    "ROC-AUC":   [round(roc_auc_score(y_test, y_proba_lr),4),  round(roc_auc_score(y_test, y_proba_rf),4)],
})
print(results.to_string(index=False))
results.to_csv("model_comparison.csv", index=False)
print("\n  → Saved: model_comparison.csv")
print("=" * 60)
