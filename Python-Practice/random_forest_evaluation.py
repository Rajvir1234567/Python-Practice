import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')

"""
==============================================================
  Random Forest Classification + Evaluation Metrics
  Reference: github.com/Bhavya-Sakhuja/AIML_internship
  Files:     PlacementPredictionUsingRF.py
==============================================================

Dataset: Student Placement Prediction (same as KNN demo)
Features: CGPA, Aptitude Score, Number of Projects
Target:   Placement (Yes / No)

Demonstrates:
  1. Random Forest with Min-Max Scaled features
  2. Full Evaluation Matrix:
       - Accuracy, Precision, Recall, F1-Score
       - Confusion Matrix
       - ROC Curve + AUC Score
       - Cross-Validation Score
  3. Feature Importance Plot
  4. Comparison: Without scaling vs With scaling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    classification_report
)

# ============================================================
#  Dataset  (same as KNN demo for fair comparison)
# ============================================================
data = {
    "CGPA":      [6.5, 7.0, 7.2, 7.5, 7.8, 8.0, 8.2, 8.5, 8.8, 9.0,
                  9.2, 9.5, 6.8, 7.3, 8.1, 8.6, 7.9, 9.1, 6.9, 8.4],
    "Aptitude":  [55,  60,  58,  65,  68,  70,  72,  75,  80,  85,
                  90,  95,  57,  63,  71,  78,  69,  88,  56,  76],
    "Projects":  [1,   2,   1,   2,   2,   3,   3,   4,   4,   5,
                  5,   6,   1,   2,   3,   4,   3,   5,   1,   4],
    "Placement": ["No","No","No","No","No","Yes","Yes","Yes","Yes","Yes",
                  "Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes"]
}

df = pd.DataFrame(data)
features = ["CGPA", "Aptitude", "Projects"]

# Encode target: No=0, Yes=1
le = LabelEncoder()
df["Placement_enc"] = le.fit_transform(df["Placement"])

X = df[features].values
y = df["Placement_enc"].values

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Min-Max Scaling
scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("=" * 65)
print("  Random Forest Classification - Student Placement Prediction")
print("=" * 65)
print(f"\n  Dataset     : {len(df)} students")
print(f"  Features    : {features}")
print(f"  Target      : Placement (No={sum(y==0)}, Yes={sum(y==1)})")
print(f"  Train size  : {len(X_train)}  |  Test size: {len(X_test)}")
print(f"  Scaling     : Min-Max Scaler ([0, 1])")

# ============================================================
#  SECTION 1 - Random Forest WITHOUT Scaling
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 1 - Random Forest  WITHOUT  Min-Max Scaling")
print("=" * 65)

rf_raw = RandomForestClassifier(n_estimators=100, random_state=42)
rf_raw.fit(X_train, y_train)
y_pred_raw  = rf_raw.predict(X_test)
y_proba_raw = rf_raw.predict_proba(X_test)[:, 1]

acc_raw  = accuracy_score(y_test, y_pred_raw)
prec_raw = precision_score(y_test, y_pred_raw, zero_division=0)
rec_raw  = recall_score(y_test, y_pred_raw, zero_division=0)
f1_raw   = f1_score(y_test, y_pred_raw, zero_division=0)
auc_raw  = roc_auc_score(y_test, y_proba_raw)

print(f"\n  Accuracy  : {acc_raw  * 100:.2f}%")
print(f"  Precision : {prec_raw * 100:.2f}%")
print(f"  Recall    : {rec_raw  * 100:.2f}%")
print(f"  F1-Score  : {f1_raw   * 100:.2f}%")
print(f"  AUC Score : {auc_raw  :.4f}")
print("\n  Classification Report:\n")
print(classification_report(y_test, y_pred_raw,
                             target_names=le.classes_, zero_division=0))

# ============================================================
#  SECTION 2 - Random Forest WITH Min-Max Scaling
# ============================================================
print("=" * 65)
print("  SECTION 2 - Random Forest  WITH  Min-Max Scaling")
print("=" * 65)

rf_sc = RandomForestClassifier(n_estimators=100, random_state=42)
rf_sc.fit(X_train_sc, y_train)
y_pred_sc  = rf_sc.predict(X_test_sc)
y_proba_sc = rf_sc.predict_proba(X_test_sc)[:, 1]

acc_sc  = accuracy_score(y_test, y_pred_sc)
prec_sc = precision_score(y_test, y_pred_sc, zero_division=0)
rec_sc  = recall_score(y_test, y_pred_sc, zero_division=0)
f1_sc   = f1_score(y_test, y_pred_sc, zero_division=0)
auc_sc  = roc_auc_score(y_test, y_proba_sc)

print(f"\n  Accuracy  : {acc_sc  * 100:.2f}%")
print(f"  Precision : {prec_sc * 100:.2f}%")
print(f"  Recall    : {rec_sc  * 100:.2f}%")
print(f"  F1-Score  : {f1_sc   * 100:.2f}%")
print(f"  AUC Score : {auc_sc  :.4f}")
print("\n  Classification Report:\n")
print(classification_report(y_test, y_pred_sc,
                             target_names=le.classes_, zero_division=0))

# ============================================================
#  SECTION 3 - Cross-Validation (5-Fold Stratified)
# ============================================================
print("=" * 65)
print("  SECTION 3 - 5-Fold Stratified Cross-Validation (Scaled)")
print("=" * 65)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv = RandomForestClassifier(n_estimators=100, random_state=42)

cv_acc   = cross_val_score(rf_cv, X_train_sc, y_train, cv=cv, scoring="accuracy")
cv_prec  = cross_val_score(rf_cv, X_train_sc, y_train, cv=cv, scoring="precision")
cv_rec   = cross_val_score(rf_cv, X_train_sc, y_train, cv=cv, scoring="recall")
cv_f1    = cross_val_score(rf_cv, X_train_sc, y_train, cv=cv, scoring="f1")

print(f"\n  {'Metric':<12}  {'Fold Scores':<40}  {'Mean':>6}  {'Std':>6}")
print("  " + "-" * 70)
for name, scores in [("Accuracy", cv_acc), ("Precision", cv_prec),
                     ("Recall",   cv_rec),  ("F1-Score", cv_f1)]:
    fold_str = "  ".join([f"{s:.2f}" for s in scores])
    print(f"  {name:<12}  {fold_str:<40}  {scores.mean():.4f}  {scores.std():.4f}")

# ============================================================
#  SECTION 4 - Feature Importance
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 4 - Feature Importance (from RF with scaling)")
print("=" * 65)

importances = rf_sc.feature_importances_
importance_df = pd.DataFrame({
    "Feature":    features,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\n  Feature Importance Ranking:\n")
print(f"  {'Rank':<6}  {'Feature':<14}  {'Importance':>12}  {'Bar'}")
print("  " + "-" * 50)
for i, row in enumerate(importance_df.itertuples(), 1):
    bar = "#" * int(row.Importance * 40)
    print(f"  {i:<6}  {row.Feature:<14}  {row.Importance:>12.4f}  {bar}")

# ============================================================
#  SECTION 5 - Comparison Table: Without vs With Scaling
# ============================================================
print("\n" + "=" * 65)
print("  SECTION 5 - Evaluation Matrix Comparison")
print("=" * 65)

print(f"\n  {'Metric':<12}  {'Without Scaling':>16}  {'With Scaling':>14}  {'Diff':>8}")
print("  " + "-" * 56)
metrics = [
    ("Accuracy",  acc_raw,  acc_sc),
    ("Precision", prec_raw, prec_sc),
    ("Recall",    rec_raw,  rec_sc),
    ("F1-Score",  f1_raw,   f1_sc),
    ("AUC Score", auc_raw,  auc_sc),
]
for name, v1, v2 in metrics:
    diff = v2 - v1
    sign = "+" if diff >= 0 else ""
    print(f"  {name:<12}  {v1*100:>15.2f}%  {v2*100:>13.2f}%  {sign}{diff*100:>6.2f}%")

# ============================================================
#  VISUALISATIONS
# ============================================================

# -- Figure 1: Confusion Matrices --------------------------
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
fig1.suptitle("Random Forest - Confusion Matrices", fontsize=14, fontweight="bold")

ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_raw),
    display_labels=le.classes_
).plot(ax=ax1, colorbar=False, cmap="Oranges")
ax1.set_title("Without Scaling", fontsize=12, color="#C0392B")

ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_sc),
    display_labels=le.classes_
).plot(ax=ax2, colorbar=False, cmap="Greens")
ax2.set_title("With Min-Max Scaling", fontsize=12, color="#27AE60")

plt.tight_layout()
plt.savefig("rf_confusion_matrices.png", dpi=150, bbox_inches="tight")
print("\n  Plot saved -> rf_confusion_matrices.png")
plt.close()

# -- Figure 2: ROC Curve -----------------------------------
fig2, ax = plt.subplots(figsize=(8, 6))

fpr_raw, tpr_raw, _ = roc_curve(y_test, y_proba_raw)
fpr_sc,  tpr_sc,  _ = roc_curve(y_test, y_proba_sc)

ax.plot(fpr_raw, tpr_raw, color="#E07B54", lw=2.5,
        label=f"Without Scaling (AUC = {auc_raw:.3f})")
ax.plot(fpr_sc,  tpr_sc,  color="#4E9AF1", lw=2.5,
        label=f"With Min-Max Scaling (AUC = {auc_sc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier (AUC = 0.500)")

ax.fill_between(fpr_sc, tpr_sc, alpha=0.08, color="#4E9AF1")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve - Random Forest", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("rf_roc_curve.png", dpi=150, bbox_inches="tight")
print("  Plot saved -> rf_roc_curve.png")
plt.close()

# -- Figure 3: Feature Importance Bar Chart ----------------
fig3, ax = plt.subplots(figsize=(8, 5))
colors = ["#4E9AF1", "#2ECC71", "#E07B54"]
bars = ax.barh(importance_df["Feature"], importance_df["Importance"],
               color=colors[:len(features)], edgecolor="white",
               height=0.5, alpha=0.9)

for bar, val in zip(bars, importance_df["Importance"]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=11, fontweight="bold")

ax.set_xlabel("Importance Score", fontsize=12)
ax.set_title("Feature Importance - Random Forest", fontsize=14, fontweight="bold")
ax.set_xlim(0, importance_df["Importance"].max() + 0.08)
ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=150, bbox_inches="tight")
print("  Plot saved -> rf_feature_importance.png")
plt.close()

# -- Figure 4: Evaluation Metrics Bar Chart ----------------
fig4, ax = plt.subplots(figsize=(10, 5))
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC Score"]
vals_raw = [acc_raw, prec_raw, rec_raw, f1_raw, auc_raw]
vals_sc  = [acc_sc,  prec_sc,  rec_sc,  f1_sc,  auc_sc]

x     = np.arange(len(metric_names))
width = 0.35

b1 = ax.bar(x - width / 2, [v * 100 for v in vals_raw], width,
            label="Without Scaling", color="#E07B54", alpha=0.88, edgecolor="white")
b2 = ax.bar(x + width / 2, [v * 100 for v in vals_sc],  width,
            label="With Min-Max Scaling", color="#4E9AF1", alpha=0.88, edgecolor="white")

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
            f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Evaluation Metrics Comparison - Random Forest", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylim(0, 115)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("rf_metrics_comparison.png", dpi=150, bbox_inches="tight")
print("  Plot saved -> rf_metrics_comparison.png")
plt.close()

# -- Figure 5: Cross-Validation Scores --------------------
fig5, ax = plt.subplots(figsize=(9, 5))
cv_data   = [cv_acc, cv_prec, cv_rec, cv_f1]
cv_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
cv_colors = ["#4E9AF1", "#2ECC71", "#E07B54", "#9B59B6"]

bp = ax.boxplot(cv_data, patch_artist=True, notch=False,
                medianprops=dict(color="white", linewidth=2.5))

for patch, color in zip(bp["boxes"], cv_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

for whisker in bp["whiskers"]:
    whisker.set(color="gray", linewidth=1.5)
for cap in bp["caps"]:
    cap.set(color="gray", linewidth=1.5)
for flier in bp["fliers"]:
    flier.set(marker="o", color="red", alpha=0.6)

ax.set_xticklabels(cv_labels, fontsize=11)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("5-Fold Cross-Validation Score Distribution", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for i, (scores, color) in enumerate(zip(cv_data, cv_colors), 1):
    ax.text(i, scores.mean() + 0.03, f"Mean={scores.mean():.2f}",
            ha="center", fontsize=9, color=color, fontweight="bold")

plt.tight_layout()
plt.savefig("rf_cross_validation.png", dpi=150, bbox_inches="tight")
print("  Plot saved -> rf_cross_validation.png")
plt.close()

print("\n" + "=" * 65)
print("  Random Forest + Evaluation Matrix - Complete!")
print("=" * 65)
