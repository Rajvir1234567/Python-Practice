import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')  # Non-interactive - saves to files without display

"""
==============================================================
  KNN Classification WITH Min-Max Feature Scaling
  Reference: github.com/Bhavya-Sakhuja/AIML_internship
  Files:     placementPredictionUsingKNN.py
             performanceDetectbyKNN.py
==============================================================

Dataset: Student Placement Prediction
Features: CGPA, Aptitude Score, Number of Projects
Target:   Placement (Yes / No)

Demonstrates:
  1. KNN WITHOUT scaling  -> shows bias toward high-range features
  2. KNN WITH MinMaxScaler -> shows fair, improved classification
  3. Decision-boundary visualisation for both cases
  4. Best-k selection using the Elbow method
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# --------------------------------------------------------
#  Dataset  (consistent with the internship repo style)
# --------------------------------------------------------
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

# Encode target
le = LabelEncoder()
df["Placement_enc"] = le.fit_transform(df["Placement"])   # No=0, Yes=1

X = df[features].values
y = df["Placement_enc"].values

# Train / Test split (fixed seed for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ============================================================
#  SECTION 1 - KNN  WITHOUT  Feature Scaling
# ============================================================
print("=" * 62)
print("  SECTION 1 - KNN  WITHOUT  Min-Max Scaling  (k=5)")
print("=" * 62)

knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
y_pred_raw = knn_raw.predict(X_test)

acc_raw = accuracy_score(y_test, y_pred_raw)
print(f"\n  Test Accuracy (no scaling) : {acc_raw * 100:.2f}%\n")
print("  Classification Report:\n")
print(classification_report(y_test, y_pred_raw,
                             target_names=le.classes_, zero_division=0))

# ============================================================
#  SECTION 2 - Apply Min-Max Scaling, then KNN
# ============================================================
print("=" * 62)
print("  SECTION 2 - KNN  WITH  Min-Max Scaling  (k=5)")
print("=" * 62)

scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit ONLY on training data
X_test_sc  = scaler.transform(X_test)        # transform test with same params

print("\n  Scaled training features (first 5 rows):")
df_scaled_preview = pd.DataFrame(X_train_sc, columns=features)
print(df_scaled_preview.head().to_string(index=False))

knn_sc = KNeighborsClassifier(n_neighbors=5)
knn_sc.fit(X_train_sc, y_train)
y_pred_sc = knn_sc.predict(X_test_sc)

acc_sc = accuracy_score(y_test, y_pred_sc)
print(f"\n  Test Accuracy (with scaling): {acc_sc * 100:.2f}%\n")
print("  Classification Report:\n")
print(classification_report(y_test, y_pred_sc,
                             target_names=le.classes_, zero_division=0))

improvement = (acc_sc - acc_raw) * 100
print(f"  Accuracy Improvement: {improvement:+.2f} percentage points")

# ============================================================
#  SECTION 3 - Elbow / Best-k  (with scaling)
# ============================================================
print("\n" + "=" * 62)
print("  SECTION 3 - Best k Selection  (Elbow Method)")
print("=" * 62)

k_range     = range(1, 11)
train_accs  = []
test_accs   = []

for k in k_range:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_sc, y_train)
    train_accs.append(accuracy_score(y_train, knn_k.predict(X_train_sc)))
    test_accs.append(accuracy_score(y_test,  knn_k.predict(X_test_sc)))

best_k   = list(k_range)[test_accs.index(max(test_accs))]
best_acc = max(test_accs)

print(f"\n  {'k':>4}  {'Train Acc':>10}  {'Test Acc':>10}")
print("  " + "-" * 30)
for k, tr, te in zip(k_range, train_accs, test_accs):
    marker = " <- best" if k == best_k else ""
    print(f"  {k:>4}  {tr*100:>9.1f}%  {te*100:>9.1f}%{marker}")

print(f"\n  Best k = {best_k}  ->  Test Accuracy = {best_acc*100:.2f}%")

# ============================================================
#  SECTION 4 - Predict a new student
# ============================================================
print("\n" + "=" * 62)
print("  SECTION 4 - Predict a New Student")
print("=" * 62)

new_student = pd.DataFrame({"CGPA": [8.3], "Aptitude": [74], "Projects": [4]})
new_scaled  = scaler.transform(new_student[features])

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_sc, y_train)

prediction = knn_best.predict(new_scaled)
proba      = knn_best.predict_proba(new_scaled)[0]

print(f"\n  Student Profile  ->  CGPA=8.3  |  Aptitude=74  |  Projects=4")
print(f"\n  Scaled values    ->  {new_scaled[0]}")
print(f"\n  Prediction       ->  {le.inverse_transform(prediction)[0]}")
print(f"  Probabilities    ->  No={proba[0]*100:.1f}%   Yes={proba[1]*100:.1f}%")

# ============================================================
#  VISUALISATIONS
# ============================================================

# -- Figure 1: Confusion Matrices (side by side) -----------
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
fig1.suptitle("Confusion Matrices - KNN (k=5)", fontsize=14, fontweight="bold")

ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_raw),
    display_labels=le.classes_
).plot(ax=ax1, colorbar=False, cmap="Blues")
ax1.set_title("Without Scaling", fontsize=12, color="#C0392B")

ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_sc),
    display_labels=le.classes_
).plot(ax=ax2, colorbar=False, cmap="Greens")
ax2.set_title("With Min-Max Scaling", fontsize=12, color="#27AE60")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
print("\n  Plot saved -> confusion_matrices.png")

# -- Figure 2: Elbow Curve ---------------------------------
fig2, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(k_range), [a * 100 for a in train_accs],
        "o-", color="#4E9AF1", lw=2, label="Train Accuracy")
ax.plot(list(k_range), [a * 100 for a in test_accs],
        "s-", color="#E07B54", lw=2, label="Test Accuracy")
ax.axvline(x=best_k, color="#2ECC71", lw=2, linestyle="--",
           label=f"Best k = {best_k}")
ax.set_xlabel("k (Number of Neighbours)", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Elbow Method - Choosing the Best k", fontsize=14, fontweight="bold")
ax.set_xticks(list(k_range))
ax.set_ylim(40, 105)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("knn_elbow_curve.png", dpi=150, bbox_inches="tight")
print("  Plot saved -> knn_elbow_curve.png")

# -- Figure 3: 2-D Decision Boundary ----------------------
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle("KNN Decision Boundary  (CGPA vs Aptitude)",
              fontsize=14, fontweight="bold")

def plot_boundary(ax, X2d_train, X2d_test, y_train_, y_test_, k, title):
    h = 0.005
    xx, yy = np.meshgrid(
        np.arange(X2d_train[:, 0].min() - 0.05, X2d_train[:, 0].max() + 0.05, h),
        np.arange(X2d_train[:, 1].min() - 0.05, X2d_train[:, 1].max() + 0.05, h),
    )
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X2d_train, y_train_)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    cmap_bg = mcolors.LinearSegmentedColormap.from_list("bg", ["#FAD7D0", "#D0E8FA"])
    ax.contourf(xx, yy, Z, alpha=0.45, cmap=cmap_bg)

    scatter_colors = ["#E74C3C" if lbl == 0 else "#2ECC71" for lbl in y_train_]
    ax.scatter(X2d_train[:, 0], X2d_train[:, 1],
               c=scatter_colors, edgecolors="k", s=80, label="Train", zorder=3)
    scatter_test = ["#E74C3C" if lbl == 0 else "#2ECC71" for lbl in y_test_]
    ax.scatter(X2d_test[:, 0], X2d_test[:, 1],
               c=scatter_test, edgecolors="navy", s=120, marker="*",
               label="Test", zorder=4)

    acc_ = accuracy_score(y_test_, clf.predict(X2d_test))
    ax.set_title(f"{title}\n(Accuracy: {acc_*100:.1f}%)", fontsize=12)
    ax.set_xlabel("CGPA (Feature 1)")
    ax.set_ylabel("Aptitude (Feature 2)")
    leg_elements = [
        Patch(facecolor="#E74C3C", edgecolor="k", label="No Placement"),
        Patch(facecolor="#2ECC71", edgecolor="k", label="Placed"),
    ]
    ax.legend(handles=leg_elements, loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)

# Without scaling
X2d_raw_train = X_train[:, :2]
X2d_raw_test  = X_test[:, :2]
plot_boundary(axes3[0], X2d_raw_train, X2d_raw_test, y_train, y_test,
              k=5, title="WITHOUT Feature Scaling")

# With scaling
X2d_sc_train = X_train_sc[:, :2]
X2d_sc_test  = X_test_sc[:, :2]
plot_boundary(axes3[1], X2d_sc_train, X2d_sc_test, y_train, y_test,
              k=5, title="WITH Min-Max Scaling")

plt.tight_layout()
plt.savefig("knn_decision_boundary.png", dpi=150, bbox_inches="tight")
print("  Plot saved -> knn_decision_boundary.png")
plt.close('all')

print("\n" + "=" * 62)
print("  All tasks completed successfully!")
print("=" * 62)
