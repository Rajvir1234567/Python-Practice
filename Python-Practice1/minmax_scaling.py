import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - saves to file without display

"""
==============================================================
  Min-Max Feature Scaling - Full Demonstration
  Reference: github.com/Bhavya-Sakhuja/AIML_internship
==============================================================

Formula:
    X_scaled = (X - X_min) / (X_max - X_min)

Result: Every feature is mapped to the range [0, 1]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------------
#  Sample dataset: Student Placement Records
# --------------------------------------------------------
data = {
    "Student":    ["Alice", "Bob",  "Carol", "David", "Eve",
                   "Frank", "Grace","Hank",  "Iris",  "Jake"],
    "CGPA":       [6.5,  7.0,  7.5,  8.0,  8.2,
                   8.5,  8.8,  9.0,  9.2,  9.5],
    "Aptitude":   [55,   60,   65,   70,   72,
                   75,   80,   85,   90,   95],
    "Projects":   [1,    2,    2,    3,    3,
                   4,    4,    5,    5,    6],
    "Placement":  ["No","No","No","Yes","Yes",
                   "Yes","Yes","Yes","Yes","Yes"]
}

df = pd.DataFrame(data)
features = ["CGPA", "Aptitude", "Projects"]

# ============================================================
#  PART A  -  Manual Min-Max Scaling (from scratch)
# ============================================================
print("=" * 60)
print("  PART A - Manual Min-Max Scaling")
print("=" * 60)
print("\n  Formula: X_scaled = (X - X_min) / (X_max - X_min)\n")
print("Original DataFrame:\n")
print(df[["Student"] + features].to_string(index=False))

df_manual = df[features].copy()

for col in features:
    col_min = df[col].min()
    col_max = df[col].max()
    df_manual[col + "_scaled"] = (df[col] - col_min) / (col_max - col_min)

print("\n\nManually Scaled Features (0 to 1):\n")
scaled_cols = [c + "_scaled" for c in features]
display_df = pd.concat([df["Student"], df_manual[scaled_cols]], axis=1)
display_df.columns = ["Student"] + features
print(display_df.to_string(index=False))

# ============================================================
#  PART B  -  sklearn  MinMaxScaler
# ============================================================
print("\n" + "=" * 60)
print("  PART B - sklearn MinMaxScaler")
print("=" * 60)

scaler = MinMaxScaler()
X_raw    = df[features].values
X_scaled = scaler.fit_transform(X_raw)

df_sklearn = pd.DataFrame(X_scaled, columns=features)
df_sklearn.insert(0, "Student", df["Student"])

print("\nsklearn MinMaxScaler output:\n")
print(df_sklearn.to_string(index=False))

print("\n--- Scaler parameters ---")
for col, mn, mx in zip(features, scaler.data_min_, scaler.data_max_):
    print(f"  {col:<12}  min={mn:.2f}  max={mx:.2f}  range={mx-mn:.2f}")

# ============================================================
#  PART C  -  Verification: both methods are identical
# ============================================================
print("\n" + "=" * 60)
print("  PART C - Verification (Manual == sklearn)")
print("=" * 60)
manual_vals  = df_manual[scaled_cols].values
sklearn_vals = X_scaled
match = np.allclose(manual_vals, sklearn_vals, atol=1e-8)
if match:
    print("\n  Results identical: True  [PASS]")
else:
    print("\n  Mismatch found [FAIL]")

# ============================================================
#  PART D  -  Visualisation
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Min-Max Scaling: Before vs After", fontsize=15, fontweight="bold", y=1.02)

palette_before = "#E07B54"
palette_after  = "#4E9AF1"

for idx, col in enumerate(features):
    ax = axes[idx]
    before = df[col].values
    after  = X_scaled[:, idx]
    x      = np.arange(len(before))
    width  = 0.38

    ax.bar(x - width / 2, before, width, label="Before", color=palette_before,
           alpha=0.88, edgecolor="white")
    ax.bar(x + width / 2, after,  width, label="After",  color=palette_after,
           alpha=0.88, edgecolor="white")

    ax.set_title(col, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Student"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.97, 0.97,
            f"Range (before): {before.min():.1f}-{before.max():.1f}\n"
            f"Range (after): 0.00-1.00",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

legend_patches = [
    mpatches.Patch(color=palette_before, label="Before scaling"),
    mpatches.Patch(color=palette_after,  label="After Min-Max scaling"),
]
fig.legend(handles=legend_patches, loc="upper center",
           ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.05))

plt.tight_layout()
plt.savefig("minmax_scaling_plot.png", dpi=150, bbox_inches="tight")
print("\n  Plot saved -> minmax_scaling_plot.png")
plt.close('all')

print("\n" + "=" * 60)
print("  Min-Max Scaling demo complete!")
print("=" * 60)
