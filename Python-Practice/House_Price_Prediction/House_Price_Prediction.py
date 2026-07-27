# ============================================================
#   House Price Prediction - Machine Learning Project
# ============================================================
#   Name      : Rajvir Singh
#   ID        : GU-2024-1545
#   Course    : BCA
#   Semester  : 5
#   Section   : C
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import os
import warnings

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------
DATASET_PATH   = "Housing.csv"
MODEL_PATH     = "models/house_price_model.pkl"
RESULTS_PATH   = "results/predictions.csv"
PLOTS_DIR      = "plots"

# -----------------------------------------------------------------
# STEP 1 - LOAD DATA
# -----------------------------------------------------------------

def load_data(path):
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# -----------------------------------------------------------------
# STEP 2 - EXPLORATORY DATA ANALYSIS
# -----------------------------------------------------------------

def explore_data(df):
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nDataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nStatistical Summary:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())


# -----------------------------------------------------------------
# STEP 3 - DATA PREPROCESSING
# -----------------------------------------------------------------

def preprocess_data(df):
    df = df.copy()
    before = len(df)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print(f"Rows after cleaning: {len(df)} (removed {before - len(df)})")

    binary_cols = [
        "mainroad", "guestroom", "basement",
        "hotwaterheating", "airconditioning", "prefarea"
    ]
    for col in binary_cols:
        df[col] = df[col].map({"yes": 1, "no": 0})

    furnish_map = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
    df["furnishingstatus"] = df["furnishingstatus"].map(furnish_map)
    print("Preprocessing complete.")
    return df


# -----------------------------------------------------------------
# STEP 4 - FEATURE ENGINEERING
# -----------------------------------------------------------------

def feature_engineering(df):
    df = df.copy()
    df["price_per_sqft"] = df["price"] / df["area"]
    df["luxury_score"] = (
        df["airconditioning"] +
        df["furnishingstatus"] +
        df["prefarea"] +
        df["guestroom"] +
        df["hotwaterheating"]
    )
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    print("Feature engineering complete.")
    return df


# -----------------------------------------------------------------
# STEP 5 - TRAIN / TEST SPLIT
# -----------------------------------------------------------------

def split_data(df):
    drop_cols = ["price", "price_per_sqft"]
    X = df.drop(columns=drop_cols)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------
# STEP 6 - TRAIN MULTIPLE MODELS
# -----------------------------------------------------------------

def train_models(X_train, y_train):
    models = {
        "Linear Regression"   : LinearRegression(),
        "Ridge Regression"    : Ridge(alpha=1.0),
        "Lasso Regression"    : Lasso(alpha=1000),
        "Decision Tree"       : DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest"       : RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting"   : GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  Trained: {name}")
    return trained


# -----------------------------------------------------------------
# STEP 7 - EVALUATE MODELS
# -----------------------------------------------------------------

def evaluate_models(models, X_test, y_test):
    records = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        records.append({
            "Model"    : name,
            "MAE"      : round(mae, 2),
            "RMSE"     : round(rmse, 2),
            "R2_Score" : round(r2, 4),
        })
    results_df = pd.DataFrame(records).sort_values("R2_Score", ascending=False).reset_index(drop=True)
    print("\nModel Evaluation Results:")
    print(results_df.to_string(index=False))
    return results_df


# -----------------------------------------------------------------
# STEP 8 - SAVE BEST MODEL
# -----------------------------------------------------------------

def save_best_model(models, results_df, X_test, y_test):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    best_name  = results_df.iloc[0]["Model"]
    best_model = models[best_name]
    y_pred     = best_model.predict(X_test)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best model '{best_name}' saved -> {MODEL_PATH}")

    pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": np.round(y_pred, 2)})
    pred_df.to_csv(RESULTS_PATH, index=False)
    print(f"Predictions saved -> {RESULTS_PATH}")
    return best_name, best_model, y_pred


# -----------------------------------------------------------------
# STEP 9 - VISUALIZATION
# -----------------------------------------------------------------

def create_visualizations(df, models, results_df, X_test, y_test, y_pred, best_name):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    BG    = "#0f1117"
    CARD  = "#1a1d2e"
    ACCENT= "#7c3aed"
    GOLD  = "#fbbf24"
    TEAL  = "#06b6d4"

    plt.rcParams.update({
        "figure.facecolor" : BG,
        "axes.facecolor"   : CARD,
        "axes.edgecolor"   : "#374151",
        "axes.labelcolor"  : "#e5e7eb",
        "xtick.color"      : "#9ca3af",
        "ytick.color"      : "#9ca3af",
        "text.color"       : "#e5e7eb",
        "grid.color"       : "#374151",
        "grid.linewidth"   : 0.5,
    })

    # Plot 1: Price Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["price"] / 1e6, bins=30, color=ACCENT, edgecolor=BG, alpha=0.85)
    ax.set_title("House Price Distribution", fontsize=16, fontweight="bold", color=GOLD)
    ax.set_xlabel("Price (Millions INR)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/01_price_distribution.png", dpi=150)
    plt.close()

    # Plot 2: Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", color=GOLD)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/02_correlation_heatmap.png", dpi=150)
    plt.close()

    # Plot 3: Area vs Price
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(df["area"], df["price"] / 1e6,
                    c=df["bedrooms"], cmap="plasma", alpha=0.7, s=40)
    fig.colorbar(sc, ax=ax, label="Bedrooms")
    ax.set_title("Area vs House Price (colored by Bedrooms)", fontsize=14, fontweight="bold", color=GOLD)
    ax.set_xlabel("Area (sq ft)", fontsize=12)
    ax.set_ylabel("Price (Millions INR)", fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/03_area_vs_price.png", dpi=150)
    plt.close()

    # Plot 4: Model Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = [ACCENT, TEAL, "#f472b6", "#34d399", "#fb923c", GOLD]
    axes[0].barh(results_df["Model"], results_df["R2_Score"],
                 color=colors[:len(results_df)], edgecolor=BG, height=0.6)
    axes[0].set_title("R2 Score Comparison", fontsize=14, fontweight="bold", color=GOLD)
    axes[0].set_xlabel("R2 Score")
    axes[1].barh(results_df["Model"], results_df["RMSE"] / 1e6,
                 color=colors[:len(results_df)], edgecolor=BG, height=0.6)
    axes[1].set_title("RMSE Comparison (Millions INR)", fontsize=14, fontweight="bold", color=GOLD)
    axes[1].set_xlabel("RMSE (Millions INR)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/04_model_comparison.png", dpi=150)
    plt.close()

    # Plot 5: Actual vs Predicted
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(y_test / 1e6, y_pred / 1e6, color=TEAL, alpha=0.65, s=50, label="Predictions")
    min_val = min(y_test.min(), y_pred.min()) / 1e6
    max_val = max(y_test.max(), y_pred.max()) / 1e6
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit")
    ax.set_title(f"Actual vs Predicted Prices ({best_name})", fontsize=14, fontweight="bold", color=GOLD)
    ax.set_xlabel("Actual Price (Millions INR)", fontsize=12)
    ax.set_ylabel("Predicted Price (Millions INR)", fontsize=12)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/05_actual_vs_predicted.png", dpi=150)
    plt.close()

    # Plot 6: Residuals
    residuals = y_test.values - y_pred
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_pred / 1e6, residuals / 1e6, color=GOLD, alpha=0.6, s=40)
    ax.axhline(0, color="red", linestyle="--", lw=2)
    ax.set_title("Residual Plot", fontsize=14, fontweight="bold", color=GOLD)
    ax.set_xlabel("Predicted Price (Millions INR)", fontsize=12)
    ax.set_ylabel("Residuals (Millions INR)", fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/06_residuals.png", dpi=150)
    plt.close()

    # Plot 7: Feature Importance
    best_model = models[best_name]
    feature_names = list(df.drop(columns=["price", "price_per_sqft"]).columns)
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": imp}).sort_values("Importance", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(fi_df["Feature"], fi_df["Importance"], color=ACCENT, edgecolor=BG, height=0.65)
        ax.set_title(f"Feature Importance ({best_name})", fontsize=14, fontweight="bold", color=GOLD)
        ax.set_xlabel("Importance Score")
        ax.grid(True, axis="x")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/07_feature_importance.png", dpi=150)
        plt.close()
    elif hasattr(best_model, "coef_"):
        coef = best_model.coef_
        fi_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coef}).sort_values("Coefficient", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_colors = [ACCENT if c >= 0 else "#ef4444" for c in fi_df["Coefficient"]]
        ax.barh(fi_df["Feature"], fi_df["Coefficient"], color=bar_colors, edgecolor=BG, height=0.65)
        ax.axvline(0, color="#e5e7eb", lw=1)
        ax.set_title(f"Feature Coefficients ({best_name})", fontsize=14, fontweight="bold", color=GOLD)
        ax.set_xlabel("Coefficient Value")
        ax.grid(True, axis="x")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/07_feature_importance.png", dpi=150)
        plt.close()

    # Plot 8: Avg Price by Furnishing Status
    furnish_labels = {0: "Unfurnished", 1: "Semi-Furnished", 2: "Furnished"}
    furnish_avg = df.groupby("furnishingstatus")["price"].mean().rename(index=furnish_labels) / 1e6
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(furnish_avg.index, furnish_avg.values,
                  color=[TEAL, ACCENT, GOLD], edgecolor=BG, width=0.5)
    for bar, val in zip(bars, furnish_avg.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f"INR {val:.2f}M", ha="center", fontsize=10)
    ax.set_title("Average Price by Furnishing Status", fontsize=14, fontweight="bold", color=GOLD)
    ax.set_ylabel("Average Price (Millions INR)")
    ax.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/08_price_by_furnishing.png", dpi=150)
    plt.close()

    print(f"All plots saved in '{PLOTS_DIR}/' directory.")


# -----------------------------------------------------------------
# STEP 10 - PREDICT NEW HOUSE
# -----------------------------------------------------------------

def predict_new_house(model, feature_names):
    sample = {
        "area"             : 5000,
        "bedrooms"         : 3,
        "bathrooms"        : 2,
        "stories"          : 2,
        "mainroad"         : 1,
        "guestroom"        : 0,
        "basement"         : 1,
        "hotwaterheating"  : 0,
        "airconditioning"  : 1,
        "parking"          : 1,
        "prefarea"         : 0,
        "furnishingstatus" : 1,
        "luxury_score"     : 2,
        "total_rooms"      : 5,
    }
    input_df = pd.DataFrame([sample])[feature_names]
    price    = model.predict(input_df)[0]

    print("\nSAMPLE PREDICTION - NEW HOUSE")
    print("-" * 40)
    for k, v in sample.items():
        print(f"  {k:25s}: {v}")
    print(f"\n  Predicted Price: INR {price:,.2f}  (~INR {price/1e6:.2f} Million)")


# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  HOUSE PRICE PREDICTION - MINI PROJECT")
    print("  Rajvir Singh | GU-2024-1545 | BCA Sem 5 Sec C")
    print("=" * 60)

    df = load_data(DATASET_PATH)
    explore_data(df)
    df = preprocess_data(df)
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test = split_data(df)
    models = train_models(X_train, y_train)
    results_df = evaluate_models(models, X_test, y_test)
    best_name, best_model, y_pred = save_best_model(models, results_df, X_test, y_test)
    create_visualizations(df, models, results_df, X_test, y_test, y_pred, best_name)
    predict_new_house(best_model, list(X_train.columns))

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print(f"  Best Model : {best_name}")
    print(f"  R2 Score   : {results_df.iloc[0]['R2_Score']}")
    print(f"  RMSE       : INR {results_df.iloc[0]['RMSE']:,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
