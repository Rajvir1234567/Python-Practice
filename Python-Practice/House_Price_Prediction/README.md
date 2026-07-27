# 🏠 House Price Prediction — Mini Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 👤 Student Details

| Field        | Details        |
|--------------|----------------|
| **Name**     | Rajvir Singh   |
| **ID**       | GU-2024-1545   |
| **Course**   | BCA            |
| **Semester** | 5              |
| **Section**  | C              |

---

## 📌 Project Overview

This is an end-to-end **Machine Learning** project that predicts **house prices** in Indian Rupees using various regression algorithms. It demonstrates the complete ML workflow from data loading, exploratory analysis, feature engineering, model training & evaluation, to saving the trained model.

### Input Features Used:
| Feature | Description |
|---------|-------------|
| `area` | Total area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of stories |
| `mainroad` | Connected to main road (yes/no) |
| `guestroom` | Guest room available (yes/no) |
| `basement` | Basement available (yes/no) |
| `hotwaterheating` | Hot water heating (yes/no) |
| `airconditioning` | Air conditioning (yes/no) |
| `parking` | Number of parking spaces |
| `prefarea` | Located in preferred area (yes/no) |
| `furnishingstatus` | unfurnished / semi-furnished / furnished |

### Engineered Features:
| Feature | Description |
|---------|-------------|
| `luxury_score` | Combined luxury amenities score |
| `total_rooms` | Total count of bedrooms + bathrooms |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python 3.x |
| Data Analysis | Pandas |
| Numerical Computing | NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter Notebook |
| Model Saving | Pickle |

---

## ⚙️ Machine Learning Workflow

```
  Housing.csv
       │
       ▼
  Data Loading
       │
       ▼
  Data Cleaning & Preprocessing
       │
       ▼
  Feature Engineering
       │
       ▼
  Train-Test Split (80/20)
       │
       ▼
  ┌────────────────────────────────────────┐
  │   Train 6 Regression Models:          │
  │   ◆ Linear Regression                 │
  │   ◆ Ridge Regression                  │
  │   ◆ Lasso Regression                  │
  │   ◆ Decision Tree                     │
  │   ◆ Random Forest                     │
  │   ◆ Gradient Boosting                 │
  └────────────────────────────────────────┘
       │
       ▼
  Model Evaluation (MAE, RMSE, R²)
       │
       ▼
  Save Best Model (house_price_model.pkl)
       │
       ▼
  Save Predictions (predictions.csv)
       │
       ▼
  Generate 8 Visualizations
```

---

## 📂 Dataset

| Property | Detail |
|----------|--------|
| **File** | `Housing.csv` |
| **Records** | 545+ Houses |
| **Raw Features** | 12 |
| **Engineered Features** | 2 |
| **Target Variable** | `price` (in INR) |
| **Source** | Inspired by [Rajvardhan180/House_Price_Prediction](https://github.com/Rajvardhan180/House_Price_Prediction) |

---

## 📁 Project Structure

```
house_price_prediction/
│
├── Housing.csv                      # Raw dataset (545+ records)
├── house_price_prediction.py        # Main Python script (full pipeline)
├── House_Price_Prediction.py        # Alternative Python script
├── house_price_prediction.ipynb     # Jupyter Notebook version
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
│
├── models/
│   └── house_price_model.pkl        # Trained best model (auto-generated)
│
├── results/
│   └── predictions.csv              # Actual vs Predicted (auto-generated)
│
└── plots/                           # Visualizations (auto-generated)
    ├── 01_price_distribution.png
    ├── 02_correlation_heatmap.png
    ├── 03_area_vs_price.png
    ├── 04_model_comparison.png
    ├── 05_actual_vs_predicted.png
    ├── 06_residuals.png
    ├── 07_feature_importance.png
    └── 08_price_by_furnishing.png
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Python Script
```bash
python house_price_prediction.py
```

### 3. Or Open the Jupyter Notebook
```bash
jupyter notebook house_price_prediction.ipynb
```

---

## 📈 Expected Results

| Model | R² Score | RMSE (approx) |
|-------|----------|----------------|
| Gradient Boosting | ~0.87 | ~₹8.5 Lakh |
| Random Forest | ~0.85 | ~₹8.7 Lakh |
| Decision Tree | ~0.78 | ~₹11 Lakh |
| Linear Regression | ~0.66 | ~₹13.8 Lakh |
| Ridge Regression | ~0.66 | ~₹13.8 Lakh |
| Lasso Regression | ~0.65 | ~₹14 Lakh |

> *Results may vary slightly based on dataset split.*

---

## 📊 Visualizations Generated (8 Plots)

| # | Plot | Description |
|---|------|-------------|
| 1 | Price Distribution | Histogram of all house prices |
| 2 | Correlation Heatmap | Feature-to-feature correlation matrix |
| 3 | Area vs Price | Scatter plot colored by bedrooms |
| 4 | Model Comparison | R² and RMSE across all 6 models |
| 5 | Actual vs Predicted | Scatter with perfect-fit reference line |
| 6 | Residual Plot | Residual distribution analysis |
| 7 | Feature Importance | Bar chart of feature importances/coefficients |
| 8 | Price by Furnishing | Average prices by furnishing status |

---

## 📝 References

- Original project inspiration: [Rajvardhan180/House_Price_Prediction](https://github.com/Rajvardhan180/House_Price_Prediction)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

---

## 📄 License

This project is submitted as a **Mini Project** for academic purposes under the MIT License.

---

<div align="center">
Made with ❤️ by <strong>Rajvir Singh</strong> | GU-2024-1545 | BCA Semester 5, Section C
</div>
