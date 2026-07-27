# 🎓 StudyPro — AI-Powered Academic Performance Prediction & Personalized Learning Recommendation System

> **BCA Capstone Project** | Built with Python · Streamlit · Scikit-learn · SQLite · Plotly

**StudyPro** is a fully functional, AI-powered academic decision-support and early-warning web platform designed for educators, academic coordinators, and institutions. It transforms traditional reactive student evaluation into a **proactive, data-driven early warning system** — predicting student risk, explaining the reasons behind every prediction, and recommending targeted interventions.

---

## 🌟 Key Features

| Feature | Description |
| :--- | :--- |
| 🤖 **Local ML Classification** | Random Forest Classifier trained locally on UCI Student Performance dataset — **89.87% accuracy, 90% At-Risk Recall** |
| 🧠 **Explainable AI (XAI)** | Parallel Decision Tree extracts the exact decision path for each student and converts it into plain-English rules |
| 📋 **Recommendation Engine** | Deterministic heuristic engine generates prioritized, actionable learning interventions based on risk level and specific student metrics |
| 📊 **Executive Dashboard** | Live KPI cards, Risk Category Donut Chart, Absences vs. Score Scatter Plot, and dynamic Safe vs. At-Risk comparison insights |
| 🗄️ **3NF Relational Database** | Normalized SQLite schema (`users`, `students`, `academic_records`, `predictions`, `recommendations`) with FK constraints and parameterized queries |
| 📝 **Individual Evaluation** | Single student form with 13 input fields, instant ML prediction, XAI decision path, and recommendations |
| 📁 **Batch CSV Upload** | Upload multiple students via CSV — all predictions, records, and recommendations are saved to the database automatically |
| 🗑️ **History Management** | Full prediction audit trail with per-row delete, search/filter, and a danger-zone Clear All History option |
| 💡 **Dynamic Risk Insights** | Dashboard compares Safe vs. At-Risk students across 5 key academic factors with traffic-light cards and gap analysis |
| 🆓 **100% Free & Open Source** | Every tool used is free — runs locally, deploys to Streamlit Community Cloud at zero cost |

---

## 📁 Project Structure

```
StudyPro/
├── .gitignore
├── .streamlit/
│   └── config.toml                   # Streamlit theme & server configuration
├── README.md
├── app.py                            # Main Streamlit entry point — routing & layout
├── requirements.txt                  # All Python package dependencies
│
├── data/
│   ├── generate_dataset.py           # Synthetic UCI-style dataset generator (395 records)
│   ├── sample_upload.csv             # Ready-to-use sample CSV for batch upload testing
│   ├── student-por.csv               # Training dataset (33 columns, 395 rows)
│   └── studypro.db                   # SQLite database (auto-created on first run)
│
├── models/
│   ├── train_model.py                # Full ML training pipeline script
│   ├── random_forest.joblib          # Trained primary RF model (~214 KB)
│   ├── decision_tree_xai.joblib      # XAI Decision Tree model (~3 KB)
│   ├── scaler.joblib                 # StandardScaler artifact (~1.5 KB)
│   └── feature_names.joblib          # Feature name list for DataFrame alignment
│
├── src/
│   ├── database/
│   │   ├── connection.py             # SQLite connection + 3NF schema initialization
│   │   └── queries.py                # All CRUD + delete + KPI SQL functions
│   ├── ml/
│   │   └── inference.py              # Model loading, prediction, XAI path extraction
│   └── recommendation/
│       └── engine.py                 # Deterministic rule-based recommendation logic
│
├── ui/
│   └── pages/
│       ├── dashboard.py              # Executive Dashboard — KPIs + Charts + Insights
│       ├── student_management.py     # Student evaluation form + Batch CSV processor
│       └── history.py                # Prediction history, search, filter, delete
│
└── tests/
    ├── test_inference.py             # Pytest: ML inference + recommendation engine
    └── test_database.py              # Pytest: full DB CRUD chain with isolated test DB
```

---

## 🛠️ Quick Start Guide

### Prerequisites
- Python **3.10+** installed
- VS Code (recommended) or any terminal

### Step 1 — Clone / Open the Project
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\StudyPro
```

### Step 2 — Install All Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3 — Initialize the Database
```powershell
python src/database/connection.py
```
> ✅ Output: `Database initialized successfully.`

### Step 4 — Train the ML Models *(skip if `.joblib` files already exist)*
```powershell
python models/train_model.py
```
> ✅ Output: `Random Forest Model Accuracy: 89.87%`

### Step 5 — Launch the App
```powershell
python -m streamlit run app.py
```
> ✅ App opens at **http://localhost:8501**

---

## 📋 How to Use

### Adding Students Individually
1. Go to **🎓 Student Evaluation** in the sidebar
2. Fill in the student's details in the form
3. Click **🚀 Run AI Performance Prediction**
4. View: Risk Classification · XAI Decision Path · Personalized Recommendations
5. All data is automatically saved to the SQLite database

### Adding Students via CSV (Batch Upload)
1. Go to **🎓 Student Evaluation → 📁 Batch CSV Processing**
2. Upload a CSV file with the following columns:

| Required Columns | Optional Columns |
| :--- | :--- |
| `first_name`, `last_name`, `gender` | `sex`, `internet`, `higher`, `Medu`, `Fedu` |
| `age`, `studytime`, `failures`, `absences`, `G1`, `G2` | `health`, `famrel`, `goout`, `Dalc`, `Walc`, etc. |

> A ready-to-use sample file is available at `data/sample_upload.csv`

3. Click **⚡ Run Batch AI Predictions** — all students are saved to the database
4. Go to **📊 Dashboard** to see updated charts

### Deleting History
- **Single student**: Go to **📜 Prediction History** → click **🗑️** on any row
- **All history**: Go to **📜 Prediction History** → open **Danger Zone** → **🧹 Clear ALL History**

---

## 📊 Dashboard Sections

| Section | What It Shows |
| :--- | :--- |
| **KPI Cards** | Total Students · At-Risk Count · Avg Absences · Model Accuracy |
| **Risk Donut Chart** | Share of Safe / Moderate Risk / High Risk students |
| **Absences vs. Score Scatter** | Correlation between days absent and G2 term score, colored by risk |
| **Academic Risk Insights** | Grouped bar chart comparing Safe vs. At-Risk student averages across 5 factors |
| **Traffic Light Cards** | Per-factor gap analysis between Safe and At-Risk students with plain-English explanations |

---

## 🧪 Running Automated Tests

```powershell
python -m pytest tests/ -v
```

**Test Coverage:**
- `test_inference.py` — ML output structure, probability bounds, XAI rule generation, recommendation engine
- `test_database.py` — Full CRUD chain (add student → record → prediction → recommendation → retrieve)

---

## 🤖 Machine Learning Details

| Property | Value |
| :--- | :--- |
| **Algorithm** | Random Forest Classifier |
| **Trees** | 50 estimators |
| **Max Depth** | 6 (memory-optimized) |
| **Class Balancing** | `class_weight='balanced'` + SMOTE (if available) |
| **Train/Test Split** | 80% / 20% (stratified) |
| **Overall Accuracy** | **89.87%** |
| **At-Risk Recall** | **90%** (critical metric) |
| **XAI Model** | Decision Tree (max_depth=4) |
| **Inference Time** | < 3ms per student |
| **Model Size** | ~219 KB total (all artifacts) |

### Risk Classification Thresholds
| Probability | Classification |
| :--- | :--- |
| ≥ 65% | 🔴 High Risk |
| 35% – 64% | 🟡 Moderate Risk |
| < 35% | 🟢 Safe |

---

## 💾 Database Schema (3NF)

```
users           → user_id, username, password_hash, role, created_at
students        → student_id, first_name, last_name, age, gender, created_at
academic_records→ record_id, student_id (FK), term, studytime, failures, absences, g1_score, g2_score
predictions     → prediction_id, student_id (FK), risk_probability, risk_classification, prediction_date
recommendations → recommendation_id, prediction_id (FK), intervention_text, status, created_at
```

All foreign keys use `ON DELETE CASCADE` — deleting a student removes all their related data automatically.

**Default Login Credentials:**
| Username | Password | Role |
| :--- | :--- | :--- |
| `admin` | `admin` | Admin |
| `teacher` | `admin` | Teacher |

---

## 💰 Cost

**Total project cost: ₹0 / $0**

Every tool — Python, Streamlit, Scikit-learn, SQLite, Plotly, GitHub, Streamlit Community Cloud — is completely free and open-source. No GPU, no server, no subscription required.

---

## 🚀 Deployment (Free)

Deploy to **Streamlit Community Cloud** for free:
1. Push project to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo and select `app.py`
4. Click **Deploy** — live public URL in under 2 minutes

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**StudyPro** — BCA Capstone Project © 2026  
*AI-Powered Academic Decision Support Platform*
