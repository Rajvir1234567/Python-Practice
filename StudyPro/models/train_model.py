import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_and_save_models():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'student-por.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_dataset.py first.")
        
    df = pd.read_csv(data_path)
    
    # Engineer binary target: G3 < 10 -> At-Risk (1), else Safe (0)
    df['at_risk'] = (df['G3'] < 10).astype(int)
    
    # Feature columns selection
    feature_cols = [
        'age', 'studytime', 'failures', 'absences', 'G1', 'G2',
        'health', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',
        'traveltime', 'Medu', 'Fedu'
    ]
    
    # Select categorical columns to one-hot encode
    cat_cols = ['sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'higher', 'internet']
    
    X = pd.get_dummies(df[feature_cols + cat_cols], columns=cat_cols, drop_first=True)
    y = df['at_risk']
    
    feature_names = list(X.columns)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Check if SMOTE is available, otherwise use balanced class weights
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    except ImportError:
        X_train_res, y_train_res = X_train_scaled, y_train
    
    # 1. Primary Model: Lightweight Capped Random Forest (Memory optimized)
    rf_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train_res, y_train_res)
    
    # 2. XAI Parallel Model: Decision Tree for Rule Path Extraction
    dt_xai = DecisionTreeClassifier(
        max_depth=4,
        random_state=42,
        class_weight='balanced'
    )
    dt_xai.fit(X_train_res, y_train_res)
    
    # Evaluate
    y_pred_rf = rf_model.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Model Accuracy: {acc_rf * 100:.2f}%")
    print(classification_report(y_test, y_pred_rf))
    
    # Save serialized joblib artifacts
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest.joblib'))
    joblib.dump(dt_xai, os.path.join(models_dir, 'decision_tree_xai.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.joblib'))
    
    print("All model assets successfully dumped to /models/")

if __name__ == "__main__":
    train_and_save_models()
