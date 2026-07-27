import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import export_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

@st.cache_resource
def load_ml_assets():
    """Loads pre-trained local ML model assets into memory."""
    rf_path = os.path.join(MODELS_DIR, 'random_forest.joblib')
    dt_path = os.path.join(MODELS_DIR, 'decision_tree_xai.joblib')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    features_path = os.path.join(MODELS_DIR, 'feature_names.joblib')
    
    if not (os.path.exists(rf_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError("Model assets not found. Run models/train_model.py first.")
        
    rf_model = joblib.load(rf_path)
    dt_xai = joblib.load(dt_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    
    return rf_model, dt_xai, scaler, feature_names

def prepare_input_dataframe(input_dict, feature_names):
    """Formats raw student dictionary into the exact pandas DataFrame structure required by the model."""
    df = pd.DataFrame([input_dict])
    
    cat_cols = ['sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'higher', 'internet']
    df_encoded = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True)
    
    # Reindex columns to match training feature_names exactly
    df_full = df_encoded.reindex(columns=feature_names, fill_value=0)
    return df_full

def predict_student_risk(input_dict):
    """Executes local model inference and returns risk probability, classification, and XAI rules."""
    rf_model, dt_xai, scaler, feature_names = load_ml_assets()
    
    input_df = prepare_input_dataframe(input_dict, feature_names)
    input_scaled = scaler.transform(input_df)
    
    # Predict Probability
    prob_risk = float(rf_model.predict_proba(input_scaled)[0][1])
    
    if prob_risk >= 0.65:
        risk_class = "High Risk"
    elif prob_risk >= 0.35:
        risk_class = "Moderate Risk"
    else:
        risk_class = "Safe"
        
    # Extract Local XAI Decision Rules
    xai_rules = extract_decision_path_rules(dt_xai, input_scaled[0], feature_names, input_df.iloc[0])
    
    return {
        'risk_probability': round(prob_risk, 3),
        'risk_classification': risk_class,
        'xai_rules': xai_rules,
        'feature_importances': dict(zip(feature_names, rf_model.feature_importances_))
    }

def extract_decision_path_rules(tree_model, scaled_sample, feature_names, raw_sample_row):
    """Traverses DecisionTree path to extract natural language XAI rules for a specific student."""
    node_indicator = tree_model.decision_path([scaled_sample])
    leaf_id = tree_model.apply([scaled_sample])[0]
    
    feature = tree_model.tree_.feature
    threshold = tree_model.tree_.threshold
    
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    rules = []
    for node_id in node_index:
        if leaf_id == node_id:
            continue # Reached leaf
            
        f_idx = feature[node_id]
        f_name = feature_names[f_idx]
        thresh_val = round(threshold[node_id], 2)
        actual_val = raw_sample_row.get(f_name, scaled_sample[f_idx])
        
        if scaled_sample[f_idx] <= threshold[node_id]:
            comparison = "<="
        else:
            comparison = ">"
            
        # Format human readable rule
        rule_desc = f"Key Factor: '{f_name}' (Current: {actual_val}) was {comparison} threshold baseline"
        rules.append(rule_desc)
        
    return rules if rules else ["Student metrics fall within standard baseline boundaries."]
