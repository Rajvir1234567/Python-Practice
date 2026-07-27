import pytest
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ml.inference import predict_student_risk
from src.recommendation.engine import generate_recommendations

def test_ml_inference_output_structure():
    sample_input = {
        'age': 17, 'studytime': 2, 'failures': 1, 'absences': 8,
        'G1': 8, 'G2': 9, 'Medu': 2, 'Fedu': 2, 'health': 3,
        'famrel': 4, 'freetime': 3, 'goout': 3, 'Dalc': 1, 'Walc': 1,
        'traveltime': 1, 'sex': 'F', 'internet': 'yes', 'higher': 'yes',
        'schoolsup': 'no', 'address': 'U', 'famsize': 'GT3', 'Pstatus': 'T', 'famsup': 'yes'
    }
    
    result = predict_student_risk(sample_input)
    
    assert 'risk_probability' in result
    assert 0.0 <= result['risk_probability'] <= 1.0
    assert result['risk_classification'] in ['Safe', 'Moderate Risk', 'High Risk']
    assert isinstance(result['xai_rules'], list)
    assert len(result['xai_rules']) > 0

def test_recommendation_engine_output():
    sample_input = {'studytime': 1, 'failures': 2, 'absences': 12, 'G1': 7, 'G2': 8}
    recs = generate_recommendations("High Risk", sample_input)
    
    assert isinstance(recs, list)
    assert len(recs) >= 2
    assert any("Attendance Warning" in r for r in recs)
