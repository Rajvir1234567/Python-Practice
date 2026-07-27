import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database.connection import init_db
from src.database.queries import add_student, add_academic_record, add_prediction, add_recommendation, get_all_students_with_latest_prediction

def test_database_crud_operations(tmp_path):
    db_file = str(tmp_path / "test_studypro.db")
    init_db(db_file)
    
    # 1. Add student
    student_id = add_student("Jane", "Doe", 17, "F", db_file=db_file)
    assert student_id is not None
    assert student_id > 0
    
    # 2. Add academic record
    rec_id = add_academic_record(student_id, "Term 1", 3, 0, 2, 14.0, 15.0, db_file=db_file)
    assert rec_id is not None
    
    # 3. Add prediction
    pred_id = add_prediction(student_id, 0.15, "Safe", db_file=db_file)
    assert pred_id is not None
    
    # 4. Add recommendation
    rec_item_id = add_recommendation(pred_id, "Maintain current habits", db_file=db_file)
    assert rec_item_id is not None
    
    # 5. Retrieve all students
    records = get_all_students_with_latest_prediction(db_file=db_file)
    assert len(records) == 1
    assert records[0]['first_name'] == "Jane"
    assert records[0]['risk_classification'] == "Safe"
