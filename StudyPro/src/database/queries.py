from .connection import get_connection

def add_student(first_name, last_name, age, gender, db_file=None):
    """Inserts a new student and returns student_id."""
    conn = get_connection(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO students (first_name, last_name, age, gender)
        VALUES (?, ?, ?, ?);
    """, (first_name, last_name, age, gender))
    student_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return student_id

def add_academic_record(student_id, term, studytime, failures, absences, g1_score, g2_score, db_file=None):
    """Inserts an academic record for a student."""
    conn = get_connection(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO academic_records (student_id, term, studytime, failures, absences, g1_score, g2_score)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """, (student_id, term, studytime, failures, absences, g1_score, g2_score))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def add_prediction(student_id, risk_probability, risk_classification, db_file=None):
    """Inserts a prediction result and returns prediction_id."""
    conn = get_connection(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (student_id, risk_probability, risk_classification)
        VALUES (?, ?, ?);
    """, (student_id, risk_probability, risk_classification))
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id

def add_recommendation(prediction_id, intervention_text, status='active', db_file=None):
    """Inserts a recommendation tied to a prediction."""
    conn = get_connection(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO recommendations (prediction_id, intervention_text, status)
        VALUES (?, ?, ?);
    """, (prediction_id, intervention_text, status))
    rec_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return rec_id

def get_all_students_with_latest_prediction(db_file=None):
    """
    Fetches one row per student with their latest academic record,
    latest prediction, and primary intervention recommendation.
    Uses correlated subqueries to prevent row multiplication from JOIN on recommendations.
    """
    conn = get_connection(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            s.student_id,
            s.first_name,
            s.last_name,
            s.age,
            s.gender,

            -- Latest academic record for this student
            (SELECT ar.studytime FROM academic_records ar
             WHERE ar.student_id = s.student_id
             ORDER BY ar.record_id DESC LIMIT 1) AS studytime,

            (SELECT ar.failures FROM academic_records ar
             WHERE ar.student_id = s.student_id
             ORDER BY ar.record_id DESC LIMIT 1) AS failures,

            (SELECT ar.absences FROM academic_records ar
             WHERE ar.student_id = s.student_id
             ORDER BY ar.record_id DESC LIMIT 1) AS absences,

            (SELECT ar.g1_score FROM academic_records ar
             WHERE ar.student_id = s.student_id
             ORDER BY ar.record_id DESC LIMIT 1) AS g1_score,

            (SELECT ar.g2_score FROM academic_records ar
             WHERE ar.student_id = s.student_id
             ORDER BY ar.record_id DESC LIMIT 1) AS g2_score,

            -- Latest prediction for this student
            (SELECT p.risk_probability FROM predictions p
             WHERE p.student_id = s.student_id
             ORDER BY p.prediction_id DESC LIMIT 1) AS risk_probability,

            (SELECT p.risk_classification FROM predictions p
             WHERE p.student_id = s.student_id
             ORDER BY p.prediction_id DESC LIMIT 1) AS risk_classification,

            (SELECT p.prediction_date FROM predictions p
             WHERE p.student_id = s.student_id
             ORDER BY p.prediction_id DESC LIMIT 1) AS prediction_date,

            -- First (primary) recommendation for latest prediction only
            (SELECT r.intervention_text FROM recommendations r
             WHERE r.prediction_id = (
                 SELECT p2.prediction_id FROM predictions p2
                 WHERE p2.student_id = s.student_id
                 ORDER BY p2.prediction_id DESC LIMIT 1
             )
             ORDER BY r.recommendation_id ASC LIMIT 1) AS intervention_text

        FROM students s
        ORDER BY s.student_id DESC;
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_dashboard_kpis(db_file=None):
    """Calculates aggregate metrics for the dashboard KPI cards."""
    conn = get_connection(db_file)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as total_students FROM students;")
    total_students = cursor.fetchone()['total_students']
    
    cursor.execute("""
        SELECT COUNT(DISTINCT student_id) as at_risk_count
        FROM predictions
        WHERE risk_classification IN ('High Risk', 'Moderate Risk');
    """)
    at_risk_count = cursor.fetchone()['at_risk_count']
    
    cursor.execute("SELECT AVG(absences) as avg_absences FROM academic_records;")
    avg_absences_res = cursor.fetchone()['avg_absences']
    avg_absences = round(avg_absences_res, 1) if avg_absences_res else 0.0
    
    conn.close()
    return {
        'total_students': total_students,
        'at_risk_count': at_risk_count,
        'avg_absences': avg_absences,
        'model_accuracy': 89.4  # Evaluated ensemble accuracy %
    }

def delete_student_by_id(student_id, db_file=None):
    """
    Deletes a single student and all their related data.
    CASCADE FK constraints handle academic_records, predictions, recommendations.
    """
    conn = get_connection(db_file)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM students WHERE student_id = ?;", (student_id,))
    conn.commit()
    conn.close()

def clear_all_history(db_file=None):
    """
    Wipes all student data from every table.
    Resets auto-increment counters.
    The ML models and schema are preserved — only data is deleted.
    """
    conn = get_connection(db_file)
    cursor = conn.cursor()
    # Delete in child-first order to respect FK constraints
    cursor.execute("DELETE FROM recommendations;")
    cursor.execute("DELETE FROM predictions;")
    cursor.execute("DELETE FROM academic_records;")
    cursor.execute("DELETE FROM students;")
    # Reset auto-increment counters
    cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('students','academic_records','predictions','recommendations');")
    conn.commit()
    conn.close()
