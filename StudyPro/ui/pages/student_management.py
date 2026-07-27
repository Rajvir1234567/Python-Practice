import streamlit as st
import pandas as pd
from src.ml.inference import predict_student_risk
from src.recommendation.engine import generate_recommendations
from src.database.queries import add_student, add_academic_record, add_prediction, add_recommendation

def render_student_management_page():
    st.title("🎓 Student Evaluation & Management")
    st.markdown("Assess student academic risk using local machine learning inference and Explainable AI (XAI).")

    tabs = st.tabs(["📝 Individual Student Assessment", "📁 Batch CSV Processing"])
    
    with tabs[0]:
        st.subheader("Single Student Risk Evaluation")
        with st.form("student_eval_form"):
            c1, c2 = st.columns(2)
            with c1:
                first_name = st.text_input("First Name", "Alex")
                last_name = st.text_input("Last Name", "Rivera")
                age = st.number_input("Age", min_value=12, max_value=25, value=17)
                gender = st.selectbox("Gender", ["F", "M"])
                studytime = st.slider("Weekly Study Hours (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)", 1, 4, 2)
                failures = st.number_input("Past Class Failures", 0, 4, 0)
                absences = st.number_input("School Absences (Days)", 0, 90, 4)
            with c2:
                g1 = st.slider("Term 1 Score (G1: 0-20)", 0, 20, 11)
                g2 = st.slider("Term 2 Score (G2: 0-20)", 0, 20, 10)
                medu = st.selectbox("Mother's Education (0: None to 4: Higher)", [0, 1, 2, 3, 4], index=2)
                fedu = st.selectbox("Father's Education (0: None to 4: Higher)", [0, 1, 2, 3, 4], index=2)
                internet = st.selectbox("Internet Access at Home", ["yes", "no"])
                higher = st.selectbox("Wants Higher Education", ["yes", "no"])
                schoolsup = st.selectbox("Extra Educational Support", ["no", "yes"])

            submitted = st.form_submit_button("🚀 Run AI Performance Prediction", use_container_width=True)

        if submitted:
            input_dict = {
                'age': age, 'studytime': studytime, 'failures': failures,
                'absences': absences, 'G1': g1, 'G2': g2, 'Medu': medu, 'Fedu': fedu,
                'health': 3, 'famrel': 4, 'freetime': 3, 'goout': 3, 'Dalc': 1, 'Walc': 1,
                'traveltime': 1, 'sex': gender, 'internet': internet, 'higher': higher,
                'schoolsup': schoolsup, 'address': 'U', 'famsize': 'GT3', 'Pstatus': 'T', 'famsup': 'yes'
            }
            
            with st.spinner("Executing local Random Forest model inference..."):
                res = predict_student_risk(input_dict)
                recs = generate_recommendations(res['risk_classification'], input_dict)
                
                # Save to database
                student_id = add_student(first_name, last_name, age, gender)
                add_academic_record(student_id, "Term 2", studytime, failures, absences, g1, g2)
                pred_id = add_prediction(student_id, res['risk_probability'], res['risk_classification'])
                for rec in recs:
                    add_recommendation(pred_id, rec)
                    
            st.success(f"Evaluation complete for {first_name} {last_name}! Saved to SQLite database.")
            
            # Display Prediction Results
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.subheader("🎯 Risk Output")
                prob = res['risk_probability']
                cls = res['risk_classification']
                
                if cls == "High Risk":
                    st.error(f"**Classification**: {cls} (Probability: {prob*100:.1f}%)")
                elif cls == "Moderate Risk":
                    st.warning(f"**Classification**: {cls} (Probability: {prob*100:.1f}%)")
                else:
                    st.success(f"**Classification**: {cls} (Probability: {prob*100:.1f}%)")
                    
                st.progress(prob)

            with col_res2:
                st.subheader("🧠 Explainable AI (XAI) Decision Path")
                st.info("Local decision tree path triggering this output:")
                for rule in res['xai_rules']:
                    st.write(f"• {rule}")

            st.markdown("---")
            st.subheader("📋 Targeted Learning Recommendations")
            for r in recs:
                st.markdown(r)

    with tabs[1]:
        st.subheader("Batch Student Processing via CSV")
        uploaded_file = st.file_uploader("Upload CSV containing student records", type=['csv'])
        
        if uploaded_file:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.write("Uploaded Payload Preview:", df_upload.head(3))
                
                required_cols = ['age', 'studytime', 'failures', 'absences', 'G1', 'G2', 'first_name', 'last_name', 'gender']
                missing = [c for c in required_cols if c not in df_upload.columns]
                
                if missing:
                    st.error(f"Invalid Schema! Missing columns: {missing}")
                else:
                    if st.button("⚡ Run Batch AI Predictions"):
                        with st.spinner("Processing batch and saving to database..."):
                            for _, row in df_upload.iterrows():
                                input_dict = row.to_dict()
                                pred = predict_student_risk(input_dict)
                                recs = generate_recommendations(pred['risk_classification'], input_dict)
                                
                                student_id = add_student(row['first_name'], row['last_name'], row['age'], row['gender'])
                                add_academic_record(student_id, "Batch Process", row['studytime'], row['failures'], row['absences'], row['G1'], row['G2'])
                                pred_id = add_prediction(student_id, pred['risk_probability'], pred['risk_classification'])
                                for rec in recs:
                                    add_recommendation(pred_id, rec)
                        st.success(f"Successfully processed {len(df_upload)} students and saved to database.")
            except Exception as e:
                st.error(f"Error parsing CSV file: {e}")
