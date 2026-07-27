import streamlit as st
import pandas as pd
from src.database.queries import (
    get_all_students_with_latest_prediction,
    delete_student_by_id,
    clear_all_history
)

def render_history_page():
    st.title("📜 Prediction History & Audit Trail")
    st.markdown("Longitudinal records and past risk evaluations stored securely in the local SQLite database.")

    # ── Danger Zone: Clear All ──────────────────────────────────────────────
    with st.expander("🗑️ Danger Zone — Delete Options", expanded=False):
        st.warning("⚠️ These actions are **permanent** and cannot be undone.")

        col_d1, col_d2 = st.columns([1, 2])
        with col_d1:
            if st.button("🧹 Clear ALL History", type="primary", use_container_width=True):
                st.session_state['confirm_clear'] = True

        if st.session_state.get('confirm_clear'):
            with col_d2:
                st.error("Are you sure? This will delete ALL students, predictions, and recommendations.")
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("✅ Yes, Delete Everything", use_container_width=True):
                        clear_all_history()
                        st.session_state['confirm_clear'] = False
                        st.success("✅ All history cleared successfully!")
                        st.rerun()
                with cc2:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state['confirm_clear'] = False
                        st.rerun()

    st.markdown("---")

    students = get_all_students_with_latest_prediction()

    if not students:
        st.info("No prediction history found in the database. Run student evaluations first.")
        return

    df = pd.DataFrame(students)

    # ── Filters ─────────────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        search_query = st.text_input("🔍 Search Student Name", "")
    with col_f2:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=["Safe", "Moderate Risk", "High Risk"],
            default=["Safe", "Moderate Risk", "High Risk"]
        )

    if search_query:
        df = df[
            df['first_name'].str.contains(search_query, case=False, na=False) |
            df['last_name'].str.contains(search_query, case=False, na=False)
        ]

    if risk_filter:
        df = df[df['risk_classification'].isin(risk_filter)]

    st.subheader(f"Records Found: {len(df)}")

    # ── Records Table with per-row Delete ───────────────────────────────────
    display_cols = [
        'student_id', 'first_name', 'last_name', 'age', 'gender',
        'studytime', 'failures', 'absences', 'g1_score', 'g2_score',
        'risk_probability', 'risk_classification', 'prediction_date', 'intervention_text'
    ]
    existing_cols = [c for c in display_cols if c in df.columns]

    # Header row
    header_cols = st.columns([1, 1.5, 1.5, 0.8, 0.8, 1, 1, 1, 0.8, 0.8, 1, 1.5, 1.5, 2.5, 0.8])
    headers = ['ID', 'First Name', 'Last Name', 'Age', 'Gender',
               'Study', 'Fails', 'Absent', 'G1', 'G2',
               'Risk %', 'Classification', 'Timestamp', 'Recommendation', 'Del']
    for col, h in zip(header_cols, headers):
        col.markdown(f"**{h}**")

    st.markdown("---")

    for _, row in df.iterrows():
        risk_cls = row.get('risk_classification', 'Unknown')
        color = "#ef4444" if risk_cls == "High Risk" else "#f59e0b" if risk_cls == "Moderate Risk" else "#10b981"

        rcols = st.columns([1, 1.5, 1.5, 0.8, 0.8, 1, 1, 1, 0.8, 0.8, 1, 1.5, 1.5, 2.5, 0.8])
        rcols[0].write(row.get('student_id', ''))
        rcols[1].write(row.get('first_name', ''))
        rcols[2].write(row.get('last_name', ''))
        rcols[3].write(row.get('age', ''))
        rcols[4].write(row.get('gender', ''))
        rcols[5].write(row.get('studytime', ''))
        rcols[6].write(row.get('failures', ''))
        rcols[7].write(row.get('absences', ''))
        rcols[8].write(row.get('g1_score', ''))
        rcols[9].write(row.get('g2_score', ''))
        prob = row.get('risk_probability', 0)
        rcols[10].write(f"{prob*100:.1f}%" if prob else "N/A")
        rcols[11].markdown(f"<span style='color:{color};font-weight:bold'>{risk_cls}</span>", unsafe_allow_html=True)
        date_val = str(row.get('prediction_date', ''))[:16]
        rcols[12].write(date_val)
        rec_text = str(row.get('intervention_text', ''))[:60] + "..." if row.get('intervention_text') else "—"
        rcols[13].write(rec_text)

        sid = int(row.get('student_id', 0))
        if rcols[14].button("🗑️", key=f"del_{sid}", help=f"Delete student {sid}"):
            delete_student_by_id(sid)
            st.success(f"Student #{sid} deleted.")
            st.rerun()
