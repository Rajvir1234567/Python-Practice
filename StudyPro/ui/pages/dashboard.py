import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.database.queries import get_dashboard_kpis, get_all_students_with_latest_prediction

def render_dashboard_page():
    st.title("📊 Executive Analytics Dashboard")
    st.markdown("Real-time academic indicators, risk distributions, and machine learning model metrics.")

    kpis = get_dashboard_kpis()

    # KPI Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Students", value=kpis['total_students'])
    with col2:
        st.metric(label="At-Risk Students", value=kpis['at_risk_count'], delta="Early Warning", delta_color="inverse")
    with col3:
        st.metric(label="Avg Absences", value=f"{kpis['avg_absences']} days")
    with col4:
        st.metric(label="Model Accuracy", value=f"{kpis['model_accuracy']}%")

    st.markdown("---")

    students_data = get_all_students_with_latest_prediction()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🎯 Risk Category Distribution")
        if students_data:
            df_students = pd.DataFrame(students_data)
            df_students['risk_classification'] = df_students['risk_classification'].fillna('Unanalyzed')

            fig_pie = px.pie(
                df_students,
                names='risk_classification',
                title="Student Academic Risk Share",
                hole=0.4,
                color='risk_classification',
                color_discrete_map={'Safe': '#10b981', 'Moderate Risk': '#f59e0b', 'High Risk': '#ef4444', 'Unanalyzed': '#94a3b8'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No student records found. Add students in Student Management to view analytics.")

    with col_right:
        st.subheader("📉 Absences vs. Academic Score")
        if students_data:
            df_students = pd.DataFrame(students_data)
            df_students['g2_score'] = df_students['g2_score'].fillna(10)
            df_students['absences'] = df_students['absences'].fillna(0)
            df_students['risk_classification'] = df_students['risk_classification'].fillna('Safe')

            fig_scatter = px.scatter(
                df_students,
                x='absences',
                y='g2_score',
                color='risk_classification',
                hover_data=['first_name', 'last_name'],
                labels={'absences': 'Absences (Days)', 'g2_score': 'G2 Term Grade (0-20)'},
                title="Absences vs. Term Score Correlation",
                color_discrete_map={'Safe': '#10b981', 'Moderate Risk': '#f59e0b', 'High Risk': '#ef4444'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No plot data available.")

    # ── Dynamic Academic Insights Section ─────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Academic Risk Insights — Your Students")
    st.caption("Live analysis of your current student database. Shows how Safe vs At-Risk students differ across key academic factors.")

    if not students_data:
        st.info("Add students first to see live academic insights.")
        return

    df = pd.DataFrame(students_data)

    # Separate safe vs at-risk
    df_safe    = df[df['risk_classification'] == 'Safe']
    df_atrisk  = df[df['risk_classification'].isin(['Moderate Risk', 'High Risk'])]

    if df_safe.empty or df_atrisk.empty:
        st.info("Need both Safe and At-Risk students in the database to show comparison insights.")
        return

    # ── Comparison Bar Chart: Safe vs At-Risk averages ──────────────────────
    factors = {
        'G2 Term Score (out of 20)':   ('g2_score',  True,  "Higher is better. At-risk students typically score below 10."),
        'G1 Term Score (out of 20)':   ('g1_score',  True,  "First-term score — a strong predictor of final outcomes."),
        'Weekly Study Time (1–4 scale)': ('studytime', True,  "1 = less than 2hrs/week. At-risk students average close to 1."),
        'School Absences (days)':       ('absences',  False, "More absences = higher risk. Safe students average fewer days missed."),
        'Past Failures (count)':        ('failures',  False, "Number of previously failed courses. Even 1 failure increases risk."),
    }

    safe_avgs   = []
    atrisk_avgs = []
    labels      = []
    descriptions = []
    higher_is_better = []

    for label, (col, hib, desc) in factors.items():
        if col in df.columns:
            labels.append(label)
            safe_avgs.append(round(df_safe[col].mean(), 2))
            atrisk_avgs.append(round(df_atrisk[col].mean(), 2))
            descriptions.append(desc)
            higher_is_better.append(hib)

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        name='✅ Safe Students',
        x=labels,
        y=safe_avgs,
        marker_color='#10b981',
        text=[f"{v}" for v in safe_avgs],
        textposition='outside'
    ))
    fig_compare.add_trace(go.Bar(
        name='🔴 At-Risk Students',
        x=labels,
        y=atrisk_avgs,
        marker_color='#ef4444',
        text=[f"{v}" for v in atrisk_avgs],
        textposition='outside'
    ))
    fig_compare.update_layout(
        barmode='group',
        title="Safe Students vs At-Risk Students — Average Academic Metrics",
        xaxis_title="",
        yaxis_title="Average Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=10)
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # ── Traffic Light Cards ──────────────────────────────────────────────────
    st.markdown("#### 🚦 Factor-by-Factor Analysis")
    st.caption("Each card shows the gap between Safe and At-Risk students for that factor.")

    cards = st.columns(len(labels))
    for i, (col_ui, label, sv, rv, hib, desc) in enumerate(
        zip(cards, labels, safe_avgs, atrisk_avgs, higher_is_better, descriptions)
    ):
        gap = round(sv - rv, 2)
        # Determine if gap is good or bad
        is_good_gap = (gap > 0 and hib) or (gap < 0 and not hib)
        icon  = "🟢" if is_good_gap else "🔴"
        color = "#10b981" if is_good_gap else "#ef4444"
        gap_label = f"+{gap}" if gap > 0 else str(gap)

        col_ui.markdown(
            f"""
            <div style="border:1px solid #e2e8f0; border-radius:10px; padding:14px; text-align:center; background:#fff;">
                <div style="font-size:22px;">{icon}</div>
                <div style="font-weight:700; font-size:13px; color:#0f172a; margin:6px 0;">{label.split('(')[0].strip()}</div>
                <div style="color:#10b981; font-size:13px;">✅ Safe: <b>{sv}</b></div>
                <div style="color:#ef4444; font-size:13px;">🔴 At-Risk: <b>{rv}</b></div>
                <div style="color:{color}; font-weight:700; font-size:15px; margin-top:6px;">Gap: {gap_label}</div>
                <div style="color:#64748b; font-size:11px; margin-top:6px;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
