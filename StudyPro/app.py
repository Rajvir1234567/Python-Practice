import streamlit as st
import os
from src.database.connection import init_db
from ui.pages.dashboard import render_dashboard_page
from ui.pages.student_management import render_student_management_page
from ui.pages.history import render_history_page

# Page Config
st.set_page_config(
    page_title="StudyPro | Academic Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI cards and executive styling
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .css-1d372ab, .stSidebar {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database Schema
@st.cache_resource
def setup_database():
    init_db()

setup_database()

# Session State Setup
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"

# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🎓 **StudyPro**")
    st.caption("AI-Powered Academic Decision Support")
    st.markdown("---")
    
    page_choice = st.radio(
        "Navigation Menu",
        options=["📊 Dashboard", "🎓 Student Evaluation", "📜 Prediction History"],
        index=0 if st.session_state.page == "Dashboard" else (1 if st.session_state.page == "Student Evaluation" else 2)
    )
    
    st.markdown("---")
    st.markdown("🔒 **System Status**: Online")
    st.markdown("⚡ **Model**: Local Random Forest + XAI")
    st.markdown("💾 **Storage**: SQLite 3NF")
    st.markdown("---")
    st.caption("StudyPro Capstone Project © 2026")

# Navigation Routing
if "Dashboard" in page_choice:
    st.session_state.page = "Dashboard"
    render_dashboard_page()
elif "Student Evaluation" in page_choice:
    st.session_state.page = "Student Evaluation"
    render_student_management_page()
elif "Prediction History" in page_choice:
    st.session_state.page = "Prediction History"
    render_history_page()
