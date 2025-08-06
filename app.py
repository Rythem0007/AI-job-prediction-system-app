import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ✅ Updated custom module imports
from models.traditional_ml import TraditionalMLModel
from models.deep_learning import DeepLearningModel
from processors.pdf_processor import PDFProcessor
from processors.text_processor import TextProcessor
from models.feature_extractor import FeatureExtractor
from components.model_comparison import ModelComparison
from components.results_display import ResultsDisplay

# Page configuration
st.set_page_config(
    page_title="AI Job Prediction System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Cached Model Loader ----
@st.cache_resource
def load_models():
    """Initialize and load the ML models (cached for session)"""
    traditional_model = TraditionalMLModel()
    deep_learning_model = DeepLearningModel()
    traditional_model.initialize()
    deep_learning_model.initialize()
    return traditional_model, deep_learning_model

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'traditional_model' not in st.session_state:
    st.session_state.traditional_model = None
if 'deep_learning_model' not in st.session_state:
    st.session_state.deep_learning_model = None
if 'results' not in st.session_state:
    st.session_state.results = None

def main():
    # Header
    st.title("🎯 AI-Powered Job Prediction System")
    st.markdown("""
    ### Hybrid ML Approach: Traditional + Deep Learning
    Upload your resume and job description to get AI-powered matching and insights.
    """)

    # Sidebar
    with st.sidebar:
        st.header("🔧 System Configuration")

        st.subheader("Model Status")
        if not st.session_state.models_loaded:
            st.warning("⚠️ Models not loaded")
            if st.button("🚀 Load AI Models", type="primary"):
                with st.spinner("Loading AI Models... This may take a few minutes on first run."):
                    try:
                        st.session_state.traditional_model, st.session_state.deep_learning_model = load_models()
                        st.session_state.models_loaded = True
                        st.success("✅ Models loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error loading models: {str(e)}")
                        st.error("Please check your internet connection and try again.")
        else:
            st.success("✅ Models Ready")
            if st.button("🔄 Reload Models"):
                st.session_state.models_loaded = False
                st.session_state.traditional_model = None
                st.session_state.deep_learning_model = None
                st.rerun()

        st.divider()

        st.subheader("Analysis Options")
        use_traditional = st.checkbox("Traditional ML Analysis", value=True)
        use_deep_learning = st.checkbox("Deep Learning Analysis", value=True)

        if not (use_traditional or use_deep_learning):
            st.error("Please select at least one analysis method")

    if not st.session_state.models_loaded:
        st.info("👆 Please load the AI models from the sidebar to begin analysis.")
        return

    # Input section
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Resume Upload")
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF or DOC)",
            type=['pdf', 'doc', 'docx'],
            help="Supported formats: PDF, DOC, DOCX"
        )

        resume_text = ""
        if uploaded_file is not None:
            try:
                pdf_processor = PDFProcessor()
                resume_text = pdf_processor.extract_text(uploaded_file)

                with st.expander("📖 Extracted Resume Text (Preview)"):
                    st.text_area(
                        "Resume content:",
                        value=resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
                        height=150,
                        disabled=True
                    )
                st.success(f"✅ Resume processed ({len(resume_text)} characters)")

            except Exception as e:
                st.error(f"❌ Error processing resume: {str(e)}")
                resume_text = ""

    with col2:
        st.subheader("💼 Job Description")
        job_description = st.text_area(
            "Paste the job description here:",
            height=200,
            placeholder="Enter the complete job description including requirements, responsibilities, and qualifications..."
        )

        if job_description:
            word_count = len(job_description.split())
            st.info(f"📊 Job description: {word_count} words")

    st.divider()

    if st.button("🔍 Analyze Job Match", type="primary", use_container_width=True):
        if not resume_text:
            st.error("❌ Please upload a valid resume file")
            return

        if not job_description.strip():
            st.error("❌ Please provide a job description")
            return

        with st.spinner("🤖 AI Analysis in progress..."):
            results = {}

            try:
                if use_traditional and st.session_state.traditional_model:
                    traditional_result = st.session_state.traditional_model.predict(
                        resume_text, job_description
                    )
                    results['traditional'] = traditional_result

                if use_deep_learning and st.session_state.deep_learning_model:
                    deep_learning_result = st.session_state.deep_learning_model.predict(
                        resume_text, job_description
                    )
                    results['deep_learning'] = deep_learning_result

                st.session_state.results = results

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                return

    if st.session_state.results:
        st.divider()
        st.header("📊 Analysis Results")

        results_display = ResultsDisplay()
        model_comparison = ModelComparison()

        if 'traditional' in st.session_state.results:
            results_display.display_traditional_results(st.session_state.results['traditional'])

        if 'deep_learning' in st.session_state.results:
            results_display.display_deep_learning_results(st.session_state.results['deep_learning'])

        if len(st.session_state.results) > 1:
            st.divider()
            model_comparison.display_comparison(st.session_state.results)

        st.divider()
        results_df = pd.DataFrame([
            {
                'Model Type': model_type.replace('_', ' ').title(),
                'Match Score': f"{result['match_score']:.1%}",
                'Confidence': f"{result['confidence']:.1%}",
                'Key Skills Matched': len(result.get('matched_skills', [])),
                'Missing Skills': len(result.get('missing_skills', []))
            }
            for model_type, result in st.session_state.results.items()
        ])

        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="job_match_analysis.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
