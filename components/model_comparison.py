import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class ModelComparison:
    def __init__(self):
        pass
    
    def display_comparison(self, results):
        """Display comprehensive comparison between different models"""
        st.header("🔍 Model Comparison Analysis")
        
        # Create comparison metrics
        comparison_df = self._create_comparison_dataframe(results)
        
        # Display comparison overview
        self._display_comparison_overview(comparison_df)
        
        # Create visualizations
        self._create_comparison_charts(results, comparison_df)
        
        # Detailed breakdown
        self._display_detailed_breakdown(results)
        
        # Model performance analysis
        self._display_performance_analysis(results)
    
    def _create_comparison_dataframe(self, results):
        """Create DataFrame for model comparison"""
        comparison_data = []
        
        for model_type, result in results.items():
            comparison_data.append({
                'Model': model_type.replace('_', ' ').title(),
                'Match Score': result.get('match_score', 0),
                'Confidence': result.get('confidence', 0),
                'Matched Skills': len(result.get('matched_skills', [])),
                'Missing Skills': len(result.get('missing_skills', [])),
                'Total Resume Skills': len(result.get('resume_skills', [])),
                'Total Job Skills': len(result.get('job_skills', [])),
                'Skill Coverage': result.get('skill_coverage', 0) if 'skill_coverage' in result else (
                    len(result.get('matched_skills', [])) / max(len(result.get('job_skills', [])), 1)
                )
            })
        
        return pd.DataFrame(comparison_data)
    
    def _display_comparison_overview(self, comparison_df):
        """Display high-level comparison metrics"""
        st.subheader("📊 Model Performance Overview")
        
        # Create metrics columns
        cols = st.columns(len(comparison_df))
        
        for idx, (_, row) in enumerate(comparison_df.iterrows()):
            with cols[idx]:
                st.metric(
                    label=f"{row['Model']} Score",
                    value=f"{row['Match Score']:.1%}",
                    delta=f"±{row['Confidence']:.1%}",
                    help=f"Match score with {row['Confidence']:.1%} confidence"
                )
                
                st.write(f"**Skills:** {row['Matched Skills']}/{row['Total Job Skills']}")
                st.write(f"**Coverage:** {row['Skill Coverage']:.1%}")
    
    def _create_comparison_charts(self, results, comparison_df):
        """Create comparison visualizations"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Match Score Comparison
            fig_scores = go.Figure(data=[
                go.Bar(
                    name='Match Score',
                    x=comparison_df['Model'],
                    y=comparison_df['Match Score'],
                    marker_color='lightblue',
                    text=[f"{score:.1%}" for score in comparison_df['Match Score']],
                    textposition='auto'
                )
            ])
            
            fig_scores.update_layout(
                title="Match Score Comparison",
                yaxis_title="Score",
                yaxis=dict(tickformat=".0%"),
                showlegend=False
            )
            
            st.plotly_chart(fig_scores, use_container_width=True)
        
        with col2:
            # Skill Analysis
            skills_data = []
            for _, row in comparison_df.iterrows():
                skills_data.extend([
                    {'Model': row['Model'], 'Type': 'Matched', 'Count': row['Matched Skills']},
                    {'Model': row['Model'], 'Type': 'Missing', 'Count': row['Missing Skills']}
                ])
            
            skills_df = pd.DataFrame(skills_data)
            
            fig_skills = px.bar(
                skills_df,
                x='Model',
                y='Count',
                color='Type',
                title="Skill Matching Analysis",
                color_discrete_map={'Matched': 'green', 'Missing': 'red'}
            )
            
            st.plotly_chart(fig_skills, use_container_width=True)
    
    def _display_detailed_breakdown(self, results):
        """Display detailed breakdown of each model's results"""
        st.subheader("🔬 Detailed Model Analysis")
        
        # Create tabs for each model
        tab_names = [model_type.replace('_', ' ').title() for model_type in results.keys()]
        tabs = st.tabs(tab_names)
        
        for idx, (model_type, result) in enumerate(results.items()):
            with tabs[idx]:
                self._display_model_details(model_type, result)
    
    def _display_model_details(self, model_type, result):
        """Display detailed information for a specific model"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Model Type:** {result.get('model_type', model_type)}")
            st.write(f"**Match Score:** {result.get('match_score', 0):.1%}")
            st.write(f"**Confidence:** {result.get('confidence', 0):.1%}")
            
            if 'techniques_used' in result:
                st.write("**Techniques Used:**")
                for technique in result['techniques_used']:
                    st.write(f"• {technique}")
            elif 'features_used' in result:
                st.write("**Features Used:**")
                for feature in result['features_used']:
                    st.write(f"• {feature}")
        
        with col2:
            # Model-specific metrics
            if model_type == 'traditional':
                st.write("**Traditional ML Metrics:**")
                st.write(f"• TF-IDF Similarity: {result.get('tfidf_similarity', 0):.3f}")
                st.write(f"• Jaccard Similarity: {result.get('jaccard_similarity', 0):.3f}")
            
            elif model_type == 'deep_learning':
                st.write("**Deep Learning Metrics:**")
                st.write(f"• Sentence Similarity: {result.get('sentence_similarity', 0):.3f}")
                st.write(f"• BERT Similarity: {result.get('bert_similarity', 0):.3f}")
                st.write(f"• Neural Network Score: {result.get('neural_network_score', 0):.3f}")
        
        # Skills comparison
        if result.get('matched_skills') or result.get('missing_skills'):
            st.write("**Skill Analysis:**")
            
            skill_col1, skill_col2 = st.columns(2)
            
            with skill_col1:
                if result.get('matched_skills'):
                    st.success(f"**Matched Skills ({len(result['matched_skills'])}):**")
                    for skill in result['matched_skills'][:10]:  # Show top 10
                        st.write(f"✅ {skill}")
                    
                    if len(result['matched_skills']) > 10:
                        with st.expander(f"Show all {len(result['matched_skills'])} matched skills"):
                            for skill in result['matched_skills']:
                                st.write(f"✅ {skill}")
            
            with skill_col2:
                if result.get('missing_skills'):
                    st.error(f"**Missing Skills ({len(result['missing_skills'])}):**")
                    for skill in result['missing_skills'][:10]:  # Show top 10
                        st.write(f"❌ {skill}")
                    
                    if len(result['missing_skills']) > 10:
                        with st.expander(f"Show all {len(result['missing_skills'])} missing skills"):
                            for skill in result['missing_skills']:
                                st.write(f"❌ {skill}")
        
        # Deep learning specific breakdown
        if model_type == 'deep_learning' and 'skill_category_breakdown' in result:
            st.write("**Skill Category Breakdown:**")
            
            category_data = []
            for category, info in result['skill_category_breakdown'].items():
                category_data.append({
                    'Category': category.replace('_', ' ').title(),
                    'Matched': len(info['matched']),
                    'Missing': len(info['missing']),
                    'Coverage': info['coverage']
                })
            
            if category_data:
                category_df = pd.DataFrame(category_data)
                
                fig = px.bar(
                    category_df,
                    x='Category',
                    y=['Matched', 'Missing'],
                    title="Skills by Category",
                    color_discrete_map={'Matched': 'green', 'Missing': 'red'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_performance_analysis(self, results):
        """Display performance analysis and recommendations"""
        st.subheader("📈 Performance Analysis & Recommendations")
        
        # Calculate performance metrics
        scores = [result.get('match_score', 0) for result in results.values()]
        confidences = [result.get('confidence', 0) for result in results.values()]
        
        avg_score = np.mean(scores)
        score_variance = np.var(scores)
        avg_confidence = np.mean(confidences)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Score", f"{avg_score:.1%}")
        
        with col2:
            st.metric("Score Consistency", f"{1-score_variance:.1%}")
        
        with col3:
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        # Analysis insights
        st.write("**Analysis Insights:**")
        
        if score_variance < 0.01:  # Low variance
            st.success("✅ Models show consistent predictions - high reliability")
        else:
            st.warning("⚠️ Models show significant disagreement - consider additional analysis")
        
        if avg_score > 0.7:
            st.success("✅ Strong overall match - candidate appears well-suited for the position")
        elif avg_score > 0.5:
            st.info("ℹ️ Moderate match - candidate has potential with some skill gaps")
        else:
            st.warning("⚠️ Low match score - significant skill development may be needed")
        
        # Model-specific recommendations
        st.write("**Model-Specific Insights:**")
        
        for model_type, result in results.items():
            model_name = model_type.replace('_', ' ').title()
            score = result.get('match_score', 0)
            confidence = result.get('confidence', 0)
            
            if model_type == 'traditional':
                if score > 0.6:
                    st.write(f"🔵 **{model_name}:** Strong keyword and skill matching detected")
                else:
                    st.write(f"🔵 **{model_name}:** Consider improving keyword alignment and skill mentions")
            
            elif model_type == 'deep_learning':
                if score > 0.6:
                    st.write(f"🟢 **{model_name}:** Strong semantic similarity and contextual understanding")
                else:
                    st.write(f"🟢 **{model_name}:** Focus on improving contextual relevance and semantic alignment")
        
        # Recommendations
        st.write("**Recommendations:**")
        
        all_missing_skills = set()
        for result in results.values():
            all_missing_skills.update(result.get('missing_skills', []))
        
        if all_missing_skills:
            st.write("**Priority Skills to Develop:**")
            priority_skills = list(all_missing_skills)[:8]  # Top 8 missing skills
            for skill in priority_skills:
                st.write(f"• {skill}")
        
        if avg_score < 0.5:
            st.info("""
            **Improvement Suggestions:**
            - Enhance resume with more relevant keywords from the job description
            - Add specific examples and quantifiable achievements
            - Include more technical skills mentioned in the job posting
            - Consider obtaining certifications in missing skill areas
            """)
