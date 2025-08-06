import streamlit as st
import plotly.graph_objects as go
import pandas as pd

class ResultsDisplay:
    def __init__(self):
        pass

    def display_traditional_results(self, result):
        st.subheader("📘 Traditional ML Analysis")
        self._display_result_summary(result)
        self._plot_skill_bar(result)

    def display_deep_learning_results(self, result):
        st.subheader("🤖 Deep Learning Analysis")
        self._display_result_summary(result)
        self._plot_skill_bar(result)

    def _display_result_summary(self, result):
        st.markdown(f"**Model Type:** {result.get('model_type', 'Unknown')}")
        st.metric("🎯 Match Score", f"{result.get('match_score', 0):.1%}")
        st.metric("📊 Confidence", f"{result.get('confidence', 0):.1%}")

        st.markdown("**Top Techniques/Features Used:**")
        if 'techniques_used' in result:
            for tech in result['techniques_used']:
                st.write(f"• {tech}")
        elif 'features_used' in result:
            for feature in result['features_used']:
                st.write(f"• {feature}")

        st.markdown("**Skill Summary:**")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"Matched Skills: {len(result.get('matched_skills', []))}")
        with col2:
            st.error(f"Missing Skills: {len(result.get('missing_skills', []))}")

    def _plot_skill_bar(self, result):
        matched = len(result.get('matched_skills', []))
        missing = len(result.get('missing_skills', []))
        total = matched + missing

        if total == 0:
            st.warning("No skills detected to visualize.")
            return

        fig = go.Figure(data=[
            go.Bar(name='Matched', x=['Skills'], y=[matched], marker_color='green'),
            go.Bar(name='Missing', x=['Skills'], y=[missing], marker_color='red')
        ])
        fig.update_layout(barmode='stack', title="Skill Match Overview")
        st.plotly_chart(fig, use_container_width=True)

        # Optional: list a few matched and missing skills
        with st.expander("🔎 Matched Skills"):
            if matched:
                for skill in result['matched_skills'][:10]:
                    st.write(f"✅ {skill}")
            else:
                st.info("No matched skills found.")

        with st.expander("❌ Missing Skills"):
            if missing:
                for skill in result['missing_skills'][:10]:
                    st.write(f"❌ {skill}")
            else:
                st.info("No missing skills found.")
