"""Streamlit dashboard for Skills Demand Forecasting Platform."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import logging

# Configure page
st.set_page_config(
    page_title="Skills Demand Forecasting Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.insight-box {
    background-color: #e8f4fd;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #1f77b4;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

def load_sample_data():
    """Load sample data for demo purposes."""
    # Sample skills data
    skills_data = {
        'skill_name': ['Python', 'JavaScript', 'React', 'AWS', 'Docker', 'Kubernetes', 'PostgreSQL', 'MongoDB'],
        'frequency': [1250, 1100, 950, 800, 600, 450, 700, 550],
        'frequency_percentage': [25.5, 22.8, 19.2, 16.4, 12.1, 9.8, 14.2, 11.3],
        'growth_rate': [15.2, 8.5, 22.1, 28.3, 18.7, 35.4, 5.2, 12.8],
        'avg_salary_impact': [12000, 8000, 10000, 15000, 8500, 12000, 6000, 7500]
    }
    return pd.DataFrame(skills_data)

def create_skill_demand_chart(df):
    """Create skill demand visualization."""
    fig = px.bar(
        df.head(10),
        x='skill_name',
        y='frequency',
        title='Top 10 Most In-Demand Skills',
        labels={'frequency': 'Job Mentions', 'skill_name': 'Skill'},
        color='growth_rate',
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(height=400)
    return fig

def create_growth_trend_chart(df):
    """Create growth trend visualization."""
    fig = px.scatter(
        df,
        x='frequency_percentage',
        y='growth_rate',
        size='avg_salary_impact',
        color='skill_name',
        title='Skills: Current Demand vs Growth Rate',
        labels={
            'frequency_percentage': 'Current Market Share (%)',
            'growth_rate': 'Growth Rate (%)',
            'avg_salary_impact': 'Salary Impact ($)'
        },
        hover_data=['skill_name']
    )
    fig.update_layout(height=500)
    return fig

def create_salary_impact_chart(df):
    """Create salary impact visualization."""
    fig = px.bar(
        df.sort_values('avg_salary_impact', ascending=True).head(8),
        x='avg_salary_impact',
        y='skill_name',
        orientation='h',
        title='Average Salary Impact by Skill',
        labels={'avg_salary_impact': 'Salary Impact ($)', 'skill_name': 'Skill'},
        color='avg_salary_impact',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400)
    return fig

def main():
    """Main dashboard application."""

    # Header
    st.markdown('<h1 class="main-header">📊 Skills Demand Forecasting Platform</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Skill Analysis", "Demand Forecasting", "Salary Intelligence", "Career Guidance"]
    )

    # Load data
    df = load_sample_data()

    if page == "Overview":
        st.markdown("## 📈 Market Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Skills Tracked", "2,847", delta="127 new")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Growth Rate", "16.8%", delta="2.3%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Jobs Analyzed", "124,573", delta="1,205 today")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Forecast Accuracy", "87.2%", delta="1.1%")
            st.markdown('</div>', unsafe_allow_html=True)

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            fig1 = create_skill_demand_chart(df)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = create_salary_impact_chart(df)
            st.plotly_chart(fig2, use_container_width=True)

        # Growth trend chart
        fig3 = create_growth_trend_chart(df)
        st.plotly_chart(fig3, use_container_width=True)

        # Insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### 💡 Key Insights")
        st.markdown("""
        - **Kubernetes** shows the highest growth rate at 35.4%, indicating strong containerization adoption
        - **AWS** skills command the highest salary premium with an average impact of $15,000
        - **JavaScript** remains the most stable skill with consistent high demand
        - **Python** leads in overall job mentions, reinforcing its versatility across domains
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    elif page == "Skill Analysis":
        st.markdown("## 🔍 Skill Analysis")

        # Skill selector
        selected_skill = st.selectbox("Select a skill to analyze:", df['skill_name'].tolist())

        if selected_skill:
            skill_data = df[df['skill_name'] == selected_skill].iloc[0]

            # Skill metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Job Mentions", f"{skill_data['frequency']:,}")
            with col2:
                st.metric("Market Share", f"{skill_data['frequency_percentage']:.1f}%")
            with col3:
                st.metric("Growth Rate", f"{skill_data['growth_rate']:.1f}%")

            # Sample time series for selected skill
            dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
            base_value = skill_data['frequency'] / 12
            trend = np.linspace(0, skill_data['growth_rate']/100 * base_value, len(dates))
            noise = np.random.normal(0, base_value * 0.1, len(dates))
            values = base_value + trend + noise

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name=f'{selected_skill} Demand',
                line=dict(color='#1f77b4', width=3)
            ))

            fig.update_layout(
                title=f'{selected_skill} Demand Trend Over Time',
                xaxis_title='Date',
                yaxis_title='Job Mentions',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    elif page == "Demand Forecasting":
        st.markdown("## 🔮 Demand Forecasting")

        # Forecasting controls
        col1, col2 = st.columns(2)

        with col1:
            selected_skills = st.multiselect(
                "Select skills to forecast:",
                df['skill_name'].tolist(),
                default=['Python', 'JavaScript', 'React']
            )

        with col2:
            forecast_months = st.slider("Forecast horizon (months):", 3, 24, 12)

        if selected_skills:
            # Generate sample forecast data
            for skill in selected_skills:
                st.markdown(f"### {skill} Forecast")

                # Historical and forecast data
                historical_dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
                forecast_dates = pd.date_range(start='2025-01-01', periods=forecast_months, freq='M')

                skill_data = df[df['skill_name'] == skill].iloc[0]
                base_value = skill_data['frequency'] / 12

                # Historical data
                historical_trend = np.linspace(base_value * 0.8, base_value, len(historical_dates))
                historical_noise = np.random.normal(0, base_value * 0.1, len(historical_dates))
                historical_values = historical_trend + historical_noise

                # Forecast data
                growth_rate = skill_data['growth_rate'] / 100 / 12  # Monthly growth
                forecast_trend = [base_value * (1 + growth_rate) ** i for i in range(forecast_months)]
                forecast_noise = np.random.normal(0, base_value * 0.05, forecast_months)
                forecast_values = np.array(forecast_trend) + forecast_noise

                # Confidence intervals
                confidence_factor = 0.2
                upper_bound = forecast_values * (1 + confidence_factor)
                lower_bound = forecast_values * (1 - confidence_factor)

                # Create forecast chart
                fig = go.Figure()

                # Historical data
                fig.add_trace(go.Scatter(
                    x=historical_dates,
                    y=historical_values,
                    mode='lines',
                    name='Historical',
                    line=dict(color='#1f77b4', width=2)
                ))

                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_values,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ))

                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=list(forecast_dates) + list(forecast_dates[::-1]),
                    y=list(upper_bound) + list(lower_bound[::-1]),
                    fill='toself',
                    fillcolor='rgba(255,127,14,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ))

                fig.update_layout(
                    title=f'{skill} Demand Forecast',
                    xaxis_title='Date',
                    yaxis_title='Predicted Job Mentions',
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Forecast insights
                growth_percentage = ((forecast_values[-1] - base_value) / base_value) * 100
                st.markdown(f"""
                **Forecast Insights for {skill}:**
                - Predicted growth: {growth_percentage:.1f}% over {forecast_months} months
                - Average monthly demand: {np.mean(forecast_values):.0f} job mentions
                - Trend: {'Increasing' if growth_percentage > 0 else 'Decreasing'}
                """)

    elif page == "Salary Intelligence":
        st.markdown("## 💰 Salary Intelligence")

        # Salary prediction form
        st.markdown("### Salary Prediction Tool")

        col1, col2 = st.columns(2)

        with col1:
            selected_skills_salary = st.multiselect(
                "Your skills:",
                df['skill_name'].tolist(),
                default=['Python', 'AWS']
            )

            experience_level = st.selectbox(
                "Experience Level:",
                ["Entry Level", "Mid Level", "Senior Level", "Principal Level", "Management"]
            )

            location = st.selectbox(
                "Location:",
                ["San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX", "Remote", "Chicago, IL"]
            )

        with col2:
            employment_type = st.selectbox(
                "Employment Type:",
                ["Full-time", "Contract", "Part-time"]
            )

            company_size = st.selectbox(
                "Company Size:",
                ["Startup (1-50)", "Medium (51-200)", "Large (201-1000)", "Enterprise (1000+)"]
            )

            remote_work = st.checkbox("Remote work allowed")

        if st.button("Predict Salary", type="primary"):
            # Sample salary prediction logic
            base_salary = {
                "Entry Level": 70000,
                "Mid Level": 95000,
                "Senior Level": 130000,
                "Principal Level": 170000,
                "Management": 200000
            }[experience_level]

            # Location multiplier
            location_multiplier = {
                "San Francisco, CA": 1.4,
                "New York, NY": 1.3,
                "Seattle, WA": 1.25,
                "Austin, TX": 1.1,
                "Remote": 1.0,
                "Chicago, IL": 1.05
            }[location]

            # Skills bonus
            skill_bonus = 0
            for skill in selected_skills_salary:
                skill_data = df[df['skill_name'] == skill]
                if not skill_data.empty:
                    skill_bonus += skill_data.iloc[0]['avg_salary_impact']

            predicted_salary = base_salary * location_multiplier + skill_bonus

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Predicted Salary", f"${predicted_salary:,.0f}")
            with col2:
                st.metric("Salary Range", f"${predicted_salary*0.85:,.0f} - ${predicted_salary*1.15:,.0f}")
            with col3:
                st.metric("Market Percentile", "75th")

            # Salary breakdown
            st.markdown("### Salary Breakdown")
            breakdown_data = {
                'Component': ['Base Salary', 'Location Adjustment', 'Skills Premium', 'Total'],
                'Amount': [
                    base_salary,
                    base_salary * (location_multiplier - 1),
                    skill_bonus,
                    predicted_salary
                ]
            }

            breakdown_df = pd.DataFrame(breakdown_data)
            fig = px.bar(
                breakdown_df[:-1],  # Exclude total
                x='Component',
                y='Amount',
                title='Salary Components Breakdown',
                color='Amount',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Career Guidance":
        st.markdown("## 🎯 Career Guidance")

        # Current skills assessment
        st.markdown("### Current Skills Assessment")

        current_skills = st.multiselect(
            "What are your current skills?",
            df['skill_name'].tolist(),
            default=['Python', 'SQL']
        )

        target_role = st.selectbox(
            "What's your target role?",
            ["Data Scientist", "Full Stack Developer", "DevOps Engineer", "Product Manager", "ML Engineer"]
        )

        if current_skills and target_role:
            # Skill gap analysis
            role_skills = {
                "Data Scientist": ['Python', 'SQL', 'Machine Learning', 'Statistics', 'Pandas', 'Scikit-learn'],
                "Full Stack Developer": ['JavaScript', 'React', 'Node.js', 'Python', 'PostgreSQL', 'AWS'],
                "DevOps Engineer": ['Docker', 'Kubernetes', 'AWS', 'Terraform', 'Jenkins', 'Python'],
                "Product Manager": ['SQL', 'Analytics', 'A/B Testing', 'Project Management', 'Communication'],
                "ML Engineer": ['Python', 'TensorFlow', 'PyTorch', 'AWS', 'Docker', 'Kubernetes']
            }

            required_skills = role_skills.get(target_role, [])
            missing_skills = [skill for skill in required_skills if skill not in current_skills]

            # Display skill gap
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### ✅ Your Current Skills")
                for skill in current_skills:
                    if skill in required_skills:
                        st.markdown(f"- {skill} ✓")
                    else:
                        st.markdown(f"- {skill}")

            with col2:
                st.markdown("#### 📚 Skills to Learn")
                for skill in missing_skills:
                    skill_data = df[df['skill_name'] == skill]
                    if not skill_data.empty:
                        growth_rate = skill_data.iloc[0]['growth_rate']
                        st.markdown(f"- {skill} (Growth: {growth_rate:.1f}%)")
                    else:
                        st.markdown(f"- {skill}")

            # Learning roadmap
            st.markdown("### 🗺️ Learning Roadmap")

            if missing_skills:
                # Prioritize skills by growth rate and salary impact
                skill_priorities = []
                for skill in missing_skills:
                    skill_data = df[df['skill_name'] == skill]
                    if not skill_data.empty:
                        priority_score = skill_data.iloc[0]['growth_rate'] + (skill_data.iloc[0]['avg_salary_impact'] / 1000)
                        skill_priorities.append((skill, priority_score))

                skill_priorities.sort(key=lambda x: x[1], reverse=True)

                for i, (skill, score) in enumerate(skill_priorities, 1):
                    st.markdown(f"**{i}. {skill}**")
                    st.markdown(f"   - Priority Score: {score:.1f}")
                    st.markdown(f"   - Estimated learning time: {np.random.randint(2, 8)} weeks")
                    st.markdown(f"   - Resources: Online courses, documentation, practice projects")
                    st.markdown("")

            # Salary progression
            st.markdown("### 💰 Salary Progression Forecast")

            current_salary_estimate = 85000  # Base estimate

            progression_data = {
                'Stage': ['Current', 'After 6 months', 'After 1 year', 'Target Role'],
                'Estimated Salary': [
                    current_salary_estimate,
                    current_salary_estimate + 5000,
                    current_salary_estimate + 15000,
                    current_salary_estimate + 35000
                ],
                'Skills Acquired': [
                    len(current_skills),
                    len(current_skills) + 1,
                    len(current_skills) + 2,
                    len(required_skills)
                ]
            }

            progression_df = pd.DataFrame(progression_data)

            fig = px.line(
                progression_df,
                x='Stage',
                y='Estimated Salary',
                markers=True,
                title='Career Progression Forecast',
                labels={'Estimated Salary': 'Salary ($)'}
            )

            fig.update_traces(line=dict(width=3, color='#1f77b4'))
            fig.update_layout(height=400)

            st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "**Skills Demand Forecasting Platform** | "
        "Powered by AI and Machine Learning | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

if __name__ == "__main__":
    main()
