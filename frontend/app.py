import streamlit as st 
import pandas as pd 
from analysis import correlations, anomalies, recommendations

st.set_page_config(
    page_title="Personal Health and Wellness Aggregator",
    layout="wide"
)

st.title("Personal Health and Wellness Aggregator ❤️‍🩹")
st.write(
    """
    This prototype unifies personal health signals (sleep, activity, nutrition, and mood)
    and applies explainable analytics to surface actionable wellness insights.
    """
)

@st.cache_data
def load_data():
    df = pd.read_csv("./data/synthetic_health_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

st.header("Health Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sleep Duration")
    st.line_chart(df.set_index("date")["sleep_hours"])
    
with col2:
    st.subheader("Daily Steps")
    st.line_chart(df.set_index("date")["steps"])
    
st.subheader("Resting Heart Rate")
st.line_chart(df.set_index("date")["resting_heart_rate"])

st.header("🤖 AI-Powered Insights")

corr1, insight1 = correlations.sleep_vs_mood(df)
corr2, insight2 = correlations.sleep_vs_sugar(df)
corr3, insight3 = correlations.sleep_vs_steps(df)

insight_col1, insight_col2, insight_col3 = st.columns(3)

with insight_col1:
    st.metric(
        label="Sleep ↔ Mood Correlation",
        value=f"{corr1:.2f}"
    )
    st.write(insight1)
    
with insight_col2:
    st.metric(
        label="Sleep ↔ Sugar Correlation",
        value=f"{corr2:.2f}"
    )
    st.write(insight2)
    
with insight_col3:
    st.metric(
        label="Sleep ↔ Activity Correlation",
        value=f"{corr3:.2f}"
    )
    st.write(insight3)
    
st.header("🚨 Proactive Health Alerts")

low_sleep_days = anomalies.detect_poor_sleep(df)
high_hr_days = anomalies.detect_high_resting_hr(df)

if not low_sleep_days.empty:
    st.subheader("⚠️ Critically Low Sleep Days")
    st.dataframe(low_sleep_days[["date", "sleep_hours"]])

if not high_hr_days.empty:
    st.subheader("⚠️ Elevated Resting Heart Rate")
    st.dataframe(
        high_hr_days[["date", "resting_heart_rate", "reason"]]
    )
    
if low_sleep_days.empty and high_hr_days.empty:
    st.success("No significant anomalies detected during this period.")
    

correlations_data = {
    "sleep_sugar_corr": corr2
}

anomalies_data = {
    "high_hr": high_hr_days
}

recs = recommendations.generate_recommendations(
    df,
    correlations_data,
    anomalies_data
)

st.header("✅ Personalized Next Steps")

for rec in recs:
    st.write(f"- {rec}")
    
st.caption(
    "Insights are based on observed data patterns and are not medical advice."
)
