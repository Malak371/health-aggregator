def generate_recommendations(df, correlations, anomalies):
    recommendations = []
    
    avg_sleep = df["sleep_hours"].mean()
    if avg_sleep < 6:
        recommendations.append(
            "You’ve averaged under 6 hours of sleep. Consider prioritizing a consistent bedtime or limiting screen time before sleep."
        )
    
    sleep_sugar_corr = correlations.get("sleep_sugar_corr")
    if sleep_sugar_corr is not None and sleep_sugar_corr < -0.3:
        recommendations.append(
            "Shorter sleep appears to be associated with higher sugar intake. Improving sleep duration may help reduce cravings."
        )
    
    if not anomalies["high_hr"].empty:
        recommendations.append(
            "Several days show elevated resting heart rate. You might consider lighter activity, hydration, or additional rest on those days."
        )
    
    if not recommendations:
        recommendations.append(
            "Your recent data looks stable. Maintaining current habits may help sustain these trends."
        )
    
    return recommendations