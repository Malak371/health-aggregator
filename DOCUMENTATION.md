# Personal Health & Wellness Aggregator ❤️‍🩹

## Design Documentation

### 1. Problem Overview

Health-conscious individuals often use multiple disconnected tools to track their wellness, such as wearables for sleep and activity, apps for nutrition, and manual logs for mood. Because this data is fragmented, it becomes difficult to understand how these signals interact and how daily habits influence overall well-being.

This project addresses that problem by unifying multiple health signals into a single platform and transforming raw data into explainable, actionable insights.

### 2. Solution Overview

The Personal Health & Wellness Aggregator is a lightweight, AI-enabled prototype that:

- Unifies sleep, activity, nutrition, mood, and heart rate data

- Discovers correlations between health behaviors

- Detects anomalies relative to a user’s personal baseline

- Translates insights into low-risk, actionable next steps

The system emphasizes explainability, personalization, and responsible AI practices rather than black-box predictions.

### 3. Architecture & Design

#### Data Layer

- Synthetic health data generated via Python to preserve privacy

- Metrics include:

  - Sleep duration

  - Daily steps

  - Resting heart rate

  - Mood score

  - Sugar intake (nutrition proxy)

#### Analysis Layer

- `correlations.py`: Identifies relationships between sleep and other health signals using statistical correlation

- `anomalies.py`: Detects deviations from personal baselines using rolling averages and standard deviation

- `recommendations.py`: Converts insights and anomalies into non-prescriptive lifestyle suggestions

#### Presentation Layer

- Streamlit-based UI providing:

  - Unified health dashboard

  - AI-powered insights

  - Proactive alerts

  - Personalized next steps

### 4. AI / ML Approach

Rather than complex machine learning models, this prototype uses explainable analytical techniques:

- Pearson correlation for relationship discovery

- Rolling-window statistics for personalized baselines

- Rule-based logic for recommendations

This approach ensures transparency, interpretability, and user trust—especially important for sensitive health data.

### 5. Tech Stack

- **Language**: Python

- **Data Processing**: pandas, numpy

- **Visualization & UI**: Streamlit

- **Analytics**: Statistical correlation, rolling-window anomaly detection

- **Environment**: Local development in Visual Studio Code

### 6. Responsible AI Considerations

- Uses correlation, not causation

- Avoids medical diagnoses or prescriptive advice

- Employs user-specific baselines instead of global thresholds

- Uses synthetic data to protect privacy

- Includes clear disclaimers in the UI

### 7. Future Enhancements

- Integration with real health APIs (Apple HealthKit, Google Fit, MyFitnessPal)

- Longer-term trend modeling and seasonality detection

- User-configurable alert thresholds

- Expanded nutrition metrics (macros, meal timing)

- Secure authentication and encrypted data storage
