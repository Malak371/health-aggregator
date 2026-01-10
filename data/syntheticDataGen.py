import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

days = 30
startDate = datetime.today() - timedelta(days=days)

data = []

for i in range(days):
    date = startDate + timedelta(days=i)
    sleep = np.random.uniform(3, 10)
    
    if sleep < 6:
        resting_hr = np.random.normal(74, 3)
        steps = np.random.randint(1000, 7000)
        mood = np.random.uniform(1, 2.5)
        sugar_grams = np.random.randint(60, 120)
    else:
        resting_hr = np.random.normal(66, 3)
        steps = np.random.randint(7000, 15000)
        mood = np.random.uniform(2.5, 5)
        sugar_grams = np.random.randint(20, 60)
        
    data.append({
        "data": date.strftime("%Y-%m-%d"),
        "sleep_hours": round(sleep, 1),
        "resting_heart_rate": round(resting_hr, 1),
        "steps": steps,
        "mood_score": round(mood, 1),
        "sugar_grams": sugar_grams
    })

df = pd.DataFrame(data)
df.to_csv("synthetic_health_data.csv", index=False)
