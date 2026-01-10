import pandas as pd

def sleep_vs_mood(df: pd.DataFrame):
    corr = df["sleep_hours"].corr(df["mood_score"])
    
    if corr > 0.4:
        insight = "More sleep is associated with improved mood."
    elif corr > 0.2:
        insight = "More sleep shows a mild positive association with better mood."
    else:
        insight = "There is little association between sleep duration and mood."
        
    return corr, insight

def sleep_vs_sugar(df: pd.DataFrame):
    corr = df["sleep_hours"].corr(df["sugar_grams"])
    
    if corr < -0.4:
        insight = "Shorter sleep is strongly associated with higher sugar intake."
    elif corr < -0.2:
        insight = "Less sleep tends to be associated with increased sugar consumption."
    else:
        insight = "There is little association between sleep duration and sugar intake."
        
    return corr, insight

def sleep_vs_steps(df: pd.DataFrame):
    corr = df["sleep_hours"].corr(df["steps"])
    
    if corr > 0.4:
        insight = "More sleep is strongly associated with higher activity levels."
    elif corr > 0.2:
        insight = "More sleep shows a mild positive association with activity levels."
    else:
        insight = "There is little association between sleep duration and activity levels."
        
    return corr, insight