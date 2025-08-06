

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


vader = SentimentIntensityAnalyzer()

def load_data(path):
    
    df = pd.read_csv(path)
    df = df.rename(columns={"review": "Text", "sentiment": "Sentiment"})
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Id"}, inplace=True)
    return df

def analyze_vader(text):
    
    scores = vader.polarity_scores(text)
    return {f"vader_{k}": v for k, v in scores.items()}
