import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from tqdm import tqdm

vader = SentimentIntensityAnalyzer()

roberta_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512
)

def load_data(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"review": "Text", "sentiment": "Sentiment"})
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Id"}, inplace=True)
    return df

def analyze_vader(text):
    scores = vader.polarity_scores(text)
    return {f"vader_{k}": v for k, v in scores.items()}

def analyze_roberta(text):
    try:
        text = text[:1000]
        result = roberta_pipeline(text)[0]
        label = result['label']
        score = result['score']
        return {
            'roberta_neg': score if label == 'NEGATIVE' else 0.0,
            'roberta_neu': 0.0,
            'roberta_pos': score if label == 'POSITIVE' else 0.0
        }
    except:
        return {
            'roberta_neg': 0.0,
            'roberta_neu': 0.0,
            'roberta_pos': 0.0
        }

def run_sentiment_analysis(df):
    results = {}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text = row["Text"]
            myid = row["Id"]
            vader_scores = analyze_vader(text)
            roberta_scores = analyze_roberta(text)
            combined = {**vader_scores, **roberta_scores}
            results[myid] = combined
        except:
            pass
    results_df = pd.DataFrame(results).T
    results_df = results_df.reset_index().rename(columns={"index": "Id"})
    merged = pd.merge(df, results_df, on="Id", how="left")
    return merged
