import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

vader = SentimentIntensityAnalyzer()

device = 0 if torch.cuda.is_available() else -1

roberta_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
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

def visualize_sentiment(df):
    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='Sentiment', y='vader_compound')
    plt.title("VADER Compound Score by IMDb Sentiment")
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    sns.barplot(data=df, x='Sentiment', y='vader_pos', ax=axs[0])
    sns.barplot(data=df, x='Sentiment', y='vader_neu', ax=axs[1])
    sns.barplot(data=df, x='Sentiment', y='vader_neg', ax=axs[2])
    axs[0].set_title("VADER Positive")
    axs[1].set_title("VADER Neutral")
    axs[2].set_title("VADER Negative")
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    sns.barplot(data=df, x='Sentiment', y='roberta_pos', ax=axs[0])
    sns.barplot(data=df, x='Sentiment', y='roberta_neu', ax=axs[1])
    sns.barplot(data=df, x='Sentiment', y='roberta_neg', ax=axs[2])
    axs[0].set_title("DistilBERT Positive")
    axs[1].set_title("DistilBERT Neutral")
    axs[2].set_title("DistilBERT Negative")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csv_path = "./IMDb-Sentiment-Analyzer/IMDB Dataset.csv"

    df = load_data(csv_path)
    df_with_sentiment = run_sentiment_analysis(df)

    print(df_with_sentiment.head())

    df_with_sentiment.to_csv("./IMDb-Sentiment-Analyzer/IMDB Dataset.csv", index=False)

    visualize_sentiment(df_with_sentiment)
