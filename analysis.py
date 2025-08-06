

import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"review": "Text", "sentiment": "Sentiment"})
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Id"}, inplace=True)
    return df
