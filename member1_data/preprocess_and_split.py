import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.preprocess import clean_tweet

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

SRC = RAW / "training.1600000.processed.noemoticon.csv"

# Sentiment140 cols: target, ids, date, query, user, text
COLS = ['target','ids','date','query','user','text']

def map_label(v):
    # 0=negative, 4=positive in Sentiment140
    return 0 if int(v)==0 else 2 if int(v)==2 else 1 if int(v)==4 else 1

if __name__ == "__main__":
    print("Reading:", SRC)
    df = pd.read_csv(SRC, encoding="latin-1", names=COLS)
    df = df[['target','text']].copy()
    df['label'] = df['target'].map(map_label)
    df['text_clean'] = df['text'].astype(str).apply(clean_tweet)

    # Downsample to make training faster for a mini-project (adjust as needed)
    df_small = pd.concat([
        df[df['label']==0].sample(100000, random_state=42),
        df[df['label']==1].sample(100000, random_state=42)
    ], ignore_index=True)

    train, test = train_test_split(df_small, test_size=0.1, random_state=42, stratify=df_small['label'])
    train, val  = train_test_split(train, test_size=0.1, random_state=42, stratify=train['label'])

    for name, part in [('train',train), ('val',val), ('test',test)]:
        part[['text','text_clean','label']].to_csv(PROC/f'{name}.csv', index=False)

    print("Saved:", list(PROC.iterdir()))
