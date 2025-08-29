import os, zipfile, csv
from pathlib import Path
import requests

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
RAW = DATA / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def download_sentiment140():
    url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    out = RAW / "sentiment140.zip"
    if out.exists():
        print("Sentiment140 zip already exists.")
        return out
    print("Downloading Sentiment140...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out.write_bytes(r.content)
    print("Downloaded:", out)
    return out

def extract_zip(zpath: Path):
    with zipfile.ZipFile(zpath, 'r') as zf:
        zf.extractall(RAW)
    print("Extracted to", RAW)

if __name__ == "__main__":
    z = download_sentiment140()
    extract_zip(z)
    print("Done. Files in data/raw:", list(RAW.iterdir()))
