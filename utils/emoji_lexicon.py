import csv
from pathlib import Path

# Minimal fallback lexicon (extendable)
FALLBACK = {
    # Positive
    "🔥": 0.9, "😊": 0.8, "😁": 0.8, "😍": 0.9, "😎": 0.5, "👍": 0.6,
    # Negative (expanded)
    "😢": -0.6, "😭": -0.8, "😡": -0.8, "😠": -0.7, "😤": -0.6, "😞": -0.5,
    "😒": -0.4, "☹️": -0.6, "😕": -0.3, "😔": -0.5, "😩": -0.7, "😫": -0.7,
    "👎": -0.6, "😴": -0.3, "🤮": -0.8
}

def load_emoji_sentiment_csv(csv_path: str | Path | None):
    """
    Optional: load the Emoji Sentiment Ranking CSV (Kralj Novak et al., 2015).
    Expected columns include 'emoji' and a polarity/score column; if not present,
    fall back to the built-in mini-lexicon.
    """
    if not csv_path:
        return FALLBACK

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return FALLBACK

    lex = {}
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Try to infer a sentiment score column
        candidate_cols = [c for c in reader.fieldnames if c and 'score' in c.lower() or 'polarity' in c.lower()]
        for row in reader:
            emo = row.get('emoji') or row.get('Emoji') or row.get('char')
            if not emo:
                continue
            score = None
            for c in candidate_cols:
                try:
                    score = float(row[c])
                    break
                except:
                    pass
            if score is None:
                # Heuristic: pos-neg normalized if available
                pos = float(row.get('Positive', 0) or 0)
                neg = float(row.get('Negative', 0) or 0)
                tot = pos + neg + float(row.get('Neutral', 0) or 0) + 1e-6
                score = (pos - neg) / tot
            lex[emo] = max(min(score, 1.0), -1.0)
    return lex

def emoji_sentiment_feature(text: str, emo_lex: dict) -> float:
    """Aggregate emoji sentiment across a text."""
    s = 0.0
    n = 0
    for ch in text:
        if ch in emo_lex:
            s += emo_lex[ch]
            n += 1
    return s / n if n else 0.0
