import pandas as pd
from pathlib import Path

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
for name in ["train","val","test"]:
    df = pd.read_csv(PROC / f"{name}.csv")
    print(name, df.shape, df['label'].value_counts())
    print(df.head(3))
