"""
Save a few real validation transactions as JSON files you can send to the API.

Run from the project folder:
    python src/make_examples.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
meta = json.loads((ROOT / "models" / "metadata.json").read_text())
features = meta["features"]

valid = pd.read_parquet(ROOT / "data" / "processed" / "valid.parquet")
out = ROOT / "examples"
out.mkdir(exist_ok=True)


def clean(v):
    if pd.isna(v):
        return None
    if hasattr(v, "item"):          # numpy number -> plain Python number
        return v.item()
    return v


picks = {
    "fraud_example": valid[valid["isFraud"] == 1].sample(1, random_state=1),
    "legit_example": valid[valid["isFraud"] == 0].sample(1, random_state=1),
}

for name, row in picks.items():
    payload = {"features": {f: clean(row.iloc[0][f]) for f in features}}
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2))
    print(f"Saved examples/{name}.json  (amount ${row.iloc[0]['TransactionAmt']:.2f})")
