"""
Send real validation transactions to the running API, like live traffic.

Start the API first, then run from the project folder:
    python src/replay.py --week 0 --n 500
    python src/replay.py --week 4 --n 500
"""
import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8000"


def clean(v):
    if pd.isna(v):
        return None
    return v.item() if hasattr(v, "item") else v


def post(path, payload):
    req = urllib.request.Request(
        URL + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, default=0, help="week of the validation period (0-5)")
    ap.add_argument("--n", type=int, default=500, help="how many transactions to send")
    args = ap.parse_args()

    features = json.loads((ROOT / "models" / "metadata.json").read_text())["features"]
    valid = pd.read_parquet(ROOT / "data" / "processed" / "valid.parquet")
    week = (valid["day"] - valid["day"].min()) // 7
    batch = valid[week == args.week].sample(args.n, random_state=0)

    blocked = caught = 0
    for _, row in batch.iterrows():
        res = post("/predict", {"features": {f: clean(row[f]) for f in features}})
        blocked += res["decision"] == "block"
        caught += res["decision"] == "block" and row["isFraud"] == 1

    frauds = int(batch["isFraud"].sum())
    print(f"Week {args.week}: sent {len(batch)}, blocked {blocked}, "
          f"real frauds {frauds}, caught {caught}")

    with urllib.request.urlopen(URL + "/monitor") as r:
        print(json.dumps(json.loads(r.read()), indent=2))


if __name__ == "__main__":
    main()
