import argparse
import json
from typing import Dict

import numpy as np
import pandas as pd


def topk_site_accuracy(eval_df: pd.DataFrame, k: int) -> float:
    grouped = eval_df.groupby("molecule_key", sort=False)
    hits = 0
    total = 0
    for _, group in grouped:
        ranked = group.sort_values("pred_proba", ascending=False).head(k)
        hits += int((ranked["is_true_site"] == 1).any())
        total += 1
    return float(hits / max(1, total))


def selected_candidate_mae(eval_df: pd.DataFrame) -> float:
    top = eval_df.sort_values(["molecule_key", "pred_proba"], ascending=[True, False]).groupby("molecule_key").head(1)
    if "candidate_prior_mean_pka" not in top.columns or "pka_value" not in top.columns:
        return float("nan")
    return float(np.mean(np.abs(top["candidate_prior_mean_pka"] - top["pka_value"])))


def evaluate(eval_df: pd.DataFrame) -> Dict[str, float]:
    return {
        "top1_site_acc": topk_site_accuracy(eval_df, k=1),
        "top3_site_acc": topk_site_accuracy(eval_df, k=3),
        "top5_site_acc": topk_site_accuracy(eval_df, k=5),
        "selected_candidate_prior_mae": selected_candidate_mae(eval_df),
        "rows": int(len(eval_df)),
        "molecules": int(eval_df["molecule_key"].nunique()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate-level site/state predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions)
    required = {"molecule_key", "pred_proba", "is_true_site"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    metrics = evaluate(df)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
