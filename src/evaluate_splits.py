import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from match_products import build_matches
from utils import prepare_products


def split_store_a(A, train_size, validation_size, random_state):
    train, remainder = train_test_split(
        A,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
    )
    validation_fraction = validation_size / (1.0 - train_size)
    validation, test = train_test_split(
        remainder,
        train_size=validation_fraction,
        random_state=random_state,
        shuffle=True,
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def load_labels(path):
    if not path:
        return None
    labels = pd.read_csv(path, dtype=str)
    required = {"item_id_A", "item_id_B"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Labels file is missing required columns: {sorted(missing)}")
    return labels.drop_duplicates("item_id_A").set_index("item_id_A")["item_id_B"].to_dict()


def summarize_matches(split_name, split_df, matches, labels):
    summary = {
        "split": split_name,
        "products_A": int(len(split_df)),
        "matches": int(len(matches)),
        "coverage": round(len(matches) / len(split_df), 4) if len(split_df) else 0.0,
        "avg_score": None,
        "median_score": None,
        "accuracy": None,
        "labeled_predictions": 0,
    }

    if not matches.empty and "score" in matches.columns:
        summary["avg_score"] = round(float(matches["score"].mean()), 4)
        summary["median_score"] = round(float(matches["score"].median()), 4)

    if labels:
        scored = matches[matches["item_id_A"].isin(labels)].copy()
        summary["labeled_predictions"] = int(len(scored))
        if len(scored):
            correct = scored.apply(
                lambda row: str(row["item_id_B"]) == str(labels[str(row["item_id_A"])]),
                axis=1,
            )
            summary["accuracy"] = round(float(correct.mean()), 4)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the existing matcher on train/validation/test splits."
    )
    parser.add_argument("--store-a", default="data/products_A.csv")
    parser.add_argument("--store-b", default="data/products_B.csv")
    parser.add_argument("--labels", default=None, help="Optional CSV with item_id_A,item_id_B ground truth.")
    parser.add_argument("--output", default="outputs/split_eval.json")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-score", type=float, default=0.62)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-a", type=int, default=None)
    parser.add_argument("--sample-b", type=int, default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    args = parser.parse_args()

    A = prepare_products(
        pd.read_csv(args.store_a, dtype=str, low_memory=False, nrows=args.sample_a),
        "A",
    ).reset_index(drop=True)
    B = prepare_products(
        pd.read_csv(args.store_b, dtype=str, low_memory=False, nrows=args.sample_b),
        "B",
    ).reset_index(drop=True)
    labels = load_labels(args.labels)

    summaries = []
    for split_name, split_df in split_store_a(
        A,
        train_size=args.train_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
    ).items():
        matches, run_stats = build_matches(
            split_df,
            B,
            top_k=args.top_k,
            min_score=args.min_score,
            limit=args.limit_per_split,
        )
        summary = summarize_matches(split_name, split_df, matches, labels)
        summary["run_stats"] = run_stats
        summaries.append(summary)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    print(json.dumps(summaries, indent=2))
    if labels is None:
        print(
            "\nNo ground-truth labels were provided, so true accuracy is not available. "
            "Use --labels with item_id_A,item_id_B to compute accuracy."
        )


if __name__ == "__main__":
    main()
