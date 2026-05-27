import argparse
import os

import pandas as pd

from match_products import generate_candidates, score_pair
from utils import prepare_products


def select_item(A, item_id=None, row_index=None):
    if item_id:
        selected = A[A["item_id"] == str(item_id)]
        if selected.empty:
            raise ValueError(f"Could not find item_id in Store A: {item_id}")
        return selected.iloc[[0]].reset_index(drop=True)

    if row_index is None:
        row_index = 0
    if row_index < 0 or row_index >= len(A):
        raise ValueError(f"row_index must be between 0 and {len(A) - 1}")
    return A.iloc[[row_index]].reset_index(drop=True)


def top_matches_for_item(item_a, B, top_k):
    distances, indices = generate_candidates(item_a, B, top_k)
    a = item_a.iloc[0]
    rows = []

    for rank, (distance, b_pos) in enumerate(zip(distances[0], indices[0]), start=1):
        b = B.iloc[b_pos]
        semantic_score = 1.0 - float(distance)
        final_score = score_pair(a, b, semantic_score)
        rows.append(
            {
                "rank": rank,
                "item_id_A": a["item_id"],
                "item_name_A": a["name"],
                "item_id_B": b["item_id"],
                "item_name_B": b["name"],
                "score": round(final_score, 4),
                "semantic_score": round(semantic_score, 4),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Show top K Store B matches for one Store A item.")
    parser.add_argument("--store-a", default="data/products_A.csv")
    parser.add_argument("--store-b", default="data/products_B.csv")
    parser.add_argument("--item-id", default=None, help="Store A item_id to inspect.")
    parser.add_argument("--row-index", type=int, default=None, help="Store A row index to inspect if item-id is omitted.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sample-b", type=int, default=None, help="Optional Store B sample size for quick tests.")
    parser.add_argument("--output", default=None, help="Optional CSV path for the visualization output.")
    args = parser.parse_args()

    A_raw = pd.read_csv(args.store_a, dtype=str, low_memory=False)
    B_raw = pd.read_csv(args.store_b, dtype=str, low_memory=False, nrows=args.sample_b)

    A = prepare_products(A_raw, "A").reset_index(drop=True)
    B = prepare_products(B_raw, "B").reset_index(drop=True)

    item_a = select_item(A, item_id=args.item_id, row_index=args.row_index)
    matches = top_matches_for_item(item_a, B, args.top_k)

    a = item_a.iloc[0]
    print(f"\nStore A item:")
    print(f"{a['item_id']} | {a['name']}")
    print(f"\nTop {args.top_k} Store B matches:")
    print(matches[["rank", "item_id_B", "item_name_B", "score", "semantic_score"]].to_string(index=False))

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        matches.to_csv(args.output, index=False)
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
