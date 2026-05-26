import argparse
import os

import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from utils import prepare_products, row_to_product


def size_score(a, b):
    if pd.isna(a["size_value"]) or pd.isna(b["size_value"]):
        return 0.0
    if not a["size_unit"] or a["size_unit"] != b["size_unit"]:
        return 0.0
    max_size = max(float(a["size_value"]), float(b["size_value"]))
    if max_size <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(float(a["size_value"]) - float(b["size_value"])) / max_size)


def brand_score(a, b):
    brand_a = a["brand_norm"]
    brand_b = b["brand_norm"]
    if not brand_a or not brand_b:
        return 0.0
    if brand_a == brand_b:
        return 1.0
    if a["is_private_label_norm"] and b["brand_norm"] in {"wegmans", "great value"}:
        return 0.75
    return fuzz.token_set_ratio(brand_a, brand_b) / 100


def organic_penalty(a, b):
    return 0.10 if bool(a["is_organic_norm"]) != bool(b["is_organic_norm"]) else 0.0


def score_pair(a, b, semantic_score):
    name_score = fuzz.token_set_ratio(a["name_norm"], b["name_norm"]) / 100
    category_score = fuzz.token_set_ratio(a["category_norm"], b["category_norm"]) / 100
    score = (
        0.40 * name_score
        + 0.25 * semantic_score
        + 0.15 * brand_score(a, b)
        + 0.12 * size_score(a, b)
        + 0.08 * category_score
        - organic_penalty(a, b)
    )
    return max(0.0, min(1.0, score))


def generate_candidates(A, B, top_k):
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=120_000,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(pd.concat([A["search_text"], B["search_text"]]))
    a_matrix = matrix[: len(A)]
    b_matrix = matrix[len(A) :]

    nn = NearestNeighbors(n_neighbors=top_k, metric="cosine", algorithm="brute")
    nn.fit(b_matrix)
    distances, indices = nn.kneighbors(a_matrix)
    return distances, indices


def build_matches(A, B, top_k=30, min_score=0.62, limit=None):
    distances, indices = generate_candidates(A, B, top_k)
    results = []

    for a_pos, a in tqdm(list(A.iterrows()), desc="Matching products"):
        ranked = []
        for distance, b_pos in zip(distances[a_pos], indices[a_pos]):
            b = B.iloc[b_pos]
            semantic_score = 1.0 - float(distance)
            ranked.append((score_pair(a, b, semantic_score), semantic_score, b))

        ranked.sort(key=lambda item: item[0], reverse=True)
        best_score, semantic_score, best_b = ranked[0]

        if best_score >= min_score:
            results.append(
                {
                    "item_id_A": a["item_id"],
                    "item_id_B": best_b["item_id"],
                    "method": "similarity",
                    "score": round(best_score, 4),
                    "semantic_score": round(semantic_score, 4),
                    "llm_used": False,
                    "product_a": row_to_product(a)["name"],
                    "product_b": row_to_product(best_b)["name"],
                }
            )

        if limit and len(results) >= limit:
            break

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Match BetterBasket Store A products to Store B products.")
    parser.add_argument("--store-a", default="data/products_A.csv")
    parser.add_argument("--store-b", default="data/products_B.csv")
    parser.add_argument("--output", default="outputs/matches.csv")
    parser.add_argument("--metadata-output", default="outputs/matches_with_metadata.csv")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-score", type=float, default=0.62)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-a", type=int, default=None)
    parser.add_argument("--sample-b", type=int, default=None)
    args = parser.parse_args()

    A = prepare_products(
        pd.read_csv(args.store_a, dtype=str, low_memory=False, nrows=args.sample_a),
        "A",
    ).reset_index(drop=True)
    B = prepare_products(
        pd.read_csv(args.store_b, dtype=str, low_memory=False, nrows=args.sample_b),
        "B",
    ).reset_index(drop=True)

    matches = build_matches(A, B, top_k=args.top_k, min_score=args.min_score, limit=args.limit)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if matches.empty:
        matches = pd.DataFrame(columns=["item_id_A", "item_id_B"])
    matches[["item_id_A", "item_id_B"]].to_csv(args.output, index=False)
    matches.to_csv(args.metadata_output, index=False)
    print(f"Generated {len(matches)} matches")


if __name__ == "__main__":
    main()
