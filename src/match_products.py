import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from llm_judge import judge_candidates, load_openai_client
from utils import prepare_products, row_to_product


def load_llm_cache(path):
    cache = {}
    if not path or not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            cache[record["item_id_A"]] = record["decision"]
    return cache


def append_llm_cache(path, item_id, decision):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps({"item_id_A": item_id, "decision": decision}) + "\n")


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


def exact_field_score(a, b, field):
    value_a = a.get(field, "")
    value_b = b.get(field, "")
    if not value_a or not value_b:
        return 0.0
    return 1.0 if value_a == value_b else 0.0


def category_hierarchy_score(a, b):
    weighted_levels = [
        ("category_0_norm", 0.40),
        ("category_1_norm", 0.30),
        ("category_2_norm", 0.20),
        ("category_3_norm", 0.10),
    ]
    score = 0.0
    seen_weight = 0.0
    for field, weight in weighted_levels:
        value_a = a.get(field, "")
        value_b = b.get(field, "")
        if value_a and value_b:
            seen_weight += weight
            if value_a == value_b:
                score += weight
    return score / seen_weight if seen_weight else 0.0


def item_info_score(a, b):
    category_score = category_hierarchy_score(a, b)
    storage_score = exact_field_score(a, b, "storage_type_norm")
    packaging_score = exact_field_score(a, b, "packaging_description_norm")
    return 0.65 * category_score + 0.20 * storage_score + 0.15 * packaging_score


def score_pair(a, b, semantic_score):
    name_score = fuzz.token_set_ratio(a["name_norm"], b["name_norm"]) / 100
    category_score = fuzz.token_set_ratio(a["category_norm"], b["category_norm"]) / 100
    score = (
        0.36 * name_score
        + 0.22 * semantic_score
        + 0.15 * brand_score(a, b)
        + 0.12 * size_score(a, b)
        + 0.07 * category_score
        + 0.08 * item_info_score(a, b)
        - organic_penalty(a, b)
    )
    return max(0.0, min(1.0, score))


def generate_candidates(A, B, top_k):
    vectorizer = CountVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=120_000,
        stop_words="english",
    )
    b_counts = vectorizer.fit_transform(B["search_text"])
    a_counts = vectorizer.transform(A["search_text"])

    k1 = 1.5
    b = 0.75
    doc_lengths = np.asarray(b_counts.sum(axis=1)).ravel()
    avg_doc_length = doc_lengths.mean() if len(doc_lengths) else 1.0
    doc_freq = np.asarray((b_counts > 0).sum(axis=0)).ravel()
    idf = np.log((len(B) - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    bm25 = b_counts.tocoo(copy=True)
    length_norm = k1 * (1.0 - b + b * doc_lengths[bm25.row] / avg_doc_length)
    bm25.data = idf[bm25.col] * (bm25.data * (k1 + 1.0)) / (bm25.data + length_norm)
    bm25 = bm25.tocsr()

    a_binary = a_counts.copy()
    a_binary.data = np.ones_like(a_binary.data)
    scores = a_binary @ bm25.T

    all_indices = []
    all_distances = []
    for row_idx in range(scores.shape[0]):
        row = scores.getrow(row_idx)
        if row.nnz:
            order = np.argsort(row.data)[::-1][:top_k]
            indices = row.indices[order]
            row_scores = row.data[order]
            max_score = row_scores[0] if row_scores[0] > 0 else 1.0
            semantic_scores = row_scores / max_score
        else:
            indices = np.array([], dtype=int)
            semantic_scores = np.array([], dtype=float)

        if len(indices) < top_k:
            pad_count = top_k - len(indices)
            indices = np.concatenate([indices, np.zeros(pad_count, dtype=int)])
            semantic_scores = np.concatenate([semantic_scores, np.zeros(pad_count)])

        all_indices.append(indices[:top_k])
        all_distances.append(1.0 - semantic_scores[:top_k])

    return np.vstack(all_distances), np.vstack(all_indices)


def build_matches(
    A,
    B,
    top_k=30,
    min_score=0.62,
    limit=None,
    client=None,
    model=None,
    llm_min_score=0.50,
    llm_max_score=0.62,
    llm_candidates=3,
    max_llm_calls=0,
    llm_sleep=2.0,
    llm_cache_path="outputs/llm_cache.jsonl",
):
    distances, indices = generate_candidates(A, B, top_k)
    results = []
    llm_calls = 0
    llm_cache = load_llm_cache(llm_cache_path) if client else {}
    stats = {
        "similarity_matches": 0,
        "llm_candidates_seen": 0,
        "llm_calls_made": 0,
        "llm_cache_hits": 0,
        "llm_matches": 0,
        "llm_rejections": 0,
        "below_threshold": 0,
    }

    for a_pos, a in tqdm(list(A.iterrows()), desc="Matching products"):
        ranked = []
        for distance, b_pos in zip(distances[a_pos], indices[a_pos]):
            b = B.iloc[b_pos]
            semantic_score = 1.0 - float(distance)
            ranked.append((score_pair(a, b, semantic_score), semantic_score, b))

        ranked.sort(key=lambda item: item[0], reverse=True)
        best_score, semantic_score, best_b = ranked[0]

        if best_score >= min_score:
            stats["similarity_matches"] += 1
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
        elif (
            client
            and llm_min_score <= best_score < llm_max_score
            and (max_llm_calls <= 0 or llm_calls < max_llm_calls)
        ):
            stats["llm_candidates_seen"] += 1
            candidate_products = []
            candidate_by_id = {}
            for candidate_score, _, candidate_row in ranked[:llm_candidates]:
                product = row_to_product(candidate_row)
                product["candidate_score"] = round(candidate_score, 4)
                candidate_products.append(product)
                candidate_by_id[str(product["item_id"])] = (candidate_score, candidate_row)

            if a["item_id"] in llm_cache:
                decision = llm_cache[a["item_id"]]
                stats["llm_cache_hits"] += 1
            else:
                decision = judge_candidates(client, model, row_to_product(a), candidate_products)
                append_llm_cache(llm_cache_path, a["item_id"], decision)
                llm_calls += 1
                stats["llm_calls_made"] += 1
                if llm_sleep > 0:
                    time.sleep(llm_sleep)

            selected_id = decision.get("selected_item_id_B")
            confidence = float(decision.get("confidence") or 0)
            if selected_id and str(selected_id) in candidate_by_id and confidence >= 0.70:
                stats["llm_matches"] += 1
                selected_score, selected_b = candidate_by_id[str(selected_id)]
                results.append(
                    {
                        "item_id_A": a["item_id"],
                        "item_id_B": selected_b["item_id"],
                        "method": "llm_adjudicated",
                        "score": round(selected_score, 4),
                        "semantic_score": "",
                        "llm_used": True,
                        "product_a": row_to_product(a)["name"],
                        "product_b": row_to_product(selected_b)["name"],
                        "llm_confidence": confidence,
                        "llm_reason": decision.get("reason", ""),
                    }
                )
            else:
                stats["llm_rejections"] += 1
        else:
            stats["below_threshold"] += 1

        if limit and len(results) >= limit:
            break

    return pd.DataFrame(results), stats


def main():
    parser = argparse.ArgumentParser(description="Match BetterBasket Store A products to Store B products.")
    parser.add_argument("--store-a", default="data/products_A.csv")
    parser.add_argument("--store-b", default="data/products_B.csv")
    parser.add_argument("--output", default="outputs/matches.csv")
    parser.add_argument("--metadata-output", default="outputs/matches_with_metadata.csv")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-score", type=float, default=0.62)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--creds-path", default="openai_creds.yaml")
    parser.add_argument("--llm-min-score", type=float, default=0.50)
    parser.add_argument("--llm-max-score", type=float, default=0.62)
    parser.add_argument("--llm-candidates", type=int, default=3)
    parser.add_argument("--max-llm-calls", type=int, default=100)
    parser.add_argument("--llm-sleep", type=float, default=2.0)
    parser.add_argument("--llm-cache", default="outputs/llm_cache.jsonl")
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

    client = model = None
    if args.use_llm:
        client, model = load_openai_client(args.creds_path)

    matches, stats = build_matches(
        A,
        B,
        top_k=args.top_k,
        min_score=args.min_score,
        limit=args.limit,
        client=client,
        model=model,
        llm_min_score=args.llm_min_score,
        llm_max_score=args.llm_max_score,
        llm_candidates=args.llm_candidates,
        max_llm_calls=args.max_llm_calls,
        llm_sleep=args.llm_sleep,
        llm_cache_path=args.llm_cache,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if matches.empty:
        matches = pd.DataFrame(columns=["item_id_A", "item_id_B"])
    matches[["item_id_A", "item_id_B"]].to_csv(args.output, index=False)
    matches.to_csv(args.metadata_output, index=False)
    print(f"Generated {len(matches)} matches")
    print("Run summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
