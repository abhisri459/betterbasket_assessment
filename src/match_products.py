import os
import re
import json
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz
from llm_judge import load_openai_client, judge_match


def normalize_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9\s\.]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_upc(x):
    if pd.isna(x):
        return ""
    x = str(x)
    x = re.sub(r"\D", "", x)
    return x.lstrip("0")


def extract_size(text):
    text = normalize_text(text)
    pattern = r"(\d+\.?\d*)\s?(oz|ounce|ounces|lb|lbs|pound|pounds|g|gram|grams|kg|ml|l|liter|liters|ct|count|pack)"
    m = re.search(pattern, text)
    if not m:
        return None, None

    value = float(m.group(1))
    unit = m.group(2)

    unit_map = {
        "ounce": "oz", "ounces": "oz",
        "pound": "lb", "pounds": "lb", "lbs": "lb",
        "gram": "g", "grams": "g",
        "liter": "l", "liters": "l",
        "count": "ct", "pack": "ct"
    }

    return value, unit_map.get(unit, unit)


def prepare(df):
    df = df.copy()

    # Change these column names if needed after inspecting your files
    df["item_id"] = df["item_id"].astype(str)
    df["name"] = df["name"].fillna("")
    df["brand"] = df["brand"].fillna("") if "brand" in df.columns else ""
    df["size"] = df["size"].fillna("") if "size" in df.columns else ""
    df["category"] = df["category"].fillna("") if "category" in df.columns else ""
    df["upc"] = df["upc"].fillna("") if "upc" in df.columns else ""

    df["name_norm"] = df["name"].apply(normalize_text)
    df["brand_norm"] = df["brand"].apply(normalize_text)
    df["category_norm"] = df["category"].apply(normalize_text)
    df["upc_norm"] = df["upc"].apply(normalize_upc)

    size_extracted = df.apply(
        lambda r: extract_size(str(r["name"]) + " " + str(r["size"])),
        axis=1
    )
    df["size_value"] = [x[0] for x in size_extracted]
    df["size_unit"] = [x[1] for x in size_extracted]

    return df


def score_pair(a, b):
    name_score = fuzz.token_set_ratio(a["name_norm"], b["name_norm"]) / 100
    brand_score = fuzz.token_set_ratio(a["brand_norm"], b["brand_norm"]) / 100

    category_score = fuzz.token_set_ratio(a["category_norm"], b["category_norm"]) / 100

    size_score = 0
    if pd.notna(a["size_value"]) and pd.notna(b["size_value"]) and a["size_unit"] == b["size_unit"]:
        diff = abs(a["size_value"] - b["size_value"])
        max_size = max(a["size_value"], b["size_value"])
        if max_size > 0:
            size_score = max(0, 1 - diff / max_size)

    score = (
        0.60 * name_score +
        0.15 * brand_score +
        0.15 * size_score +
        0.10 * category_score
    )

    return score


def build_upc_map(B):
    upc_map = {}
    for _, b in B.iterrows():
        upc = b["upc_norm"]
        if upc:
            upc_map.setdefault(upc, []).append(b)
    return upc_map


def parse_llm_json(text):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {
            "is_match": False,
            "confidence": 0,
            "reason": "Could not parse LLM response"
        }


def row_to_dict(row):
    return {
        "item_id": row.get("item_id", ""),
        "name": row.get("name", ""),
        "brand": row.get("brand", ""),
        "size": row.get("size", ""),
        "category": row.get("category", "")
    }


def main():
    A = pd.read_csv("data/products_A.csv")
    B = pd.read_csv("data/products_B.csv")

    A = prepare(A)
    B = prepare(B)

    client, model = load_openai_client("openai_creds.yaml")

    upc_map = build_upc_map(B)
    results = []
    matched_A = set()

    # 1. UPC exact matches
    for _, a in tqdm(A.iterrows(), total=len(A), desc="UPC matching"):
        upc = a["upc_norm"]
        if upc and upc in upc_map:
            b = upc_map[upc][0]
            results.append({
                "item_id_A": a["item_id"],
                "item_id_B": b["item_id"],
                "method": "upc",
                "score": 1.0,
                "llm_used": False,
                "reason": "UPC exact match"
            })
            matched_A.add(a["item_id"])

    # 2. Similarity + GPT-assisted matching
    for _, a in tqdm(A.iterrows(), total=len(A), desc="Similarity matching"):
        if a["item_id"] in matched_A:
            continue

        candidates = []
        for _, b in B.iterrows():
            s = score_pair(a, b)
            candidates.append((s, b))

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_b = candidates[0]

        if best_score >= 0.92:
            results.append({
                "item_id_A": a["item_id"],
                "item_id_B": best_b["item_id"],
                "method": "high_confidence_similarity",
                "score": best_score,
                "llm_used": False,
                "reason": "High fuzzy/name/size/category similarity"
            })
            matched_A.add(a["item_id"])

        elif best_score >= 0.55:
            llm_text = judge_match(
                client,
                model,
                row_to_dict(a),
                row_to_dict(best_b)
            )
            llm_result = parse_llm_json(llm_text)

            if llm_result["is_match"] and llm_result["confidence"] >= 0.70:
                results.append({
                    "item_id_A": a["item_id"],
                    "item_id_B": best_b["item_id"],
                    "method": "llm_judged_similarity",
                    "score": best_score,
                    "llm_used": True,
                    "reason": llm_result.get("reason", "")
                })
                matched_A.add(a["item_id"])

        if len(results) >= 4500:
            # Stop after comfortably crossing 4000 minimum.
            # Remove this if you want full coverage.
            break

    os.makedirs("outputs", exist_ok=True)

    output = pd.DataFrame(results)
    output[["item_id_A", "item_id_B"]].to_csv("outputs/matches.csv", index=False)

    # Helpful debug file
    output.to_csv("outputs/matches_with_metadata.csv", index=False)

    print(f"Generated {len(output)} matches")
    print("Saved outputs/matches.csv")


if __name__ == "__main__":
    main()