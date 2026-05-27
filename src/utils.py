import ast
import json
import re

import pandas as pd


UNIT_MAP = {
    "ounce": "oz",
    "ounces": "oz",
    "oz": "oz",
    "fl oz": "fl oz",
    "fl. oz": "fl oz",
    "fluid ounce": "fl oz",
    "fluid ounces": "fl oz",
    "pound": "lb",
    "pounds": "lb",
    "lbs": "lb",
    "lb": "lb",
    "gram": "g",
    "grams": "g",
    "g": "g",
    "kilogram": "kg",
    "kilograms": "kg",
    "kg": "kg",
    "milliliter": "ml",
    "milliliters": "ml",
    "ml": "ml",
    "liter": "l",
    "liters": "l",
    "l": "l",
    "count": "ct",
    "ct": "ct",
    "pack": "ct",
}


def normalize_text(value):
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[^a-z0-9\s\.]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_bool(value):
    if pd.isna(value) or value == "":
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_mapping(value):
    if pd.isna(value) or value == "":
        return {}
    if isinstance(value, dict):
        return value
    text = str(value)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}


def flatten_mapping(mapping, prefix=""):
    parts = []
    for key, value in mapping.items():
        if isinstance(value, dict):
            parts.extend(flatten_mapping(value, key))
        elif value not in (None, "", [], {}):
            label = f"{prefix}_{key}" if prefix else key
            parts.append(f"{label} {value}")
    return parts


def category_text(row):
    item_info = parse_mapping(row.get("item_info", ""))
    values = [
        row.get("category", ""),
        row.get("department", ""),
        row.get("subcategory", ""),
        item_info.get("category_0", ""),
        item_info.get("category_1", ""),
        item_info.get("category_2", ""),
        item_info.get("category_3", ""),
    ]
    return normalize_text(" ".join(str(v) for v in values if v))


def item_info_value(row, key):
    item_info = parse_mapping(row.get("item_info", ""))
    return normalize_text(item_info.get(key, ""))


def friendly_size(row):
    item_info = parse_mapping(row.get("sizing_comp", ""))
    if item_info.get("size_user_friendly"):
        return str(item_info["size_user_friendly"])
    if row.get("size_raw"):
        return str(row["size_raw"])
    return ""


def extract_size(text):
    text = normalize_text(text)
    pattern = (
        r"(\d+(?:\.\d+)?)\s?"
        r"(fl\.?\s?oz|fluid ounces?|oz|ounces?|lbs?|pounds?|kg|kilograms?|g|grams?|"
        r"ml|milliliters?|l|liters?|ct|count|pack)"
    )
    match = re.search(pattern, text)
    if not match:
        return None, ""
    value = float(match.group(1))
    unit = re.sub(r"\s+", " ", match.group(2).replace(".", " ")).strip()
    return value, UNIT_MAP.get(unit, unit)


def prepare_products(df, store):
    df = df.copy()
    df["store"] = store
    df["item_id"] = df["item_id"].astype(str)
    df["name"] = df["name"].fillna("")
    for column in ["brand_raw", "name_clean", "description", "tags", "is_private_label", "is_organic"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("")
    df["size_text"] = df.apply(friendly_size, axis=1)

    df["name_norm"] = df.apply(
        lambda row: normalize_text(row["name_clean"] or row["name"]),
        axis=1,
    )
    df["brand_norm"] = df["brand_raw"].apply(normalize_text)
    df["category_norm"] = df.apply(category_text, axis=1)
    df["category_0_norm"] = df.apply(lambda row: item_info_value(row, "category_0"), axis=1)
    df["category_1_norm"] = df.apply(lambda row: item_info_value(row, "category_1"), axis=1)
    df["category_2_norm"] = df.apply(lambda row: item_info_value(row, "category_2"), axis=1)
    df["category_3_norm"] = df.apply(lambda row: item_info_value(row, "category_3"), axis=1)
    df["storage_type_norm"] = df.apply(lambda row: item_info_value(row, "storage_type"), axis=1)
    df["packaging_description_norm"] = df.apply(
        lambda row: item_info_value(row, "packaging_description"),
        axis=1,
    )
    df["is_private_label_norm"] = df.get("is_private_label", False).apply(normalize_bool)
    df["is_organic_norm"] = df.get("is_organic", False).apply(normalize_bool)

    size_values = df.apply(
        lambda row: extract_size(f"{row['name']} {row['size_text']} {row.get('sizing_comp', '')}"),
        axis=1,
    )
    df["size_value"] = [value for value, _ in size_values]
    df["size_unit"] = [unit for _, unit in size_values]

    df["search_text"] = df.apply(
        lambda row: " ".join(
            part
            for part in [
                row["name_norm"],
                row["brand_norm"],
                row["category_norm"],
                row["storage_type_norm"],
                row["packaging_description_norm"],
                normalize_text(row["tags"]),
                normalize_text(row["size_text"]),
            ]
            if part
        ),
        axis=1,
    )
    return df


def row_to_product(row):
    return {
        "item_id": row.get("item_id", ""),
        "name": row.get("name", ""),
        "brand": row.get("brand_raw", ""),
        "size": row.get("size_text", ""),
        "category": row.get("category_norm", ""),
        "storage_type": row.get("storage_type_norm", ""),
        "packaging": row.get("packaging_description_norm", ""),
        "is_private_label": bool(row.get("is_private_label_norm", False)),
        "is_organic": bool(row.get("is_organic_norm", False)),
    }
