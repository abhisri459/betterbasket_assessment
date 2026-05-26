import json
import time

import yaml
from openai import OpenAI
from openai import RateLimitError


def load_openai_client(creds_path="openai_creds.yaml"):
    with open(creds_path, "r", encoding="utf-8") as file:
        creds = yaml.safe_load(file)

    config = creds.get("openai", creds)
    client = OpenAI(
        base_url=config["endpoint"],
        api_key=config["api_key"],
    )
    return client, config["deployment_name"]


def parse_json_response(text):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except (TypeError, json.JSONDecodeError):
        return {"selected_item_id_B": None, "confidence": 0.0, "reason": "Could not parse model response"}


def judge_candidates(client, model, product_a, candidates, retries=3, retry_delay=15):
    candidate_lines = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_lines.append(
            "\n".join(
                [
                    f"Candidate {index}",
                    f"item_id_B: {candidate.get('item_id')}",
                    f"name: {candidate.get('name')}",
                    f"brand: {candidate.get('brand')}",
                    f"size: {candidate.get('size')}",
                    f"category: {candidate.get('category')}",
                    f"is_private_label: {candidate.get('is_private_label')}",
                    f"is_organic: {candidate.get('is_organic')}",
                ]
            )
        )

    prompt = f"""
You are matching grocery products from Store A to Store B.

Choose the best Store B candidate for Product A, or return null if none are close enough.

Match when the customer would reasonably consider the products equivalent:
- Same national brand product when brand is meaningful.
- Private label or fresh products may match across retailers when type, form, size, and attributes are equivalent.
- Flavor, scent, variety, organic status, dietary variant, and package size matter.
- Do not match products that are merely in the same category.

Return only JSON:
{{
  "selected_item_id_B": "id string or null",
  "confidence": 0.0,
  "reason": "short reason"
}}

Product A:
item_id_A: {product_a.get("item_id")}
name: {product_a.get("name")}
brand: {product_a.get("brand")}
size: {product_a.get("size")}
category: {product_a.get("category")}
is_private_label: {product_a.get("is_private_label")}
is_organic: {product_a.get("is_organic")}

Store B candidates:
{chr(10).join(candidate_lines)}
"""

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise grocery product matching adjudicator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            return parse_json_response(response.choices[0].message.content)
        except RateLimitError:
            if attempt == retries:
                return {
                    "selected_item_id_B": None,
                    "confidence": 0.0,
                    "reason": "Skipped after repeated OpenAI rate limit responses",
                }
            time.sleep(retry_delay * (attempt + 1))
