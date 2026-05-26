import yaml
from openai import OpenAI

def load_openai_client(creds_path="openai_creds.yaml"):
    with open(creds_path, "r") as f:
        creds = yaml.safe_load(f)

    client = OpenAI(
        base_url=creds["ENDPOINT"],
        api_key=creds["API_KEY"]
    )

    return client, creds["DEPLOYMENT_NAME"]


def judge_match(client, model, product_a, product_b):
    prompt = f"""
You are helping match grocery products across two different retailers.

Decide whether Product B is a good match for Product A.

A good match means:
- Exact same national brand product, OR
- For private label/fresh products, a customer would consider them essentially the same product.
- Size, flavor, form, organic status, and product type should be very close.
- Do not match if flavor, size, product type, or dietary variant is clearly different.

Return ONLY valid JSON in this format:
{{
  "is_match": true or false,
  "confidence": number from 0 to 1,
  "reason": "short explanation"
}}

Product A:
Name: {product_a.get("name", "")}
Brand: {product_a.get("brand", "")}
Size: {product_a.get("size", "")}
Category: {product_a.get("category", "")}

Product B:
Name: {product_b.get("name", "")}
Brand: {product_b.get("brand", "")}
Size: {product_b.get("size", "")}
Category: {product_b.get("category", "")}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise grocery product matching judge."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content