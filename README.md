# BetterBasket Engineering Assessment

## Approach

The solution matches products from Store A to the closest product in Store B using a hybrid pipeline:

1. UPC based exact matching when UPCs are available.
2. Text normalization over product name, brand, category, and size.
3. Size/unit extraction from product names and size fields.
4. Similarity based candidate generation using fuzzy token matching.
5. GPT assisted judging for ambiguous candidates where rule based confidence is moderate.

## Why hybrid matching?

UPC matching alone is insufficient because scraped grocery data often has missing UPCs. Also, private label and fresh products may be equivalent from a customer perspective even when UPCs differ. Therefore, the solution combines deterministic matching with semantic validation.

## GPT usage

The provided GPT deployment is used only for ambiguous candidate pairs. The model receives structured product attributes for Product A and Product B and returns a JSON decision containing:

- is_match
- confidence
- reason

This avoids unnecessary LLM calls while still using the model for cases requiring semantic grocery judgment.

## Output

The main output is:

outputs/matches.csv

with columns:

item_id_A,item_id_B

A metadata file is also generated for debugging:

outputs/matches_with_metadata.csv

## How to run

pip install -r requirements.txt

python src/match_products.py

## Assumptions

- Same UPC means exact match.
- Products with very similar name, size, category, and brand are accepted directly.
- Private label products can match across brands if they represent the same product to a customer.
- Size, flavor, form, and organic status are important matching signals.
- GPT is used for borderline cases, not as the primary search mechanism.