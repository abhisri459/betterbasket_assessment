# BetterBasket Engineering Assessment

This project matches Store A grocery products to the closest Store B products and
produces the required submission file:

```text
outputs/matches.csv
```

with columns:

```csv
item_id_A,item_id_B
```

## Approach

The matcher uses a hybrid pipeline:

1. Normalize product names, brands, category hierarchy, tags, and size fields.
2. Parse structured assessment fields such as `item_info`, `sizing_comp`,
   `brand_raw`, `is_private_label`, and `is_organic`.
3. Generate Store B candidates with TF-IDF cosine similarity over normalized
   product text.
4. Re-rank candidates with fuzzy name/category similarity, brand compatibility,
   size compatibility, and organic/private-label signals.
5. Optionally use the provided OpenAI deployment to adjudicate ambiguous
   candidate sets only.

## OpenAI Usage

The OpenAI deployment is intentionally not used for every product pair. Instead,
it is used as a quality-control layer for borderline candidates after deterministic
candidate generation.

This keeps the solution scalable and uses the model where semantic grocery
judgment matters most: private-label equivalence, missing brands, fresh items,
and close candidate ties.

LLM decisions are cached in:

```text
outputs/llm_cache.jsonl
```

so repeated runs do not waste API calls.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Or run with `uv`:

```bash
uv run --with pandas --with rapidfuzz --with scikit-learn --with tqdm --with pyyaml --with openai python src/match_products.py --help
```

Place the provided CSV files in:

```text
data/products_A.csv
data/products_B.csv
```

Place the provided OpenAI credentials file at:

```text
openai_creds.yaml
```

The raw data, generated outputs, and credentials are ignored by Git.

## Run

Generate the baseline submission:

```bash
uv run --with pandas --with rapidfuzz --with scikit-learn --with tqdm --with pyyaml --with openai \
python src/match_products.py \
  --limit 4000 \
  --top-k 30 \
  --min-score 0.62 \
  --output outputs/matches.csv \
  --metadata-output outputs/matches_with_metadata.csv
```

Run with limited GPT adjudication:

```bash
uv run --with pandas --with rapidfuzz --with scikit-learn --with tqdm --with pyyaml --with openai \
python src/match_products.py \
  --limit 4000 \
  --top-k 30 \
  --min-score 0.68 \
  --use-llm \
  --llm-min-score 0.50 \
  --llm-max-score 0.68 \
  --max-llm-calls 50 \
  --llm-sleep 5 \
  --output outputs/matches.csv \
  --metadata-output outputs/matches_with_metadata.csv
```

## Outputs

- `outputs/matches.csv`: required two-column submission file.
- `outputs/matches_with_metadata.csv`: debug file with method, score, product
  names, and LLM metadata.
- `outputs/llm_cache.jsonl`: cached OpenAI adjudication decisions.

## Assumptions

- Products with highly similar names, sizes, brands, and categories can be
  accepted directly.
- Private-label items can match across retailers when product type, size, form,
  and key attributes are equivalent.
- Flavor, form, organic status, dietary variant, and package size are important
  matching signals.
- GPT is used only for ambiguous candidate adjudication, not as the primary
  search mechanism.
