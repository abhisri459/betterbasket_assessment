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

## Chosen Approach

I used a hybrid matching pipeline:

1. Normalize product names, brands, category hierarchy, tags, size fields, and
   structured metadata.
2. Parse useful assessment fields including `item_info`, `sizing_comp`,
   `brand_raw`, `is_private_label`, and `is_organic`.
3. Retrieve Store B candidates using TF-IDF cosine similarity over normalized
   product text.
4. Re-rank candidates using product-specific signals:
   - fuzzy name similarity
   - TF-IDF semantic/lexical similarity
   - brand compatibility
   - size and unit compatibility
   - category hierarchy from `item_info`
   - storage type and packaging description from `item_info`
   - organic/private-label signals
5. Optionally use the provided OpenAI deployment to adjudicate ambiguous
   candidate sets only.

The final submitted approach is TF-IDF retrieval plus deterministic metadata
re-ranking. This was the best balance of quality, speed, simplicity, and
reproducibility for the assessment.

## Why This Approach

TF-IDF works well here because grocery product matching is often lexical and
attribute-heavy. Exact tokens such as brand, flavor, count, size, form, and
category matter a lot. A purely semantic match can look reasonable while still
being wrong, for example matching a different flavor, pack size, battery type, or
product variant.

The structured `item_info` field is also important. It contains category levels,
storage type, and packaging description, so the matcher uses it both during
candidate retrieval and during final scoring.

## Alternatives Considered

I also tested two alternate retrieval branches:

- `bm25-retrieval`
- `sentence-transformer-retrieval`

BM25 was attractive because it is a strong search-style lexical retrieval method.
However, on this dataset it was much slower in the current implementation and
did not clearly improve match quality enough to justify replacing the simpler
TF-IDF baseline.

Sentence Transformers produced useful semantic candidates, but it was slower and
sometimes matched products that were semantically similar but operationally
different, such as different sizes, flavors, formats, or variants. That behavior
is risky for grocery matching, where exact product attributes matter.

For production, I would consider a stronger version of this pipeline:

```text
metadata blocking -> vector/BM25 retrieval -> deterministic re-ranking -> GPT adjudication for ambiguous cases
```

For this assessment, I kept the final submission conservative and reproducible.

## OpenAI Usage

The OpenAI deployment is intentionally not used for every product pair. Instead,
it is used as a quality-control layer for borderline candidates after
deterministic candidate generation.

This keeps the solution scalable and uses the model where semantic grocery
judgment matters most: private-label equivalence, missing brands, fresh items,
and close candidate ties.

LLM decisions are cached in:

```text
outputs/llm_cache.jsonl
```

so repeated runs do not waste API calls.

## Example Candidate Output

The utility below shows the top K Store B candidates for a single Store A item:

```bash
uv run --with pandas --with rapidfuzz --with scikit-learn --with tqdm --with pyyaml --with openai \
python src/visualize_top_matches.py \
  --row-index 0 \
  --top-k 20 \
  --output outputs/top_matches_row0.csv
```

Example Store A item:

```text
363563 | Great Value Corn on The Cob, 24 Count (Frozen)
```

Top 20 Store B candidates:

| Rank | item_id_B | item_name_B | Score |
|---:|---:|---|---:|
| 1 | 1091961 | Goya Mini Corn on the Cob | 0.3679 |
| 2 | 1086356 | Wegmans Fresh Corn Off The Cob | 0.2954 |
| 3 | 93848 | Green Giant Nibblers Corn on the Cob Mini Ears | 0.3020 |
| 4 | 954382 | Green Giant Corn-On-The-Cob, Extra Sweet, Mini | 0.2961 |
| 5 | 954456 | Wegmans Frozen Mini Spanakopita, 24 Count, FAMILY PACK | 0.3868 |
| 6 | 954457 | Wegmans Frozen Entertainment Collection, 24 Count, FAMILY PACK | 0.4066 |
| 7 | 95423 | Wegmans Frozen Mini Quiche Collection, 24 Count, FAMILY PACK | 0.4089 |
| 8 | 1086427 | Wegmans Roasted Butternut Squash Veggie | 0.2112 |
| 9 | 956307 | Wegmans Roasted Sweet Potato Halves, FAMILY PACK | 0.2047 |
| 10 | 2683635 | Pepsi Zero Sugar 12 Fl Oz 24 Count Can | 0.2656 |
| 11 | 92436 | Wegmans Part-Skim Mozzarella Cheese Sticks, 24 Count | 0.3549 |
| 12 | 1087565 | Wegmans Reduced Fat Part-Skim Mozzarella Cheese Sticks, 24 Count | 0.3424 |
| 13 | 1091165 | Touch of Color Cutlery, Premium, Classic Red, 24 Count | 0.3654 |
| 14 | 2019953 | Wegmans Diced Peaches Bowl, 24 COUNT, FAMILY PACK | 0.3481 |
| 15 | 2682267 | Tyson Game Time Chicken Nuggets Lightly Breaded Fully Cooked Frozen | 0.2146 |
| 16 | 94265 | Wegmans Gluten Free Chipotle Corn Veggie Cakes, 6 Count | 0.2820 |
| 17 | 103589 | Wegmans Cinnamon Applesauce Pouches, 24 COUNT, FAMILY PACK | 0.3567 |
| 18 | 105402 | Wegmans Peach Applesauce Pouches, 24 COUNT, FAMILY PACK | 0.3395 |
| 19 | 100143 | Wegmans Unsweetened Applesauce Pouches, 24 COUNT, FAMILY PACK | 0.3370 |
| 20 | 2685878 | Loopini Bianca Al Tartufo Pizza-Frozen | 0.2351 |

This example is intentionally useful for inspection: it shows the candidate list
for the first Store A item even though the scores are below the final acceptance
threshold. The final matcher only writes accepted matches to `outputs/matches.csv`.

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
