# Steps to Integrate Bliss Symbols into LLaMA

This document outlines the process for integrating Bliss symbols into a LLaMA language model by adding a new token for each Bliss symbol. Each token is formatted as `[BLISS_{blissID}]`.

## Step 1: Add Single-Token Gloss Symbols

The working directory for this step is at `[job/bliss-gloss/add-single-token-gloss-symbols](../jobs/bliss-gloss/add-single-token-gloss-symbols)`.

The first step is to identify and add symbols whose glosses meet specific criteria. This is handled by the script `[add_single_token_gloss_symbols.py](../jobs/bliss-gloss/add-single-token-gloss-symbols/add_single_token_gloss_symbols.py)`.

### Criteria

A Bliss symbol is added if:

1. It has **only one gloss**.
2. That gloss is **a single token** according to the LLaMA tokenizer.

### Description

The script:

- Iterates through the Bliss dictionary.
- Checks if each symbol meets the above criteria.
- For qualifying symbols:
  - Adds a new token `[BLISS_{blissID}]`.
  - Copies the input and output embeddings from the original gloss token to the new token.

A total of **2,369 symbol tokens** are added during this step.

### Script Details

* Location: [`jobs/bliss-gloss/add-single-token-gloss-symbols/add_single_token_gloss_symbols.py`](../jobs/bliss-gloss/add-single-token-gloss-symbols/add_single_token_gloss_symbols.py)

* Usage:

```bash
python add_single_token_gloss_symbols.py {input_gloss_json} {output_file_with_added_id} {output_file_with_not_added_id}
```

* Example:

```bash
python ~/bliss_gloss/add_single_token_gloss_symbols.py ./data/bliss_gloss_cleaned.json ./outputs/bliss_ids_added.json ./outputs/bliss_ids_not_added.json
```

* Input:

**Bliss Dictionary JSON**: Located at `[jobs/bliss-gloss/add-single-token-gloss-symbols/data/bliss_gloss_cleaned.json](../jobs/bliss-gloss/add-single-token-gloss-symbols/data/bliss_gloss_cleaned.json)`

* Output:

After execution (on the Alliance server), the following are generated:

* **The output model** is saved in `/projects/ctb-whkchun/s2_bliss_LLMs/integrate_single_token_gloss_symbols`.
* **Added symbols list**: `jobs/bliss-gloss/add-single-token-gloss-symbols/output/bliss_ids_added.json`
* **Skipped Symbols List** (symbols not added): `jobs/bliss-gloss/add-single-token-gloss-symbols/output/bliss_ids_not_added.json`
