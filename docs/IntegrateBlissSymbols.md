# Steps to Integrate Bliss Symbols into LLaMA

This document outlines the process for integrating Bliss symbols into a LLaMA language model by adding a new token for each Bliss symbol. Each token is formatted as `[BLISS_{blissID}]`.

The working directory for all steps is: [`job/bliss-gloss/integrate-bliss-symbols`](../jobs/bliss-gloss/integrate-bliss-symbols).

## Step 1: Add Single-Token Gloss Symbols

**Working directory**: [`job/bliss-gloss/integrate-bliss-symbols/1-add-single-token-gloss-symbols`](../jobs/bliss-gloss/integrate-bliss-symbols/1-add-single-token-gloss-symbols)

This step identifies and adds symbols that meet specific criteria.

### Criteria

A Bliss symbol is added if:

1. It has **only one gloss**.
2. That gloss is **a single token** according to the LLaMA tokenizer.

### Description

The script:

- Iterates through the Bliss dictionary.
- Checks if each symbol meets the above criteria.
- For qualifying symbols:
  - Adds a new token in the format of `[BLISS_{blissID}]`.
  - Copies the input and output embeddings from the original gloss token to the new token.

**Total tokens added in this step:** 2369

### Script Details

* Script: [`add_single_token_gloss_symbols.py`](../jobs/bliss-gloss/integrate-bliss-symbols/1-add-single-token-gloss-symbols/add_single_token_gloss_symbols.py)

* Usage:

```bash
python add_single_token_gloss_symbols.py {input_gloss_json} {output_file_with_added_id} {output_file_with_not_added_id}
```

* Example:

```bash
python ~/bliss_gloss/add_single_token_gloss_symbols.py ./data/bliss_gloss_cleaned.json ./outputs/bliss_ids_added.json ./outputs/bliss_ids_not_added.json
```

* Input:

**Bliss Dictionary JSON**: [`bliss_gloss_cleaned.json`](../jobs/bliss-gloss/integrate-bliss-symbols/1-add-single-token-gloss-symbols/data/bliss_gloss_cleaned.json)

* Output:

* **The output model** is saved in `/projects/ctb-whkchun/s2_bliss_LLMs/integrate_single_token_gloss_symbols`.
* **Added symbols list**: [`output/bliss_ids_added.json`](../jobs/bliss-gloss/integrate-bliss-symbols/1-add-single-token-gloss-symbols/output/bliss_ids_added.json)
* **Skipped Symbols List** (symbols not added): [`/output/bliss_ids_not_added.json`](../jobs/bliss-gloss/integrate-bliss-symbols/1-add-single-token-gloss-symbols/output/bliss_ids_not_added.json)

## Step 2: Find most frequently used symbols

**Working directoy** [`jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/)

Two methods were used to identify frequently used symbols:

1. The Bliss standard board

  * Symbols in the board are extracted from [`standard_board_symbols.json`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/output/standard_board_symbols.json).
  * Output file: [`output/standard_board_symbols.json`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/output/standard_board_symbols.json)
  * A total of **479 symbols** are extracted: 
    * **323 symbols** are single symbols.
    * **156 symbols** are Composite symbols

2. Frequency-Tagged Spreadsheet (by Mats Lundälv)

Mats Lundälv, a former chairman of BCI, sorted the Bliss words into groups for frequency analysis. Group 1 is higher frequency, Group 5 is lower. Ungrouped are even lower. 

  * [The spreadsheet](https://docs.google.com/spreadsheets/d/1E4H45LYffyWKT0Devp32wfB1NOPBnPEp/edit?usp=drive_link&ouid=111581944030695000923&rtpof=true&sd=true).
  * Symbols are grouped by its tag number and reported in [this output file](`jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/output/frequency_tagged_symbols.json`).
  * Group counts:
    * Group 1: **644 symbols**
    * Group 2: **127 symbols**
    * Group 3: **141 symbols**
    * Group 4: **95 symbols**
    * Group 5: **13 symbols**
    * Ungrouped: **4815 symbols**

### Overlap Analysis

Based on the above sources:
* **288 symbols** appear in both the standard board and frequency group 1
* **191 symbols** are only in the standard board
* **356 symbols** are only in frequency group 1

So, the primary focus is the **288 overlapping symbols**.

### Script Details

#### Extract symbols from Bliss standard board

* Script: [`get_standard_board_symbols.py`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/get_standard_board_symbols.py)

* Usage: 

```bash
python get_standard_board_symbols.py <initial_json_file> <output_symbols_file>
```

* Example:

```bash
python get_standard_board_symbols.py ../../../../adaptive-palette/public/palettes/bliss_standard_chart.json ./output/standard_board_symbols.json
```

* Output: [`output/standard_board_symbols.json`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/output/standard_board_symbols.json)

#### Extract symbols by their frequency group

* Script: [`get_frequency_tagged_symbols.py`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/get_frequency_tagged_symbols.py)

* Usage:

```bash
python get_frequency_tagged_symbols.py <frequency_tagged_symbols_csv> <output_tagged_symbols_json>
```

* Example:

```bash
python get_frequency_tagged_symbols.py ./data/Bliss_frequency_tags.csv ./output/frequency_tagged_symbols.json
```
* Output: [`output/frequency_tagged_symbols.json`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/output/frequency_tagged_symbols.json)

#### Compare symbols in the standard board with symbols in the frequency group 1

* Script: [`compare_most_frequent_symbols.py`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/compare_most_frequent_symbols.py)

* Usage:

```bash
python get_frequency_tagged_symbols.py <frequency_tagged_symbols_csv> <output_tagged_symbols_json>
```

* Example:

```bash
python get_frequency_tagged_symbols.py ./data/Bliss_frequency_tags.csv ./output/frequency_tagged_symbols.json
```

* Output file:
  * [`output/overlapped_symbols.json`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/output/overlapped_symbols.json): In both standard board and frequency group 1
  * [`output/only_in_standard_board.json`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/output/only_in_standard_board.json): Only in standard board
  * [`output/only_in_tag1.json`](../jobs/bliss-gloss/integrate-bliss-symbols/2-find-frequently-used-symbols/output/only_in_tag1.json): Only in frequency group 1

## Step 3: Identify Missing High-Frequency Symbols

**Working directoy**: [`jobs/bliss-gloss/integrate-bliss-symbols/3-find-missing-symbols/`](../jobs/bliss-gloss/integrate-bliss-symbols/3-find-missing-symbols/).

This step compares **288 most frequently used symbols** from Step 2 aginst the list of symbols already been added in the step 1 and finds high-priority Bliss symbols that were not added during Step 1.

**160 symbols** were found to be missing. 

They are written to [`output/missing_symbols.json`](../jobs/bliss-gloss/integrate-bliss-symbols/3-find-missing-symbols/output/missing_symbols.json).

### Script Details

* Script: [`find_missing_symbols.py`](../jobs/bliss-gloss/integrate-bliss-symbols/3-find-missing-symbols/find_missing_symbols.py)

* Usage:

```bash
python find_missing_symbols.py <symbols_in_model_json> <most_common_symbols_json>
```

* Example:
```bash
python find_missing_symbols.py ./data/Bliss_frequency_tags.csv ./output/frequency_tagged_symbols.json
```

* Output file:
  * [`output/missing_symbols.json`](../jobs/bliss-gloss/integrate-bliss-symbols/3-find-missing-symbols/output/missing_symbols.json): 160 missing symbols
