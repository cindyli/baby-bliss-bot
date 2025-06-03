# python find_missing_symbols.py <symbols_in_model_json> <most_common_symbols_json>
# python find_missing_symbols.py ../1-add-single-token-gloss-symbols/output/bliss_ids_added.json ../2-find-frequently-used-symbols/output/overlapped_symbols.json ./output/missing_symbols.json

import json
import sys

if len(sys.argv) != 4:
    print("Usage: python find_missing_symbols.py <symbols_in_model_json> <most_common_symbols_json> <output_missing_symbols_json>")
    sys.exit(1)

symbols_in_model_json = sys.argv[1]
most_common_symbols_json = sys.argv[2]
output_missing_symbols_json = sys.argv[3]

with open(symbols_in_model_json, 'r') as f:
    symbols_in_model = json.load(f)
existing_symbols = set(map(int, symbols_in_model.keys()))

with open(most_common_symbols_json, 'r') as f:
    most_common_symbols = json.load(f)

# Find missing elements
missing_ids = [bci_av_id for bci_av_id in most_common_symbols if bci_av_id not in existing_symbols]

# Output the result
with open(output_missing_symbols_json, 'w') as f:
    json.dump(missing_ids, f, indent=2)
    print(f"{len(missing_ids)} missing symbol IDs saved to {output_missing_symbols_json}")
