# python compare_most_frequent_symbols.py <output_overlapped_symbols_json> <output_only_in_standard_board_json> <output_only_in_tag1_json>
# python compare_most_frequent_symbols.py ./output/overlapped_symbols.json ./output/only_in_standard_board.json ./output/only_in_tag1.json

import json
import sys

if len(sys.argv) != 4:
    print("Usage: python compare_most_frequent_symbols.py <output_overlapped_symbols_json> <output_only_in_standard_board_json> <output_only_in_tag1_json>")
    sys.exit(1)

output_overlapped_symbols_json = sys.argv[1]
output_only_in_standard_board_json = sys.argv[2]
output_only_in_tag1_json = sys.argv[3]

standard_board_symbols_json = "./output/standard_board_symbols.json"
frequency_tagged_symbols_json = "./output/frequency_tagged_symbols.json"

standard_board_symbols = json.load(open(standard_board_symbols_json, "r"))
frequency_tagged_symbols = json.load(open(frequency_tagged_symbols_json, "r"))["1"]

overlapping = [x for x in standard_board_symbols if x in frequency_tagged_symbols]

# Elements present in A but not in B
only_in_standard_board = [x for x in standard_board_symbols if x not in frequency_tagged_symbols]

# Elements present in B but not in A (including lists)
only_in_tag1 = [x for x in frequency_tagged_symbols if x not in standard_board_symbols]

print(f"Standard board symbols: {len(standard_board_symbols)}")
print(f"Frequency tagged symbols: {len(frequency_tagged_symbols)}")

# Write the result to JSON
with open(output_overlapped_symbols_json, "w") as jsonfile:
    json.dump(overlapping, jsonfile, indent=2)
    print(f"{len(overlapping)} Overlapping symbols written to {output_overlapped_symbols_json}")

with open(output_only_in_standard_board_json, "w") as jsonfile:
    json.dump(only_in_standard_board, jsonfile, indent=2)
    print(f"{len(only_in_standard_board)} Only in standard board symbols written to {output_only_in_standard_board_json}")

with open(output_only_in_tag1_json, "w") as jsonfile:
    json.dump(only_in_tag1, jsonfile, indent=2)
    print(f"{len(only_in_tag1)} Only in tag1 symbols written to {output_only_in_tag1_json}")
