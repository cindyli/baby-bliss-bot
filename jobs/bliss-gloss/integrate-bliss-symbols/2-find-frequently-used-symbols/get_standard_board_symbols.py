# python get_standard_board_symbols.py <initial_json_file> <output_symbols_file>
# python get_standard_board_symbols.py ../../../../adaptive-palette/public/palettes/bliss_standard_chart.json ./output/standard_board_symbols.json

import sys
import os
import json
from collections import deque

if len(sys.argv) != 3:
    print("Usage: python get_standard_board_symbols.py <initial_json_file> <output_symbols_file>")
    sys.exit(1)

initial_file = sys.argv[1]
output_file = sys.argv[2]

palette_directory = os.path.dirname(initial_file)

processed = set()
queued = set()
queue = deque()
bci_av_ids = set()

# Start with the initial file
if os.path.isfile(initial_file):
    queue.append(initial_file)
    queued.add(initial_file)
else:
    print(f"Error: Initial file {initial_file} not found")
    sys.exit(1)

while queue:
    file_path = queue.popleft()
    queued.remove(file_path)
    processed.add(file_path)
    print(f"processing {file_path}")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error processing {file_path}: {str(e)}")
        continue

    cells = data.get("cells", {})
    for cell_id, cell in cells.items():
        options = cell.get("options", {})
        branch_to = options.get("branchTo")
        bci_av_id = options.get("bciAvId")
        bci_av_id = (bci_av_id[0] if len(bci_av_id) == 1 else tuple(bci_av_id)) if isinstance(bci_av_id, list) else bci_av_id
        print(f"Processing cell {cell_id} with branchTo: {branch_to} and bciAvId: {bci_av_id}")

        if bci_av_id is not None:
            bci_av_ids.add(bci_av_id)
            print(f"Found bciAvId and pushed: {bci_av_id}")

        if branch_to is not None:
            # Process referenced file
            referenced_file = os.path.join(palette_directory, f"{branch_to}.json")
            if os.path.isfile(referenced_file):
                if referenced_file not in processed and referenced_file not in queued:
                    queue.append(referenced_file)
                    queued.add(referenced_file)
                    print(f"Queued referenced file: {referenced_file}")
            else:
                print(f"Warning: Referenced file {branch_to}.json not found")

# Convert to sorted list for consistent output
sorted_ids = sorted(bci_av_ids, key=lambda x: str(x))

final = [list(item) if isinstance(item, tuple) else item for item in sorted_ids]
print("Unique bciAvId values:")
for item in sorted_ids:
    print(f" - {item}")
print(f"\nTotal count of symbols: {len(sorted_ids)}")

with open(output_file, 'w') as f:
    json.dump(final, f, indent=2)
