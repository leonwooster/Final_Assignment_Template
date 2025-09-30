from datasets import load_dataset

ds = load_dataset("arterm-sedov/agent-course-final-assignment", "init")  # default config
rows = ds["train"][:200]  # grab 200 rows (dict of lists)

# Convert to a list of dicts (row-wise) for JSON export:
rowwise = [dict(zip(rows.keys(), values)) for values in zip(*rows.values())]

import json
with open("rows.json", "w", encoding="utf-8") as f:
    json.dump(rowwise, f, ensure_ascii=False, indent=2)
