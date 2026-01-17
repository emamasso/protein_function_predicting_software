# build_go_parents.py
# Converts go-basic.obo -> go_parents.json

import json
from collections import defaultdict

OBO_PATH = "data/train/go-basic.obo"
OUT_PATH = "data/train/go_parents.json"

parents = defaultdict(set)

current_term = None

with open(OBO_PATH, "r") as f:
    for line in f:
        line = line.strip()

        if line == "[Term]":
            current_term = None

        elif line.startswith("id: GO:"):
            current_term = line.split("id: ")[1]

        elif line.startswith("is_a: GO:") and current_term is not None:
            parent = line.split("is_a: ")[1].split(" !")[0]
            parents[current_term].add(parent)

# convert sets → lists for json
parents = {k: sorted(list(v)) for k, v in parents.items()}

with open(OUT_PATH, "w") as f:
    json.dump(parents, f, indent=2)

print(f"Saved GO parent map to {OUT_PATH}")
print(f"Number of GO terms: {len(parents)}")
