from collections import defaultdict

def load_go_parents(obo_path):
    """
    Load GO parents from go-basic.obo
    Returns: dict GO_ID -> set(parent_GO_IDs)
    """
    parents = defaultdict(set)
    current_go = None

    with open(obo_path, "r") as f:
        for line in f:
            line = line.strip()

            if line == "[Term]":
                current_go = None

            elif line.startswith("id: GO:"):
                current_go = line.split("id: ")[1]

            elif current_go and line.startswith("is_a: GO:"):
                parent = line.split("is_a: GO:")[1].split()[0]
                parents[current_go].add(parent)

            elif current_go and line.startswith("relationship: part_of GO:"):
                parent = line.split("relationship: part_of GO:")[1].split()[0]
                parents[current_go].add(parent)

    return parents


def propagate_terms(pred_dict, go_parents):
    """
    pred_dict: dict {GO_ID: score}
    go_parents: dict {GO_ID: set(parent_GO_IDs)}

    Returns: dict with propagated terms
    """
    propagated = dict(pred_dict)
    stack = list(pred_dict.items())

    while stack:
        go, score = stack.pop()
        for parent in go_parents.get(go, []):
            if parent not in propagated or propagated[parent] < score:
                propagated[parent] = score
                stack.append((parent, score))

    return propagated