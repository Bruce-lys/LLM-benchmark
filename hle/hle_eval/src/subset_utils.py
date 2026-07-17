import json
import random
from collections import Counter, defaultdict

FIELD_ALIASES = {
    "difficulty": "answer_type",
}


def resolve_stratify_fields(stratify_by):
    if isinstance(stratify_by, str):
        fields = [field.strip() for field in stratify_by.split(",") if field.strip()]
    else:
        fields = list(stratify_by)
    if not fields:
        raise ValueError("At least one stratification field is required.")
    return fields


def resolve_row_field(row, field):
    source_field = FIELD_ALIASES.get(field, field)
    if source_field not in row:
        raise KeyError(f"Missing stratification field '{field}' (resolved to '{source_field}').")
    return row[source_field]


def stratify_key(row, stratify_fields):
    return tuple(resolve_row_field(row, field) for field in stratify_fields)


def load_subset_file(path):
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return set(data)
    if isinstance(data, dict) and "ids" in data:
        return set(data["ids"])
    raise ValueError(f"Subset file must be a JSON list of ids or an object with an 'ids' field: {path}")


def filter_questions_by_subset(questions, subset_ids):
    return [q for q in questions if q["id"] in subset_ids]


def create_stratified_subset(rows, ratio, seed, stratify_by="category", min_per_group=1):
    stratify_fields = resolve_stratify_fields(stratify_by)
    by_group = defaultdict(list)
    for row in rows:
        by_group[stratify_key(row, stratify_fields)].append(row)

    rng = random.Random(seed)
    selected = []
    for group in sorted(by_group.keys()):
        items = list(by_group[group])
        rng.shuffle(items)
        k = max(min_per_group, round(len(items) * ratio))
        k = min(k, len(items))
        selected.extend(items[:k])

    selected.sort(key=lambda row: row["id"])
    return selected


def summarize_subset(rows, stratify_by="category"):
    stratify_fields = resolve_stratify_fields(stratify_by)
    if len(stratify_fields) == 1:
        field = stratify_fields[0]
        return Counter(resolve_row_field(row, field) for row in rows)

    return Counter(stratify_key(row, stratify_fields) for row in rows)


def summarize_subset_breakdown(rows):
    return {
        "category_counts": Counter(row["category"] for row in rows),
        "difficulty_counts": Counter(row.get("difficulty", row.get("answer_type")) for row in rows),
        "stratum_counts": Counter((row["category"], row.get("difficulty", row.get("answer_type"))) for row in rows),
    }


def save_subset_file(path, ids, metadata):
    payload = {
        **metadata,
        "total_ids": len(ids),
        "ids": ids,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)
    return payload
