import argparse
import os

from datasets import Dataset, load_dataset

from subset_utils import (
    create_stratified_subset,
    resolve_stratify_fields,
    save_subset_file,
    summarize_subset_breakdown,
)


def load_rows(args):
    if args.dataset_path:
        dataset = Dataset.from_file(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset, split="test")

    rows = []
    for batch in dataset._data.to_batches(max_chunksize=500):
        for i in range(batch.num_rows):
            image = batch["image"][i].as_py() if "image" in batch.column_names else ""
            answer_type = batch["answer_type"][i].as_py() if "answer_type" in batch.column_names else ""
            rows.append(
                {
                    "id": batch["id"][i].as_py(),
                    "category": batch["category"][i].as_py(),
                    "answer_type": answer_type,
                    "difficulty": answer_type,
                    "image": image,
                }
            )
    return rows


def format_stratum_counts(stratum_counts):
    lines = []
    for (category, difficulty), count in sorted(stratum_counts.items()):
        lines.append(f"  {category} | {difficulty}: {count}")
    return lines


def main(args):
    rows = load_rows(args)

    if args.text_only:
        rows = [row for row in rows if not row["image"]]

    stratify_fields = resolve_stratify_fields(args.stratify_by)
    selected = create_stratified_subset(
        rows,
        ratio=args.ratio,
        seed=args.seed,
        stratify_by=stratify_fields,
        min_per_group=args.min_per_group,
    )
    ids = [row["id"] for row in selected]
    breakdown = summarize_subset_breakdown(selected)

    metadata = {
        "dataset": args.dataset,
        "description": (
            f"{args.ratio:.0%} stratified subset by {', '.join(stratify_fields)}, "
            f"seed={args.seed}, min_per_group={args.min_per_group}. "
            "difficulty maps to dataset answer_type (exactMatch/multipleChoice)."
        ),
        "seed": args.seed,
        "ratio": args.ratio,
        "stratify_by": stratify_fields,
        "min_per_group": args.min_per_group,
        "text_only": args.text_only,
        "category_counts": dict(breakdown["category_counts"]),
        "difficulty_counts": dict(breakdown["difficulty_counts"]),
        "stratum_counts": {
            f"{category} | {difficulty}": count
            for (category, difficulty), count in sorted(breakdown["stratum_counts"].items())
        },
    }

    output_path = args.output
    if output_path is None:
        suffix = "text" if args.text_only else "all"
        field_tag = "_".join(stratify_fields)
        output_path = f"hle_subset_{int(args.ratio * 100)}pct_{field_tag}_{suffix}.json"

    payload = save_subset_file(output_path, ids, metadata)

    print(f"Wrote {payload['total_ids']} ids to {output_path}")
    print("Category counts:")
    for category, count in sorted(metadata["category_counts"].items()):
        print(f"  {category}: {count}")
    print("Difficulty counts:")
    for difficulty, count in sorted(metadata["difficulty_counts"].items()):
        print(f"  {difficulty}: {count}")
    print("Category | difficulty strata:")
    print("\n".join(format_stratum_counts(breakdown["stratum_counts"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a reproducible stratified HLE subset.")
    parser.add_argument("--dataset", type=str, default="cais/hle", help="Hugging Face dataset id.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional local Arrow file path. Overrides --dataset when set.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path for subset ids.")
    parser.add_argument("--ratio", type=float, default=0.1, help="Fraction of each stratum to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--stratify_by",
        type=str,
        default="category,difficulty",
        help="Comma-separated fields for stratification. Use 'difficulty' for answer_type.",
    )
    parser.add_argument("--min_per_group", type=int, default=1, help="Minimum samples per stratum.")
    parser.add_argument("--text_only", action="store_true", help="Sample only from text-only questions.")
    args = parser.parse_args()
    main(args)
