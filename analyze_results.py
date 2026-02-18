"""
analyze_results.py – Analyze one or more benchmark result JSON files.

Usage:
    python analyze_results.py results/run_*.json
    python analyze_results.py results/run_20260218_044352_google-gemma-2-2b-it.json

Prints:
  - Per-model accuracy table
  - Per-domain accuracy breakdown
  - Hallucination stats (hallucinated vs refused)
"""

import argparse
import json
import sys
from collections import defaultdict


def load_run(path: str) -> dict:
    """Load a benchmark run JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def analyze_run(data: dict, filepath: str):
    """Print detailed analysis of a single run."""
    config = data.get("config", {})
    summary = data.get("summary", {})
    results = data.get("results", [])

    model_id = config.get("model_id", "unknown")
    hardware = config.get("hardware", "unknown")

    print(f"\n{'=' * 70}")
    print(f"  Model:    {model_id}")
    print(f"  Hardware: {hardware}")
    print(f"  File:     {filepath}")
    print(f"{'=' * 70}")

    # Overall accuracy
    total = summary.get("total_tasks", len(results))
    graded = summary.get("auto_graded", 0)
    correct = summary.get("correct", 0)
    skipped = summary.get("skipped", 0)
    accuracy = summary.get("accuracy", 0)

    print(f"\n  Overall: {accuracy:.1%}  ({correct}/{graded} auto-graded, {skipped} skipped)")

    # Per-domain breakdown
    domain_stats = defaultdict(lambda: {"correct": 0, "graded": 0, "total": 0})

    # We need task domain info — try to get it from task IDs
    from bench.tasks_example import EXAMPLE_TASKS
    task_map = {t.id: t for t in EXAMPLE_TASKS}

    for r in results:
        task_id = r.get("task_id", "")
        is_correct = r.get("is_correct")
        task = task_map.get(task_id)
        domain = task.domain if task else "unknown"
        sub_domain = ""
        if task and task.metadata:
            sub_domain = task.metadata.get("sub_domain", "")

        label = sub_domain if sub_domain else domain
        domain_stats[label]["total"] += 1
        if is_correct is not None:
            domain_stats[label]["graded"] += 1
            if is_correct:
                domain_stats[label]["correct"] += 1

    print(f"\n  {'Domain':<20} {'Correct':>8} {'Graded':>8} {'Accuracy':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10}")
    for domain in sorted(domain_stats.keys()):
        s = domain_stats[domain]
        acc = s["correct"] / s["graded"] if s["graded"] else 0
        print(f"  {domain:<20} {s['correct']:>8} {s['graded']:>8} {acc:>9.1%}")

    # Hallucination stats
    h_total = summary.get("hallucination_tasks", 0)
    h_hallucinated = summary.get("hallucinated", 0)
    h_refused = summary.get("refused", 0)
    h_unclear = summary.get("unclear", 0)

    if h_total > 0:
        print(f"\n  Hallucination Analysis ({h_total} tasks):")
        print(f"    Hallucinated:          {h_hallucinated}")
        print(f"    Refused/Corrected:     {h_refused}")
        print(f"    Unclear:               {h_unclear}")
    else:
        # Try to extract from per-result metadata
        h_labels = defaultdict(int)
        for r in results:
            meta = r.get("metadata") or {}
            label = meta.get("hallucination_label")
            if label:
                h_labels[label] += 1
        if h_labels:
            print(f"\n  Hallucination Analysis ({sum(h_labels.values())} tasks):")
            for label, count in sorted(h_labels.items()):
                print(f"    {label:<26} {count}")

    print()


def compare_models(runs: list):
    """Print a comparison table across multiple runs."""
    if len(runs) < 2:
        return

    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)

    print(f"\n  {'Model':<35} {'Hardware':<18} {'Accuracy':>10} {'Halluc.':>8} {'Refused':>8}")
    print(f"  {'-'*35} {'-'*18} {'-'*10} {'-'*8} {'-'*8}")

    for filepath, data in runs:
        config = data.get("config", {})
        summary = data.get("summary", {})
        model = config.get("model_id", "unknown")
        hw = config.get("hardware", "unknown")
        acc = summary.get("accuracy", 0)
        h = summary.get("hallucinated", 0)
        r = summary.get("refused", 0)

        # Truncate long model names
        if len(model) > 33:
            model = model[:30] + "..."
        if len(hw) > 16:
            hw = hw[:13] + "..."

        print(f"  {model:<35} {hw:<18} {acc:>9.1%} {h:>8} {r:>8}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze open-factual-bench result files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Path(s) to result JSON files",
    )
    args = parser.parse_args()

    runs = []
    for filepath in args.files:
        try:
            data = load_run(filepath)
            runs.append((filepath, data))
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"WARNING: Could not load {filepath}: {e}", file=sys.stderr)

    if not runs:
        print("No valid result files found.", file=sys.stderr)
        sys.exit(1)

    # Analyze each run
    for filepath, data in runs:
        analyze_run(data, filepath)

    # Compare if multiple
    compare_models(runs)


if __name__ == "__main__":
    main()
