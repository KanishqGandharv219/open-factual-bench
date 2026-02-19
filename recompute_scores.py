"""
recompute_scores.py – Re-score existing result files with the current scorer.

Usage:
    python recompute_scores.py results/run_*.json
    python recompute_scores.py results/run_20260218_102915_google-gemma-2-2b-it.json
"""

import glob
import json
import sys
from bench.tasks_example import EXAMPLE_TASKS
from bench.scoring import score_task

TASKS_BY_ID = {t.id: t for t in EXAMPLE_TASKS}


def recompute(path: str):
    """Re-score a single result file and save if changes found."""
    with open(path, "r") as f:
        data = json.load(f)

    model_id = data.get("config", {}).get("model_id", "unknown")
    print(f"\n{'=' * 60}")
    print(f"  Re-scoring: {model_id}")
    print(f"  File: {path}")
    print(f"{'=' * 60}")

    corrected = 0
    correct_count = 0
    graded_count = 0

    for r in data["results"]:
        task = TASKS_BY_ID.get(r["task_id"])
        if not task:
            print(f"  WARNING: task {r['task_id']} not found in EXAMPLE_TASKS")
            continue

        new_score = score_task(r["model_response"], task)
        old_score = r["is_correct"]

        if new_score != old_score:
            print(f"  {r['task_id']}: {old_score} -> {new_score}")
            r["is_correct"] = new_score
            corrected += 1

        if new_score is not None:
            graded_count += 1
            if new_score:
                correct_count += 1

    # Update summary
    if "summary" in data:
        old_acc = data["summary"].get("accuracy", 0)
        new_acc = correct_count / graded_count if graded_count else 0
        data["summary"]["correct"] = correct_count
        data["summary"]["auto_graded"] = graded_count
        data["summary"]["accuracy"] = round(new_acc, 4)
        if old_acc != new_acc:
            print(f"  Accuracy: {old_acc:.1%} -> {new_acc:.1%}")

    print(f"  Updated {corrected} scores ({correct_count}/{graded_count} correct)")

    if corrected > 0:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {path}")
    else:
        print(f"  No changes — file not modified.")

    return corrected


def main():
    if len(sys.argv) < 2:
        # Default: all result files
        paths = sorted(glob.glob("results/run_*.json"))
    else:
        paths = sys.argv[1:]

    if not paths:
        print("No result files found.", file=sys.stderr)
        sys.exit(1)

    total_corrected = 0
    for path in paths:
        try:
            total_corrected += recompute(path)
        except Exception as e:
            print(f"  ERROR: {path}: {e}", file=sys.stderr)

    print(f"\n{'=' * 60}")
    print(f"  Total scores updated across all files: {total_corrected}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
