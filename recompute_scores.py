import json
from bench.tasks_example import EXAMPLE_TASKS
from bench.scoring import exact_match_score

TASKS_BY_ID = {t.id: t for t in EXAMPLE_TASKS}

def main():
    path = "results/run_20260217_060342_gemma-2b-colab-t4.json"
    with open(path, "r") as f:
        data = json.load(f)

    corrected = 0
    total = 0

    for r in data["results"]:
        task = TASKS_BY_ID[r["task_id"]]
        new_score = exact_match_score(r["model_response"], task)
        if new_score != r["is_correct"]:
            print(f"{r['task_id']}: {r['is_correct']} -> {new_score}")
            r["is_correct"] = new_score
            corrected += 1
        if new_score is not None:
            total += 1

    print(f"\nUpdated {corrected} scores")
    print(f"Total auto-graded tasks: {total}")

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
