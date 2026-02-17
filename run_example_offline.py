from dataclasses import asdict
from bench.tasks_example import EXAMPLE_TASKS
from bench.schema import BenchmarkConfig
from bench.results_schema import BenchmarkRun, TaskResult
from bench.scoring import exact_match_score
import json
import time
import os

def main():
    config = BenchmarkConfig(
        model_id="dummy-model",
        eval_type="factual_qa",
        max_new_tokens=0,
        temperature=0.0,
        hardware="offline-sim",
    )

    run = BenchmarkRun.create_new(config.__dict__)

    # Simulate predictions by just using the reference answers
    for task in EXAMPLE_TASKS:
        if task.id == "events_001":
            continue

        pred = task.reference_answer  # pretend model is perfect
        is_correct = exact_match_score(pred, task)

        result = TaskResult(
            task_id=task.id,
            model_response=pred,
            is_correct=is_correct,
            latency_ms=0.0,
            tokens_generated=0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        run.results.append(result)

    os.makedirs("results", exist_ok=True)
    output_path = f"results/{run.run_id}_offline.json"

    data = {
        "run_id": run.run_id,
        "config": run.config,
        "results": [asdict(r) for r in run.results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved offline run to {output_path}")
    correct = sum(1 for r in run.results if r.is_correct)
    print(f"Accuracy (simulated): {correct}/{len(run.results)}")

if __name__ == "__main__":
    main()
