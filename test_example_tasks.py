from bench.tasks_example import EXAMPLE_TASKS
from bench.schema import BenchmarkConfig

def main():
    print(f"Loaded {len(EXAMPLE_TASKS)} tasks:")
    for t in EXAMPLE_TASKS:
        task_type = t.metadata.get("type", "unknown") if t.metadata else "unknown"
        print(f"- {t.id} | {t.domain} | {task_type} | {t.question[:50]}...")

    cfg = BenchmarkConfig(
        model_id="google/gemma-2-2b-it",
        eval_type="factual_qa",
        max_new_tokens=64,
        temperature=0.0,
        hardware="T4 Colab",
    )
    print("\nExample config:")
    print(cfg)

if __name__ == "__main__":
    main()
