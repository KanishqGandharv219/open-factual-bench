from bench.tasks_example import EXAMPLE_TASKS
from bench.schema import BenchmarkConfig

def main():
    print("Loaded tasks:")
    for t in EXAMPLE_TASKS:
        print(f"- {t.id} | {t.domain} | {t.question} -> {t.reference_answer}")

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
