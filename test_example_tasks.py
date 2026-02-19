from bench.tasks_example import EXAMPLE_TASKS
from bench.schema import BenchmarkConfig, Task
from bench.scoring import (
    _normalize_numbers, _numeric_match, score_task
)


def test_task_loading():
    """Verify all tasks load with correct structure."""
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
    print("[PASS] Task loading\n")


def test_normalize_numbers():
    """Test number-word to digit conversion."""
    cases = [
        ("eight", "8"),
        ("eight planets", "8 planets"),
        ("twenty", "20"),
        ("the answer is eight", "the answer is 8"),
        ("one hundred", "1 100"),        # no compound handling — expected
        ("no numbers here", "no numbers here"),
        ("", ""),
    ]
    passed = 0
    for inp, expected in cases:
        result = _normalize_numbers(inp)
        ok = result == expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  [{status}] _normalize_numbers({inp!r}) = {result!r}, expected {expected!r}")
        passed += ok
    print(f"[{'PASS' if passed == len(cases) else 'FAIL'}] Number-word normalization ({passed}/{len(cases)})")


def test_numeric_match():
    """Test numeric tolerance matching."""
    cases = [
        ("3.14", "3.1416", True),     # within default epsilon 0.01
        ("3.14", "3.15", True),       # diff = 0.01, just at boundary
        ("3.14", "3.20", False),      # diff = 0.06, too far
        ("1,024", "1024", True),      # comma handling
        ("42", "42", True),           # exact integer
        ("abc", "123", False),        # non-numeric
        ("", "5", False),             # empty
    ]
    passed = 0
    for pred, ref, expected in cases:
        result = _numeric_match(pred, ref)
        ok = result == expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  [{status}] _numeric_match({pred!r}, {ref!r}) = {result!r}, expected {expected!r}")
        passed += ok
    print(f"[{'PASS' if passed == len(cases) else 'FAIL'}] Numeric tolerance ({passed}/{len(cases)})")


def test_score_task_number_words():
    """Integration test: score_task with number-word answers."""
    # Simulate a task where reference is "8" and model answers "Eight"
    task = Task(
        id="test_num",
        question="How many planets?",
        reference_answer="8",
        domain="science",
    )
    cases = [
        ("Eight", True),                          # number word
        ("eight planets", True),                   # number word in context
        ("8", True),                               # exact match
        ("There are eight planets", True),         # substring
        ("There are 9 planets", False),            # wrong answer
    ]
    passed = 0
    for pred, expected in cases:
        result = score_task(pred, task)
        ok = result == expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  [{status}] score_task({pred!r}, ref='8') = {result!r}, expected {expected!r}")
        passed += ok
    print(f"[{'PASS' if passed == len(cases) else 'FAIL'}] score_task number-word integration ({passed}/{len(cases)})")


def test_score_task_numeric_tolerance():
    """Integration test: score_task with numeric tolerance."""
    task = Task(
        id="test_tol",
        question="What is pi?",
        reference_answer="3.1416",
        domain="math",
    )
    cases = [
        ("3.14", True),         # close enough
        ("3.1416", True),       # exact
        ("3.2", False),         # too far
        ("pi", False),          # non-numeric
    ]
    passed = 0
    for pred, expected in cases:
        result = score_task(pred, task)
        ok = result == expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  [{status}] score_task({pred!r}, ref='3.1416') = {result!r}, expected {expected!r}")
        passed += ok
    print(f"[{'PASS' if passed == len(cases) else 'FAIL'}] score_task numeric tolerance ({passed}/{len(cases)})")


def main():
    print("=" * 60)
    print("  open-factual-bench — Test Suite")
    print("=" * 60 + "\n")

    test_task_loading()
    test_normalize_numbers()
    test_numeric_match()
    test_score_task_number_words()
    test_score_task_numeric_tolerance()

    print("\n" + "=" * 60)
    print("  All tests complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
