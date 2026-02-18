"""
run_benchmark.py – Evaluate a Hugging Face model on open-factual-bench tasks.

Usage (GPU):
    python run_benchmark.py --model_id google/gemma-2-2b-it --hardware "Colab T4"

Usage (CPU, slow but works):
    python run_benchmark.py --model_id google/gemma-2-2b-it --device cpu --hardware "CPU laptop"

Offline/dry-run (no model, uses reference answers as predictions):
    python run_benchmark.py --offline --hardware "offline-sim"
"""

import argparse
import json
import os
import time
from dataclasses import asdict

from bench.tasks_example import EXAMPLE_TASKS
from bench.schema import BenchmarkConfig
from bench.results_schema import BenchmarkRun, TaskResult
from bench.scoring import score_task, label_hallucination_task


# ── Helpers ────────────────────────────────────────────────────────────


def format_prompt(question: str) -> str:
    """Simple instruction-style prompt for factual QA."""
    return (
        "You are a helpful assistant. "
        "Answer the following question in one short phrase.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def load_hf_model(model_id: str, device: str, token: str | None):
    """Load model + tokenizer via transformers."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise SystemExit(
            "ERROR: transformers and torch are required.\n"
            "Install with:  pip install transformers torch accelerate"
        )

    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        device_map=device if device != "cpu" else None,
        torch_dtype=dtype,
    )
    if device == "cpu":
        model = model.to("cpu")
    print(f"[OK] {model_id} loaded on {device}\n")
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_new_tokens: int) -> str:
    """Generate a short answer from the model."""
    import torch

    prompt = format_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return only the part after "Answer:"
    if "Answer:" in full_text:
        full_text = full_text.split("Answer:")[-1].strip()

    return full_text


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run open-factual-bench on a Hugging Face model."
    )
    parser.add_argument(
        "--model_id",
        default="google/gemma-2-2b-it",
        help="Hugging Face model ID (default: google/gemma-2-2b-it)",
    )
    parser.add_argument(
        "--hardware",
        default="unknown",
        help='Hardware description, e.g. "Colab T4 16GB"',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Max tokens to generate per question (default: 64)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu"],
        help="Device: 'auto' (uses GPU if available) or 'cpu'",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Offline/dry-run mode: use reference answers as predictions",
    )
    args = parser.parse_args()

    # ── Build config ───────────────────────────────────────────────
    config = BenchmarkConfig(
        model_id=args.model_id if not args.offline else "offline-sim",
        eval_type="factual_qa",
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        hardware=args.hardware,
    )
    run = BenchmarkRun.create_new(config.__dict__)

    # ── Load model (unless offline) ────────────────────────────────
    model, tokenizer = None, None
    if not args.offline:
        token = os.environ.get("HF_TOKEN")
        model, tokenizer = load_hf_model(args.model_id, args.device, token)

    # ── Evaluate ───────────────────────────────────────────────────
    total_start = time.time()
    correct, graded, skipped = 0, 0, 0
    hallucinated, refused, unclear_h = 0, 0, 0

    print(f"Evaluating {len(EXAMPLE_TASKS)} tasks ...\n")
    print(f"{'ID':20} {'Domain':15} {'Score':8} {'Pred (first 60 chars)'}")
    print("-" * 80)

    for task in EXAMPLE_TASKS:
        t0 = time.time()

        if args.offline:
            pred = task.reference_answer
            tokens_gen = 0
        else:
            pred = generate_answer(model, tokenizer, task.question, args.max_new_tokens)
            tokens_gen = len(tokenizer.encode(pred))

        elapsed_ms = (time.time() - t0) * 1000
        is_correct = score_task(pred, task)

        # Hallucination labeling for stress-test tasks
        result_meta = {}
        task_type = (task.metadata or {}).get("type", "")
        if task_type == "hallucination_stress":
            h_label = label_hallucination_task(pred, task)
            result_meta.update(h_label)
            label_val = h_label.get("hallucination_label", "")
            if label_val == "hallucination_candidate":
                hallucinated += 1
            elif label_val == "refusal_or_correction":
                refused += 1
            else:
                unclear_h += 1

        result = TaskResult(
            task_id=task.id,
            model_response=pred,
            is_correct=is_correct,
            latency_ms=round(elapsed_ms, 2),
            tokens_generated=tokens_gen,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            metadata=result_meta if result_meta else None,
        )
        run.results.append(result)

        # Stats
        if is_correct is None:
            skipped += 1
            marker = "-"
        elif is_correct:
            correct += 1
            graded += 1
            marker = "Y"
        else:
            graded += 1
            marker = "N"

        print(f"{task.id:20} {task.domain:15} {marker:8} {pred[:60]}")

    run.total_duration_sec = round(time.time() - total_start, 2)

    # ── Summary ────────────────────────────────────────────────────
    accuracy = correct / graded if graded else 0.0
    print("\n" + "=" * 80)
    print(f"Model:     {config.model_id}")
    print(f"Hardware:  {config.hardware}")
    print(f"Accuracy:  {accuracy:.1%}  ({correct}/{graded} auto-graded)")
    print(f"Skipped:   {skipped} (not auto-gradable)")
    total_h = hallucinated + refused + unclear_h
    if total_h > 0:
        print(f"Hallucination tasks: {total_h}  "
              f"(Hallucinated: {hallucinated} | Refused: {refused} | Unclear: {unclear_h})")
    print(f"Duration:  {run.total_duration_sec:.1f}s")
    print("=" * 80)

    # ── Save ───────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    safe_model = config.model_id.replace("/", "-")
    output_path = f"results/{run.run_id}_{safe_model}.json"

    data = {
        "run_id": run.run_id,
        "config": run.config,
        "summary": {
            "total_tasks": len(EXAMPLE_TASKS),
            "auto_graded": graded,
            "correct": correct,
            "accuracy": accuracy,
            "skipped": skipped,
            "hallucination_tasks": hallucinated + refused + unclear_h,
            "hallucinated": hallucinated,
            "refused": refused,
            "unclear": unclear_h,
        },
        "results": [asdict(r) for r in run.results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
