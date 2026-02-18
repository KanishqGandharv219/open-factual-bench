# Open LLM Factuality & Hallucination Benchmark

## Vision

Large language model benchmarks are under increasing criticism for weak methodology, data contamination, and leaderboard scores that are easy to game.
At the same time, new work such as DeepMind's FACTS Benchmark Suite and FACTS Grounding highlights the need for systematic, transparent evaluation of model factuality on real-world questions.
This project aims to build a small but rigorous, open benchmark that measures the factuality and hallucination behaviour of **open LLMs** (e.g. Gemma, Llama, Mistral, Phi), with a focus on reproducibility and realistic hardware constraints.

## Scope

The benchmark is designed to be:

- **Model-agnostic**: works with any Hugging Face-style text generation model, including Gemma, Llama, Mistral, Phi and others.
- **Factuality-focused**: tests whether answers are correct, grounded, and free of hallucinated details.
- **Contamination-aware**: prioritises post-2024 topics and adversarial rewrites to reduce training-data leakage.
- **Hardware-aware**: targets common setups like Colab T4 and Kaggle GPUs, with configs that make trade-offs explicit.
- **Reproducible**: every run is tied to a config, model version, and hardware description so results can be replicated.

Planned evaluation modes:

- Factual QA (closed-book and retrieval-augmented)
- Hallucination stress tests
- Citation checking (answers must point to evidence in the provided context)
- Multi-domain tasks (science, math, code, current events)

## Quick Start

```bash
# Install dependencies
pip install transformers torch accelerate

# Offline dry-run (no GPU required)
python run_benchmark.py --offline --hardware "offline-sim"

# Evaluate a real model (GPU recommended)
export HF_TOKEN=your_token_here
python run_benchmark.py --model_id google/gemma-2-2b-it --hardware "Colab T4"

# CPU mode (slow but works anywhere)
python run_benchmark.py --model_id google/gemma-2-2b-it --device cpu --hardware "CPU laptop"
```

Results are saved to `results/` as structured JSON files.

## Prior Work: Gemma Family Comparison (100-question evaluation)

Before building this framework, we ran an ad-hoc 100-question factual QA evaluation in a separate repo: [gemma-mini-benchmark](https://github.com/KanishqGandharv219/gemma-mini-benchmark). Key findings from that evaluation (Colab T4 GPU):

| Model | Params | Type | Accuracy |
|:------|:-------|:-----|:---------|
| `google/gemma-2b` | 2B | Base | 19% |
| `google/gemma-2b-it` | 2B | Instruct | **85%** |
| `google/gemma-7b` | 7B | Base | 21% |
| `google/gemma-7b-it` | 7B | Instruct | 69% |

**Takeaways**: Instruction tuning matters more than scale (+66pp for 2B-IT vs 2B Base). That early work motivated building this structured framework.

## High-level Design

The core of the benchmark is:

- A set of **tasks**, each with a domain, question, reference answer, provenance metadata (`source`, `created_at`), and optional supporting context.
- A **benchmark configuration**, which specifies the model, decoding parameters, hardware, and evaluation mode.
- A **scoring engine** with Unicode normalization, substring matching, and multi-word answer support.

The goal is to make it easy to:

- Run the same suite across different open models.
- Compare factuality and hallucination rates under realistic constraints.
- Share results in a lightweight leaderboard-style view.

## Project Structure

```
open-factual-bench/
├── bench/
│   ├── schema.py            # Task + BenchmarkConfig dataclasses
│   ├── tasks_example.py     # 32 curated tasks across 8 domains
│   ├── scoring.py           # Scoring engine (Unicode-aware, substring, token match)
│   └── results_schema.py    # TaskResult + BenchmarkRun dataclasses
├── run_benchmark.py         # CLI: evaluate any HF model against tasks
├── run_example_offline.py   # Offline simulation runner
├── recompute_scores.py      # Re-score existing result files
├── test_example_tasks.py    # Smoke test: list tasks + print config
└── results/                 # Saved benchmark run results (JSON)
```

## Status

- [x] Define core task and config schema
- [x] Add provenance fields for contamination-aware design (`source`, `created_at`)
- [x] Implement a minimal factual QA task set (32 tasks, 8 domains)
- [x] Implement improved scoring (Unicode normalization, substring matching)
- [x] Create CLI model runner (`run_benchmark.py`)
- [x] Evaluate Gemma 2B/7B family (base + instruction-tuned)
- [ ] Evaluate more open models (e.g. Llama, Mistral, Phi)
- [ ] Build a simple leaderboard UI
- [ ] Publish example Colab/Kaggle notebooks

Feedback on the benchmark design and use cases is very welcome.

## Design Notes (Contamination-Aware)

Recent work such as LiveBench and MMLU-CF shows how easily static benchmarks become contaminated as soon as their questions leak into public training corpora.
To avoid this, open-factual-bench is designed with:

- A preference for **post-2024 or recently-updated questions**, inspired by dynamic benchmarks like LiveBench.
- **Clear metadata** on each task (e.g. approximate source and creation time) to make future contamination checks easier.
- An emphasis on **objective, auto-gradable factual questions** where possible, following the spirit of FACTS and LiveBench.
- A path to **community-reported results** via simple JSON/YAML files, aligned with how Hugging Face "Community Evals" handle open benchmarking.

## Design Notes: Scoring

The `score_task()` function in `bench/scoring.py` applies the following pipeline:

1. **Skip non-gradable tasks** — If the reference answer is a placeholder (`[...]`) or longer than 80 characters (e.g. hallucination stress-test instructions like *"This is a fictional setting; models should refuse…"*), the scorer returns `None`. These tasks are excluded from accuracy calculations.
2. **Normalize** — Both prediction and reference are Unicode-normalized (NFKD), lowercased, whitespace-collapsed, and stripped of leading articles ("the", "a", "an").
3. **Exact match** — If the normalized strings are identical → **correct**.
4. **Substring containment** — If the normalized reference appears within the first line of the prediction → **correct**. This handles models that answer "Paris is the capital of France" when the reference is just "Paris".
5. **Short-answer token match** — For references ≤5 characters (e.g. "8", "Au", "def"), the scorer also checks if the reference appears as a standalone token in the first line.

If none of these match → **incorrect**.

This keeps scoring deterministic and fast while handling the most common failure modes from real model outputs (verbose answers, Unicode variants, extra context).

## Example Run: Gemma-2B-IT on Colab T4

Run via `python run_benchmark.py --model_id google/gemma-2-2b-it --hardware "Colab T4 16GB"` on Feb 18, 2026.

| Metric | Value |
|--------|-------|
| Accuracy | **90.0%** (27/30 auto-graded) |
| Skipped | 2 (hallucination stress-tests, not auto-gradable) |
| Duration | 13.5s |

**Notable observations:**
- **Hallucination detected**: The model answered "King's Landing" for the fictional country Westeros — a classic hallucination (treating fiction as fact).
- **Correct refusal**: For the false-premise Einstein/2025 question, the model correctly responded "He did not win a Nobel Prize in 2025."
- **Post-training-cutoff**: The 2026 Australian Open question returned an empty response — the model correctly doesn't fabricate unknown facts.
- **Scoring edge case**: "Eight" vs "8" — the scorer doesn't yet convert number words to digits (future improvement).

Full results: `results/run_20260217_060342_gemma-2b-colab-t4.json`

