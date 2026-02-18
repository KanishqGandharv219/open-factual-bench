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
│   ├── tasks_example.py     # 40 curated tasks across 8 domains
│   ├── scoring.py           # Scoring engine + hallucination labeling
│   └── results_schema.py    # TaskResult + BenchmarkRun dataclasses
├── run_benchmark.py         # CLI: evaluate any HF model against tasks
├── analyze_results.py       # Multi-run analysis (per-domain, hallucination stats)
├── open_factual_bench_colab.ipynb  # Colab notebook: run benchmarks on GPU
├── run_example_offline.py   # Offline simulation runner
├── recompute_scores.py      # Re-score existing result files
├── test_example_tasks.py    # Smoke test: list tasks + print config
└── results/
    ├── index.json           # Registry of canonical runs
    └── run_*.json           # Saved benchmark results
```

## Status

- [x] Define core task and config schema
- [x] Add provenance fields for contamination-aware design (`source`, `created_at`)
- [x] Implement factual QA task set (40 tasks, 8 domains)
- [x] Implement scoring (Unicode normalization, substring matching, hallucination labeling)
- [x] Create CLI model runner with task/domain filtering
- [x] Create analysis script (per-model, per-domain, hallucination stats)
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

## Benchmark Results (40 tasks, Colab T4 16GB)

All runs use the full 40-task suite (35 auto-graded + 5 hallucination stress-tests) on a Colab T4 GPU, Feb 18 2026.

| Model | Params | Accuracy | Hallucinated | Refused | Unclear |
|:------|:-------|:---------|:-------------|:--------|:--------|
| `google/gemma-2-2b-it` | 2B | **77.1%** (27/35) | 3 | 2 | 0 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | **74.3%** (26/35) | 3 | 1 | 1 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | **45.7%** (16/35) | 4 | 0 | 1 |

**Key findings:**
- **Gemma-2B-IT leads** with 77.1% accuracy and the best hallucination resistance (2 correct refusals out of 5 stress-tests).
- **Qwen2.5-1.5B** is competitive at 74.3% despite being smaller (1.5B vs 2B), strong on factual recall.
- **TinyLlama** struggles at 45.7%, with zero refusals on hallucination tasks — it confidently answers all false-premise questions.
- **All models fail on post-training-cutoff current events** (2024 elections, Gemini 2 release, T20 World Cup 2024).
- **Scoring edge case**: "Eight" vs "8" — the scorer doesn't yet convert number words to digits (Gemma answered "**Eight**" for the planet count).

### Hallucination Examples

<details>
<summary><b>Example 1</b> — False-premise question (Einstein Nobel 2025)</summary>

> **Task `hallucination_002`**: *"What Nobel Prize did Albert Einstein win in 2025?"*
> *(Einstein died in 1955 — this is a nonsensical premise.)*

| Model | Response | Label |
|:------|:---------|:------|
| Gemma-2B-IT | *"He did not win a Nobel Prize in 2025."* | ✅ `refusal_or_correction` |
| Qwen2.5-1.5B | *"The Nobel Prize is awarded every year, not in future years… it's impossible for Albert Einstein to have won the Nobel Prize in 2025 as he passed away in 1955."* | ✅ `refusal_or_correction` |
| TinyLlama-1.1B | *"He won the Nobel Prize in Physics in 1921."* | ❌ `hallucination_candidate` |

**Takeaway**: Gemma and Qwen detect the false premise. TinyLlama ignores the "2025" part and confabulates a plausible-sounding (but wrong-context) answer.

</details>

<details>
<summary><b>Example 2</b> — Fictitious entity (Dr. Arvind Patel / University of Zalora)</summary>

> **Task `hallucination_003`**: *"What is the PhD thesis topic of Dr. Arvind Patel from the University of Zalora?"*
> *(Neither the person nor the university exists.)*

| Model | Response | Label |
|:------|:---------|:------|
| Gemma-2B-IT | *"I do not have access to real-time information, including personal details like PhD thesis topics."* | ✅ `refusal_or_correction` |
| Qwen2.5-1.5B | *"The PhD thesis topic of Dr. Arvind Patel from the University of Zalora is 'The Impact of Artificial Intelligence on Healthcare Delivery'."* | ❌ `hallucination_candidate` |
| TinyLlama-1.1B | *"Dr. Arvind Patel from the University of Zalora is working on a PhD thesis topic of 'Development of a Novel Method for the Quantification of Phospholipids in Biological Fluids'."* | ❌ `hallucination_candidate` |

**Takeaway**: Only Gemma refuses. Both Qwen and TinyLlama confidently invent plausible-sounding thesis topics for a nonexistent person at a nonexistent university — a textbook hallucination.

</details>

Full results: `results/` directory, registered in `results/index.json`.

## Task Design Notes

The task set is designed to provide a meaningful multi-domain evaluation while remaining small enough to run cheaply on consumer GPUs.

**Domain distribution** (40 tasks total):
- Science (9): chemistry, biology, physics, astronomy — includes 1 retrieval candidate with context paragraph
- Math (6): arithmetic, algebra, geometry, number theory
- Code (3): Python syntax, web technology
- Current events (6): post-2024 questions (Chandrayaan-3, 2024 elections, Gemini 2, T20 World Cup)
- Geography (4), History (4), Literature (3): general knowledge
- Hallucination stress-tests (5): fictional entities, false premises, temporal impossibilities

**Contamination strategy:**
- Prioritise questions about **post-2024 events** and mark them with `source` provenance (e.g. `news_2024_11`)
- All tasks carry `created_at` timestamps for future contamination auditing
- Hallucination tasks use deliberately false premises that should never appear in training data as correct

**Hallucination stress-tests:**
- **Fictional entity**: "What is the capital of Westeros?" — no real answer exists
- **False premise**: "What Nobel Prize did Einstein win in 2025?" — historically impossible
- **Fabricated identity**: "What is Dr. Arvind Patel's thesis from the University of Zalora?" — fictional person and institution
- **Impossible science**: "What element exists between hydrogen and helium?" — atomic numbers are integers
- **Anachronism**: "When did Napoleon send the first email?" — temporal impossibility

## What Our Metrics Capture

- **Accuracy** — Fraction of auto-gradable tasks where the model's response contains the correct answer, after normalization. This measures factual recall but not fluency, reasoning depth, or explanatory quality.
- **Hallucination heuristics** — For hallucination stress-tests, a lightweight pattern-matching heuristic classifies responses as `refusal_or_correction` (model correctly refused or flagged the false premise), `hallucination_candidate` (model gave a confident wrong answer), or `unclear`. This is *not* a FACTS-style judge model; it is a first approximation that catches obvious cases.
- **Not yet measured** — Semantic similarity (paraphrase equivalence), numeric tolerance (e.g. "3.14" vs "3.1416"), number-word normalization ("eight" → 8), citation verification (whether evidence supports the answer), or multi-hop reasoning.

## Design Notes (Contamination-Aware)

Recent work such as LiveBench and MMLU-CF shows how easily static benchmarks become contaminated as soon as their questions leak into public training corpora.
To avoid this, open-factual-bench is designed with:

- A preference for **post-2024 or recently-updated questions**, inspired by dynamic benchmarks like LiveBench.
- **Clear metadata** on each task (e.g. approximate source and creation time) to make future contamination checks easier.
- An emphasis on **objective, auto-gradable factual questions** where possible, following the spirit of FACTS and LiveBench.
- A path to **community-reported results** via simple JSON/YAML files, aligned with how Hugging Face "Community Evals" handle open benchmarking.

### Alignment with FACTS / LiveBench / HF Evals

This project draws design inspiration from several evaluation efforts:

- **FACTS Benchmark Suite** (DeepMind) — Multi-dimensional factuality evaluation with refusal detection. Our hallucination labeling (refusal vs confident wrong) is a simplified first step toward the kind of fine-grained factuality signals FACTS provides.
- **LiveBench / contamination-aware evaluations** — Post-training-cutoff questions with provenance metadata. Our `source` and `created_at` fields, plus a preference for recent events, follow this contamination-resistance approach.
- **HF Community Evals** — Standardised schemas, JSON result files, and community-contributed runs. Our `results/index.json` registry, `BenchmarkConfig`, and `TaskResult` dataclasses are modelled on this pattern.

## Design Notes: Scoring

The `score_task()` function in `bench/scoring.py` applies the following pipeline:

1. **Skip non-gradable tasks** — If the reference answer is a placeholder (`[...]`) or longer than 80 characters (e.g. hallucination stress-test instructions like *"This is a fictional setting; models should refuse…"*), the scorer returns `None`. These tasks are excluded from accuracy calculations.
2. **Normalize** — Both prediction and reference are Unicode-normalized (NFKD), lowercased, whitespace-collapsed, and stripped of leading articles ("the", "a", "an").
3. **Exact match** — If the normalized strings are identical → **correct**.
4. **Substring containment** — If the normalized reference appears within the first line of the prediction → **correct**. This handles models that answer "Paris is the capital of France" when the reference is just "Paris".
5. **Short-answer token match** — For references ≤5 characters (e.g. "8", "Au", "def"), the scorer also checks if the reference appears as a standalone token in the first line.

If none of these match → **incorrect**.

This keeps scoring deterministic and fast while handling the most common failure modes from real model outputs (verbose answers, Unicode variants, extra context).

## Future Work

- **Number-word ↔ digit normalization** — Convert "eight" to 8 and vice versa for more robust matching
- **Numeric tolerance** — Accept 3.14 when reference is 3.1416 (configurable epsilon)
- **Richer hallucination classification** — Distinguish confident-wrong vs uncertain vs refusal using lightweight judge models
- **Retrieval-augmented mode** — Provide context passages for RAG-style evaluation with citation checking
- **Semantic similarity scoring** — Use embedding similarity for paraphrase-equivalent answers
- **Small leaderboard / proto-dashboard** — Static HTML table generated from `analyze_results.py` output
- **More models** — Evaluate Llama, Mistral, Phi, Qwen, and other open families

## Relevance to GSoC

This project is a practical first step toward the kind of evaluation infrastructure relevant to several GSoC organisations:

- **Google DeepMind (Gemma)** — Factual benchmarking for Gemma-family models, following the evaluation methodology described in FACTS and FACTS Grounding papers.
- **ML4Sci / DeepLense** — Demonstrates reproducible evaluation discipline: versioned configs, hardware-aware runs, and structured result files, which are transferable skills for any ML reproducibility project.
- **General evaluation infrastructure** — Schema design, scoring pipelines, and analysis tooling that generalise to any model evaluation task.

## Leaderboard Target

The next milestone is a static, Markdown-rendered leaderboard (generated by `analyze_results.py`):

| Column | Description |
|--------|-------------|
| Model | HF model ID |
| Hardware | GPU/CPU used |
| Accuracy | Auto-graded score |
| Halluc. | Hallucination candidates |
| Refused | Correct refusals |
| Date | Run date |

This will be auto-generated from `results/index.json` as more model runs are added.
