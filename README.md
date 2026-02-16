# Open LLM Factuality & Hallucination Benchmark (WIP)

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

## High-level design (WIP)

The core of the benchmark is:

- A set of **tasks**, each with a domain, question, reference answer, and optional supporting context.  
- A **benchmark configuration**, which specifies the model, decoding parameters, hardware, and evaluation mode.

The goal is to make it easy to:

- Run the same suite across different open models.  
- Compare factuality and hallucination rates under realistic constraints.  
- Share results in a lightweight leaderboard-style view.

## Status

This repository is in the early design phase:

- [x] Define core task and config schema  
- [ ] Implement a minimal factual QA task set  
- [ ] Evaluate a small set of open models (e.g. Gemma, Llama, Mistral)  
- [ ] Build a simple leaderboard UI  
- [ ] Publish example configs and Colab/Kaggle notebooks

Feedback on the benchmark design and use cases is very welcome.

## Design notes (contamination-aware)

Recent work such as LiveBench and MMLU-CF shows how easily static benchmarks become contaminated as soon as their questions leak into public training corpora. 
To avoid this, open-factual-bench is designed with:

- A preference for **post-2024 or recently-updated questions**, inspired by dynamic benchmarks like LiveBench.
- **Clear metadata** on each task (e.g. approximate source and creation time) to make future contamination checks easier.  
- An emphasis on **objective, auto-gradable factual questions** where possible, following the spirit of FACTS and LiveBench.  
- A path to **community-reported results** via simple JSON/YAML files, aligned with how Hugging Face “Community Evals” handle open benchmarking.
