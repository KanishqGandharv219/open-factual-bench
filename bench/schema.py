from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal

Domain = Literal["science", "math", "code", "current_events", "other"]
EvalType = Literal["factual_qa", "hallucination", "retrieval_qa", "citation_check"]

@dataclass
class Task:
    id: str
    domain: Domain
    question: str
    reference_answer: str
    context: Optional[str] = None
    # more explicit provenance fields
    source: Optional[str] = None          # e.g. "synthetic_demo_v1", "news_2025_06"
    created_at: Optional[str] = None      # e.g. "2025-06"
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkConfig:
    model_id: str
    eval_type: EvalType
    max_new_tokens: int
    temperature: float
    hardware: str          # e.g. "T4 Colab", "A100 40GB"
    seed: int = 0
    extra: Optional[Dict[str, Any]] = None
