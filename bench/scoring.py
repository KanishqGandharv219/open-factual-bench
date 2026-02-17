from typing import Optional
from .schema import Task

def exact_match_score(pred: str, task: Task) -> Optional[bool]:
    """
    Very simple baseline scorer:
    - lowercases and strips both prediction and reference
    - returns True / False
    - returns None if reference looks like a placeholder
    """
    ref = task.reference_answer.strip().lower()
    if "[" in ref and "]" in ref:
        return None  # e.g. placeholder like "[To be filled...]"

    pred_norm = pred.strip().lower()
    return pred_norm == ref
