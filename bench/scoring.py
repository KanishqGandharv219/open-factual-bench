from typing import Optional
from .schema import Task

def exact_match_score(pred: str, task: Task) -> Optional[bool]:
    """
    Baseline scorer:
    - If reference looks like a placeholder (e.g. [To be filled...]), return None.
    - Otherwise:
      * Normalize both strings.
      * If they are exactly equal -> True.
      * If the reference is short (<= 5 chars), also accept if it appears
        as a standalone token in the first line of the prediction.
    """
    ref = task.reference_answer.strip().lower()
    if "[" in ref and "]" in ref:
        return None

    pred_norm = pred.strip().lower()

    # Plain exact match
    if pred_norm == ref:
        return True

    # For short answers like "def", accept if present as standalone token in first line
    if len(ref) <= 5:
        first_line = pred_norm.splitlines()[0]
        tokens = first_line.replace("`", " ").replace(".", " ").split()
        if ref in tokens:
            return True

    return False
