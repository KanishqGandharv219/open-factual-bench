import re
import unicodedata
from typing import Optional
from .schema import Task


def _normalize(text: str) -> str:
    """Normalize a string for comparison.

    Steps:
      1. Unicode NFKD normalization (e.g. subscript digits → ASCII).
      2. Lowercase.
      3. Strip leading/trailing whitespace.
      4. Collapse internal whitespace to single spaces.
      5. Remove common punctuation noise (backticks, trailing periods/commas).
      6. Strip common leading articles ("the ", "a ", "an ").
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("`", "").rstrip(".,;:!?")
    # Strip leading articles for fairer comparison
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return text.strip()


def _is_placeholder(ref: str) -> bool:
    """Return True if the reference answer looks like an unfilled placeholder."""
    ref_lower = ref.strip().lower()
    return "[" in ref_lower and "]" in ref_lower


def score_task(pred: str, task: Task) -> Optional[bool]:
    """Score a model prediction against a task's reference answer.

    Returns:
      True  – correct
      False – incorrect
      None  – cannot be auto-graded (placeholder reference or hallucination
              stress-test with long descriptive reference)

    Scoring strategy (applied in order):
      1. Skip placeholders and long descriptive references (hallucination
         stress-tests whose reference is an instruction, not a short answer).
      2. Exact match after normalization.
      3. Substring containment: the normalized reference appears inside the
         *first line* of the normalized prediction. This handles models that
         give the right answer followed by an explanation.
      4. For short numeric / symbolic answers (≤5 chars), also check whether
         the reference appears as a standalone token in the first line.
    """
    ref_raw = task.reference_answer.strip()

    # 1. Skip un-gradable tasks
    if _is_placeholder(ref_raw):
        return None
    # Long references (>80 chars) are typically descriptions, not answers
    if len(ref_raw) > 80:
        return None

    ref = _normalize(ref_raw)
    pred_full = _normalize(pred)

    # 2. Exact match
    if pred_full == ref:
        return True

    # 3. Substring containment in first line
    first_line = pred_full.split("\n")[0]
    if ref in first_line:
        return True

    # 4. Short-answer token match in first line
    if len(ref) <= 5:
        tokens = re.split(r"[\s`.,;:!?()\[\]{}\"']+", first_line)
        if ref in tokens:
            return True

    return False


# ── Hallucination labeling (heuristic) ─────────────────────────────

# Patterns that suggest the model is refusing or correcting the premise.
_REFUSAL_PATTERNS = [
    r"\b(i don'?t know|i cannot|i can'?t|not sure)\b",
    r"\b(did not|does not|doesn'?t|do not|don'?t)\b",
    r"\b(fictional|fictitious|not real|doesn'?t exist|does not exist)\b",
    r"\b(false premise|impossible|no such|never happened)\b",
    r"\b(no evidence|cannot (find|verify|confirm))\b",
    r"\b(died in \d{4}|was invented|not possible)\b",
]

_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

# Simple heuristic: a response with a capitalized multi-word phrase
# (likely a proper noun) and no refusal signal → probable hallucination.
_PROPER_NOUN_RE = re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+")


def label_hallucination_task(pred: str, task: "Task") -> dict:
    """Label a hallucination stress-test response (heuristic, not a judge model).

    Returns a dict suitable for storing in TaskResult.metadata:
      {"hallucination_label": "refusal_or_correction"}
        — model refused, said "I don't know", or flagged the false premise.
      {"hallucination_label": "hallucination_candidate"}
        — model gave a confident, concrete answer (likely wrong).
      {"hallucination_label": "unclear"}
        — response was too short or ambiguous to classify.

    This is intentionally a lightweight pattern-matching heuristic,
    not a FACTS-style judge model. It catches obvious cases only.
    """
    pred_stripped = pred.strip()

    # Empty or very short response → unclear
    if len(pred_stripped) < 3:
        return {"hallucination_label": "unclear"}

    # Check for refusal patterns
    if _REFUSAL_RE.search(pred_stripped):
        return {"hallucination_label": "refusal_or_correction"}

    # Check for confident answer: proper nouns or concrete specifics
    if _PROPER_NOUN_RE.search(pred_stripped):
        return {"hallucination_label": "hallucination_candidate"}

    # Fallback: if response is a short definitive-looking phrase, suspect hallucination
    if len(pred_stripped.split()) <= 8 and not pred_stripped.endswith("?"):
        return {"hallucination_label": "hallucination_candidate"}

    return {"hallucination_label": "unclear"}


# Backward-compatible alias
def exact_match_score(pred: str, task: Task) -> Optional[bool]:
    """Legacy alias for score_task(). Kept for backward compatibility."""
    return score_task(pred, task)
