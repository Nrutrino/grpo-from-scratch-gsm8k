"""
The verifier — the free oracle that makes RL work here.

GSM8K gold answers look like:
    ...reasoning...
    #### 18
so the gold number is whatever follows '#### '.

The model is prompted to end with '\\boxed{ANSWER}'.  We extract its answer with a
small cascade of patterns (boxed → "answer is" → last number) and compare numerically.

reward(correct) = 1.0,  reward(wrong) = 0.0.   That binary correctness IS the signal.
An optional small `format_bonus` can densify the early signal (see config), but the
default is the pure verifier.
"""
import re

_NUM = r"-?\$?\d[\d,]*\.?\d*"


def _normalize(s: str) -> str | None:
    """Strip $/commas/whitespace/trailing punctuation; return a canonical number str."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.rstrip(".")
    m = re.search(r"-?\d+\.?\d*", s)
    return m.group(0) if m else None


def extract_gold(answer_text: str) -> str | None:
    """The number after '#### ' in a GSM8K answer."""
    m = re.search(r"####\s*(" + _NUM + r")", answer_text)
    return _normalize(m.group(1)) if m else None


def extract_pred(completion: str) -> str | None:
    """Best-effort extraction of the model's final answer."""
    # 1. \boxed{...}  (preferred — the format we ask for)
    boxed = re.findall(r"\\boxed\{([^}]*)\}", completion)
    if boxed:
        return _normalize(boxed[-1])
    # 2. "answer is X" / "answer: X"
    m = re.findall(r"(?:answer|=)\s*(?:is|:)?\s*(" + _NUM + r")", completion, flags=re.I)
    if m:
        return _normalize(m[-1])
    # 3. fallback: the last number anywhere in the completion
    nums = re.findall(_NUM, completion)
    return _normalize(nums[-1]) if nums else None


def _equal(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred == gold


def compute_reward(completion: str, gold: str, format_bonus: float = 0.0) -> float:
    """1.0 if the extracted answer matches gold, else 0.0 (+ optional format bonus)."""
    pred = extract_pred(completion)
    r = 1.0 if _equal(pred, gold) else 0.0
    if format_bonus > 0.0 and re.search(r"\\boxed\{[^}]*\}", completion):
        r += format_bonus
    return r


def is_correct(completion: str, gold: str) -> bool:
    return _equal(extract_pred(completion), gold)
