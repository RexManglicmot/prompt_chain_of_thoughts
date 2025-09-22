import re, time
from collections import Counter

_punct = re.compile(r"[^\w\s]")

def normalize(s: str) -> str:
    return _punct.sub("", (s or "").strip().lower())

def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))

def f1_tokens(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    cp = Counter(p); cg = Counter(g)
    overlap = sum((cp & cg).values())
    prec = overlap / max(1, sum(cp.values()))
    rec  = overlap / max(1, sum(cg.values()))
    return 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)

def grounded(answer: str, context_id: str) -> int:
    tag = f"[id:{context_id}]".lower()
    return int(tag in (answer or "").lower())

def abstained(answer: str, abstain_phrase: str) -> int:
    return int((answer or "").strip() == abstain_phrase)

def hallucinated(correct_em: int, grounded_flag: int, has_answer_in_context: int) -> int:
    # Count a hallucination when gold isn't in context, but model still gives wrong & ungrounded content
    if has_answer_in_context: return 0
    return int((correct_em == 0) and (grounded_flag == 0))

def token_guess(s: str) -> int:
    # cheap proxy if your client doesn't return token counts
    return len((s or "").split())

def reasoning_stats(txt: str):
    """Return (coverage, lines) from the CoT 'Reasoning:' block."""
    if not txt: return 0, 0
    part = txt.split("Final Answer:")[0]
    lines = [l for l in part.splitlines() if l.strip() and "reasoning:" not in l.lower()]
    return (1 if len(lines) >= 2 else 0), len(lines)

class Timer:
    def __enter__(self): self.t0=time.time(); return self
    def __exit__(self, *a): self.ms=int((time.time()-self.t0)*1000)


