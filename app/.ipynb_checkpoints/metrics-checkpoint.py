import re, time
from collections import Counter

_punct = re.compile(r"[^\w\s]")

def normalize(s: str) -> str:
    return _punct.sub("", (s or "").strip().lower())

# Needs the normalize function above for exact_match to work
def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))

def token_guess(s: str) -> int:
    # cheap proxy if your client doesn't return token counts
    return len((s or "").split())

class Timer:
    def __enter__(self): self.t0=time.time(); return self
    def __exit__(self, *a): self.ms=int((time.time()-self.t0)*1000)


