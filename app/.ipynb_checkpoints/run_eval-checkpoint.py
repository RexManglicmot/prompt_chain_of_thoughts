# app/run_eval.py
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from app.config import cfg
from app.prompt import direct_prompt, cot_prompt
from app.metrics import (exact_match, token_guess, Timer)
from app.load_model import call_model



# Helpers
_LABEL_SET = {"yes", "no", "maybe", "not in context"}
_LABEL_ANY = re.compile(r"\b(yes|no|maybe|not\s+in\s+context)\b", re.IGNORECASE)

def _canon(s: str) -> str:
    if not s: return ""
    s = s.strip().lower()
    if "not in context" in s: return "not in context"
    if s.startswith("yes"): return "yes"
    if s.startswith("no") and s != "na": return "no"
    if s.startswith("maybe"): return "maybe"
    return s if s in _LABEL_SET else ""

def _extract_label(raw: str) -> str:
    """
    Robust label extraction:
      1) Prefer text right after 'Final Answer' up to newline/bracket/period/ID tag.
      2) Else search globally for any allowed label.
      3) Else 'not in context'.
    """
    if not raw:
        return "not in context"

    # 1) After 'Final Answer'
    m = re.search(r"final\s*answer[^:]*:\s*([^\n\[\]\.\r\|]+)", raw, re.IGNORECASE)
    if m:
        cand = _canon(m.group(1))
        if cand: return cand

    # 2) Anywhere
    m = _LABEL_ANY.search(raw or "")
    if m:
        return _canon(m.group(1))

    # 3) Fallback
    return "not in context"


def main():
    # paths
    results_csv = cfg.paths.results_csv
    summary_csv = cfg.paths.summary_csv
    Path(results_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)

    # data
    # id,question,context_id,context,gold_answer,has_answer_in_context
    df = pd.read_csv(cfg.paths.data_csv) 
    
    # experiment config
    methods = cfg.experiment.methods

    # inference config (split token budgets with fallback)
    max_new_direct = getattr(cfg.inference, "max_new_tokens_direct", None)
    max_new_cot    = getattr(cfg.inference, "max_new_tokens_cot", None)
    fallback_max   = getattr(cfg.inference, "max_new_tokens", 128)
    max_new_direct = max_new_direct or fallback_max
    max_new_cot    = max_new_cot or fallback_max
    t_direct = cfg.inference.temperature_direct
    t_cot    = cfg.inference.temperature_cot

    # Create an empty list to fill at the end
    rows = []

    # Iterate over the DataFrame with a progress bar:
    # tqdm(...) wraps the row iterator to show progress (desc text + total rows)
    # df.iterrows() yields (index, row) pairs
    # `_` discards the row index; `r` is a pandas Series for that row
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        
        # Read the gold label from this row and canonicalize it:
        # r["gold_answer"] pulls the cell value (attribute access r.gold_answer also works)
        # str(...) guards against non-strings/NaN by forcing a string
        # _canon(...) is your normalizer (e.g., trims, lowercases to {yes,no,maybe})
        gold_label = _canon(str(r.gold_answer))

        for method in methods:
            if method == "direct":
                prompt  = direct_prompt(r.context_id, r.context, r.question)
                prompts = [prompt]
                temps   = [t_direct]
                max_tok = [max_new_direct]
            else:
                base    = cot_prompt(r.context_id, r.context, r.question)
                prompts = [base] 
                temps   = [t_cot] 
                max_tok = [max_new_cot]

            # Create 2 empty lists to fill. 
            #`outs` for model outputs (strings), 
            # `lats` for per-call latency (ms)
            outs, lats = [], []

            # Iterate in lockstep over prompts, temperatures, and per-call max token limits.
            # NOTE: zip() stops at the shortest list; ensure all three lists are equal-length.
            for p, temp, mx in zip(prompts, temps, max_tok):
                # Time just the model call (encode → generate → decode).
                # TIP: If running on CUDA, ensure your Timer calls torch.cuda.synchronize()
                # in __enter__/__exit__ to get accurate timings (GPU ops are async).
                with Timer() as t:
                    # call_model returns ONLY the generated continuation
                    outs.append(call_model(p, mx, temp))
                # Store elapsed time in milliseconds for this iteration
                lats.append(t.ms)
                

            # Pick the first generated string as the model's raw output.
            # `outs` is a list of model continuations; we take index 0 if it exists.
            # If `outs` is empty (e.g., a failed generation), fall back to an empty string.
            raw = outs[0] if outs else ""

            # Reduce the raw text to a canonical label ("yes" | "no" | "maybe" | "not in context").
            # `_extract_label` first prefers text after "Final Answer:", else searches anywhere,
            # then normalizes via `_canon`. If nothing is found, it returns "not in context".
            label = _extract_label(raw)  # uses _canon() internally

            # metrics
            em   = exact_match(label, gold_label)                     # yes, no, maybe
            tok  = token_guess(raw)                                   # token cost of full output

            rows.append({
                "id": r.id,
                "method": method,
                "gold_answer": gold_label,    # canonical gold
                "model_raw": raw,             # full continuation text (unaltered)
                "model_answer": label,        # canonical label
                "correct_em": em,
                "tokens_out": tok,
                "latency_ms": sum(lats)//max(1, len(lats)),
            })

    # Convert rows into a pandas Dataframe
    res = pd.DataFrame(rows)

    # Convert res into a csv called results_csv
    res.to_csv(results_csv, index=False)

    # Create summary
    summary = (
        res.groupby("method")
          .agg(
              em=("correct_em","mean"),
              tokens_out=("tokens_out","median"),
              latency_p50=("latency_ms","median"),
              latency_p95=("latency_ms", lambda s: int(s.quantile(0.95))),
          )
          .reset_index()
    )

    # Save summary to csv called summary_csv
    summary.to_csv(summary_csv, index=False)

    # Write final note, saying goodbye
    print("[done] wrote:", results_csv, "and", summary_csv)


if __name__ == "__main__":
    main()
