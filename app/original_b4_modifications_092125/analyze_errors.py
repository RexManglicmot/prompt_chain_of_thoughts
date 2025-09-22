import re, pandas as pd
from pathlib import Path

# === config ===
RESULTS_CSV = Path("outputs/results.csv")  # change if different

ALLOWED = {"yes","no","maybe","not in context"}
NEG_TERMS = (
    "did not","no reduction","no benefit","no effect","no difference",
    "not associated","failed to","was not","were not","no improvement","no change","no evidence of benefit"
)
HEDGE_TERMS = (
    "inconclusive","no significant","not significant","trend","suggests","may","might",
    "limited","small sample","underpowered","unclear","mixed","inconsistent"
)

def canon(s:str)->str:
    s = (s or "").strip().lower()
    if "not in" in s: return "not in context"
    if s.startswith("yes"): return "yes"
    if s.startswith("no") and s!="na": return "no"
    if s.startswith("maybe"): return "maybe"
    return s

def has_negation(text:str)->bool:
    t = (text or "").lower()
    return any(k in t for k in NEG_TERMS)

def has_hedge(text:str)->bool:
    t = (text or "").lower()
    return any(k in t for k in HEDGE_TERMS)

def looks_truncated(raw:str)->bool:
    if not raw: return True
    # no pipe or super long with no closing thought
    if "|" not in raw: return True
    # last token cut-off heuristic
    return raw.strip()[-1] not in ".!?)]””'\""  # ends abruptly

def short_reasoning(raw:str)->bool:
    if "|" not in raw: return True
    reason = raw.split("|",1)[1].replace("Reasoning:","").strip()
    return len(reason.split()) < 20

def classify_row(r):
    """Return reason string for CoT errors only (when EM=0)."""
    if r.method != "cot" or r.correct_em == 1:
        return ""
    raw = str(r.model_raw or "")
    label = canon(str(r.model_answer))
    gold = canon(str(r.gold_answer))
    ctx  = str(r.context if "context" in r else "")
    support = str(getattr(r, "judge_support", "")).lower()

    # 1) Format/parse issues
    if "|" not in raw or not label or label not in ALLOWED:
        return "FORMAT_MISS (label/pipe missing)"

    # 2) Unsupported YES caught by judge/context
    if label == "yes" and (support.startswith("no") or support.startswith("unclear") or has_negation(ctx) or has_hedge(ctx)):
        return "UNSUPPORTED_YES (judge/negation/hedge)"

    # 3) Negation conflict (context negates but model said yes/maybe)
    if has_negation(ctx) and label in {"yes","maybe"}:
        return "NEGATION_CONFLICT"

    # 4) Hedged language conflict (context hedged but model said yes/no)
    if has_hedge(ctx) and label in {"yes","no"} and gold == "maybe":
        return "HEDGE_TO_DECISION (should be maybe)"

    # 5) Truncated or rambling CoT
    if looks_truncated(raw) or r.tokens_out >= r.tokens_out_quant95:
        return "TRUNCATED_OR_RAMBLE"

    # 6) No/weak reasoning
    if short_reasoning(raw):
        return "REASONING_TOO_SHORT"

    # 7) Grounding/citation miss (if you required [ID:...])
    if "grounded" in r and int(getattr(r, "grounded", 0)) == 0 and gold != "not in context":
        return "UNGROUNDED_WITH_ANSWER"

    # 8) Default catch-all
    return "OTHER"

def main():
    df = pd.read_csv(RESULTS_CSV)

    # prepare a 95th percentile to flag long outputs
    q95 = df.groupby("method")["tokens_out"].quantile(0.95).to_dict()
    df["tokens_out_quant95"] = df["method"].map(q95)

    # classify
    df["fail_reason"] = df.apply(classify_row, axis=1)

    # summary: only CoT failures
    fails = df[(df["method"]=="cot") & (df["correct_em"]==0)]
    summary = (fails["fail_reason"]
               .value_counts(dropna=False)
               .rename_axis("reason").reset_index(name="count"))

    print("=== CoT Failure Reasons (counts) ===")
    print(summary.to_string(index=False))

    # show a few examples per top reason
    print("\n=== Examples per top reason ===")
    for reason in summary["reason"].head(5):
        ex = fails[fails["fail_reason"]==reason].head(3)
        cols = ["id","gold_answer","model_answer","judge_support","tokens_out","model_raw"]
        cols = [c for c in cols if c in ex.columns]
        print(f"\n-- {reason} --")
        with pd.option_context('display.max_colwidth', 160):
            print(ex[cols].to_string(index=False))

    # optional: save a filtered csv of errors
    out = RESULTS_CSV.with_name("cot_errors.csv")
    fails.to_csv(out, index=False)
    print(f"\nSaved detailed CoT errors to: {out}")

if __name__ == "__main__":
    main()
