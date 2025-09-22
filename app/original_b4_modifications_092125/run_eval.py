# app/run_eval.py
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from app.config import cfg
from app.prompt import direct_prompt, cot_prompt
from app.metrics import (
    exact_match, f1_tokens, grounded, abstained, hallucinated, token_guess, Timer
)
from app.load_model import call_model

# -------- helpers --------


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


# # Accept the four allowed labels anywhere we look
# _LABEL_RE = re.compile(r"\b(yes|no|maybe|not in context)\b", re.IGNORECASE)

# def _canonicalize(s: str) -> str:
#     """Normalize to exactly one of: yes|no|maybe|not in context."""
#     if not s:
#         return ""
#     s = s.strip().lower()
#     if "not in context" in s or s in {"nic", "none", "na", "n/a"}:
#         return "not in context"
#     if s.startswith("yes"):   return "yes"
#     if s.startswith("no") and s != "na":  return "no"
#     if s.startswith("maybe"): return "maybe"
#     return s if s in {"yes", "no", "maybe", "not in context"} else ""

# def _extract_label(raw: str) -> str:
#     """
#     Try hard to get a clean label from the model continuation:
#     1) Prefer text after 'Final Answer:' up to newline/bracket/period.
#     2) If missing, search entire continuation for any of the four labels.
#     3) Fallback -> 'not in context'.
#     """
#     if not raw:
#         return "not in context"

#     m = re.search(r"final\s*answer\s*:\s*([^\n\[\]\.]+)", raw, re.IGNORECASE)
#     if m:
#         cand = _canonicalize(m.group(1))
#         if cand:
#             return cand

#     m = _LABEL_RE.search(raw)
#     if m:
#         return _canonicalize(m.group(1))

#     return "not in context"

# (Optional) only used if you set cot_self_consistency_k > 1
def _vote_majority(labels):
    return max(set(labels), key=labels.count) if labels else "not in context"


def main():
    # paths
    results_csv = cfg.paths.results_csv
    summary_csv = cfg.paths.summary_csv
    Path(results_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)

    # data
    df = pd.read_csv(cfg.paths.data_csv)  # id,question,context_id,context,gold_answer,has_answer_in_context
    # 9/19
    # add this line to take only 500
    df = df.sample(n=100, random_state=42).reset_index(drop=True)
    # END
    
    # experiment config
    methods        = cfg.experiment.methods
    force_cite     = cfg.experiment.force_citation
    abstain_phrase = cfg.experiment.abstain_phrase

    # inference config (split token budgets with fallback)
    max_new_direct = getattr(cfg.inference, "max_new_tokens_direct", None)
    max_new_cot    = getattr(cfg.inference, "max_new_tokens_cot", None)
    fallback_max   = getattr(cfg.inference, "max_new_tokens", 128)
    max_new_direct = max_new_direct or fallback_max
    max_new_cot    = max_new_cot or fallback_max

    t_direct = cfg.inference.temperature_direct
    t_cot    = cfg.inference.temperature_cot
    k_sc     = int(getattr(cfg.inference, "cot_self_consistency_k", 0) or 0)

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        gold_label = _canon(str(r.gold_answer))

        for method in methods:
            if method == "direct":
                prompt  = direct_prompt(r.context_id, r.context, r.question, force_cite, abstain_phrase)
                prompts = [prompt]
                temps   = [t_direct]
                max_tok = [max_new_direct]
            else:
                base    = cot_prompt(r.context_id, r.context, r.question, force_cite, abstain_phrase)
                k       = max(1, k_sc)
                prompts = [base] * k
                temps   = [t_cot] * k
                max_tok = [max_new_cot] * k

            outs, lats = [], []
            for p, temp, mx in zip(prompts, temps, max_tok):
                with Timer() as t:
                    # call_model returns ONLY the generated continuation
                    outs.append(call_model(p, mx, temp))
                lats.append(t.ms)

            # choose label (self-consistency if enabled)
            if len(outs) == 1:
                raw = outs[0] or ""
                label = _extract_label(raw)
            else:
                labels = [_extract_label(o) for o in outs if o]
                label  = _vote_majority(labels)
                raw    = outs[0] if outs else ""

                
            # # Added 9/19 
            # # --- Judge step (only for cot) ---
            # if method == "cot" and raw.strip():
            #     j_prompt = (
            #         "You are a clinical reviewer.\n\n"
            #         f"Context (ID:{r.context_id}):\n{r.context}\n\n"
            #         f"Q: {r.question}\n\n"
            #         f"Candidate: {raw}\n\n"
            #         "Does the Context support this candidate answer? "
            #         "Reply with one token only: yes | no | unclear.\n\n"
            #         "SUPPORT: "
            #     )
            #     support = (call_model(j_prompt, max_new_tokens=8, temperature=0.0) or "").strip().lower()
            #     if support.startswith("no"):
            #         label = "not in context"
            #     elif support.startswith("unclear") and label == "yes":
            #         label = "maybe"
            # # END

            # metrics
            em   = exact_match(label, gold_label)
            f1   = f1_tokens(label, gold_label)
            grd  = grounded(raw, r.context_id) if force_cite else 0   # citation check in raw text
            absn = int(label == "not in context")
            hall = hallucinated(em, grd, int(r.has_answer_in_context))
            tok  = token_guess(raw)                                   # token cost of full output

            rows.append({
                "id": r.id,
                "method": method,
                "gold_answer": gold_label,    # canonical gold
                "model_raw": raw,             # full continuation text (unaltered)
                "model_answer": label,        # canonical label
                "correct_em": em,
                "f1": f1,
                "grounded": grd,
                "abstained": absn,
                "hallucinated": hall,
                "tokens_out": tok,
                "latency_ms": sum(lats)//max(1, len(lats)),
            })

    res = pd.DataFrame(rows)
    res.to_csv(results_csv, index=False)

    summary = (
        res.groupby("method")
          .agg(
              em=("correct_em","mean"),
              f1=("f1","mean"),
              grounded=("grounded","mean"),
              abstained=("abstained","mean"),
              hallucinated=("hallucinated","mean"),
              tokens_out=("tokens_out","median"),
              latency_p50=("latency_ms","median"),
              latency_p95=("latency_ms", lambda s: int(s.quantile(0.95))),
          )
          .reset_index()
    )
    summary.to_csv(summary_csv, index=False)
    print("[done] wrote:", results_csv, "and", summary_csv)


if __name__ == "__main__":
    main()



# # app/run_eval.py
# import re
# from pathlib import Path
# import pandas as pd
# from tqdm import tqdm

# from app.config import cfg
# from app.prompt import direct_prompt, cot_prompt
# from app.metrics import (
#     exact_match, f1_tokens, grounded, abstained, hallucinated, token_guess, Timer
# )
# from app.load_model import call_model

# # -------- helpers --------

# # Accept the four allowed labels anywhere we look
# _LABEL_RE = re.compile(r"\b(yes|no|maybe|not in context)\b", re.IGNORECASE)

# def _canonicalize(s: str) -> str:
#     """Normalize to exactly one of: yes|no|maybe|not in context."""
#     if not s:
#         return ""
#     s = s.strip().lower()
#     if "not in context" in s or s in {"nic", "none", "na", "n/a"}:
#         return "not in context"
#     if s.startswith("yes"):   return "yes"
#     if s.startswith("no") and s != "na":  return "no"
#     if s.startswith("maybe"): return "maybe"
#     # if it already matches exactly, keep; else empty (we'll handle fallback)
#     return s if s in {"yes", "no", "maybe", "not in context"} else ""

# def _extract_label(raw: str) -> str:
#     """
#     Try hard to get a clean label from the model continuation:
#     1) Prefer text after 'Final Answer:' up to newline/bracket/period.
#     2) If missing, search entire continuation for any of the four labels.
#     3) Fallback -> 'not in context'.
#     """
#     if not raw:
#         return "not in context"

#     # Prefer what's after 'Final Answer:'
#     m = re.search(r"final\s*answer\s*:\s*([^\n\[\]\.]+)", raw, re.IGNORECASE)
#     if m:
#         cand = _canonicalize(m.group(1))
#         if cand:
#             return cand

#     # Else search anywhere
#     m = _LABEL_RE.search(raw)
#     if m:
#         return _canonicalize(m.group(1))

#     return "not in context"


# def _ensure_cot_reasoning(raw: str) -> str:
#     """
#     If the model skipped 'Reasoning:' entirely, tag it so it's visible in CSV.
#     Does not change scoring; purely for inspection.
#     """
#     if not raw:
#         return raw
#     if "Reasoning:" not in raw and "reasoning:" not in raw:
#         return "Reasoning: (model skipped reasoning)\n" + raw
#     return raw


# # (Optional) only used if you set cot_self_consistency_k > 1
# def _vote_majority(labels):
#     return max(set(labels), key=labels.count) if labels else "not in context"


# def main():
#     # paths
#     results_csv = cfg.paths.results_csv
#     summary_csv = cfg.paths.summary_csv
#     Path(results_csv).parent.mkdir(parents=True, exist_ok=True)
#     Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)

#     # data
#     df = pd.read_csv(cfg.paths.data_csv)  # id,question,context_id,context,gold_answer,has_answer_in_context

#     # experiment config
#     methods        = cfg.experiment.methods
#     force_cite     = cfg.experiment.force_citation
#     abstain_phrase = cfg.experiment.abstain_phrase

#     # inference config (split token budgets with fallback)
#     max_new_direct = getattr(cfg.inference, "max_new_tokens_direct", None)
#     max_new_cot    = getattr(cfg.inference, "max_new_tokens_cot", None)
#     fallback_max   = getattr(cfg.inference, "max_new_tokens", 128)
#     max_new_direct = max_new_direct or fallback_max
#     max_new_cot    = max_new_cot or fallback_max

#     t_direct = cfg.inference.temperature_direct
#     t_cot    = cfg.inference.temperature_cot
#     k_sc     = int(getattr(cfg.inference, "cot_self_consistency_k", 0) or 0)

#     rows = []
#     for _, r in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
#         gold_label = _canonicalize(str(r.gold_answer))

#         for method in methods:
#             if method == "direct":
#                 prompt  = direct_prompt(r.context_id, r.context, r.question, force_cite, abstain_phrase)
#                 prompts = [prompt]
#                 temps   = [t_direct]
#                 max_tok = [max_new_direct]
#             else:
#                 base    = cot_prompt(r.context_id, r.context, r.question, force_cite, abstain_phrase)
#                 k       = max(1, k_sc)
#                 prompts = [base] * k
#                 temps   = [t_cot] * k
#                 max_tok = [max_new_cot] * k

#             outs, lats = [], []
#             for p, temp, mx in zip(prompts, temps, max_tok):
#                 with Timer() as t:
#                     # call_model returns ONLY the generated continuation
#                     outs.append(call_model(p, mx, temp))
#                 lats.append(t.ms)

#             # choose label (self-consistency if enabled)
#             if len(outs) == 1:
#                 raw = outs[0] or ""
#                 if method == "cot":
#                     raw = _ensure_cot_reasoning(raw)
#                 label = _extract_label(raw)
#             else:
#                 labels = [_extract_label(o) for o in outs if o]
#                 label  = _vote_majority(labels)
#                 raw    = outs[0] if outs else ""
#                 if method == "cot":
#                     raw = _ensure_cot_reasoning(raw)

#             # metrics
#             em   = exact_match(label, gold_label)
#             f1   = f1_tokens(label, gold_label)
#             grd  = grounded(raw, r.context_id) if force_cite else 0   # check citation in raw text
#             absn = int(label == "not in context")
#             hall = hallucinated(em, grd, int(r.has_answer_in_context))
#             tok  = token_guess(raw)                                   # token cost of full output

#             rows.append({
#                 "id": r.id,
#                 "method": method,
#                 "gold_answer": gold_label,    # canonical gold
#                 "model_raw": raw,             # full continuation text
#                 "model_answer": label,        # canonical label
#                 "correct_em": em,
#                 "f1": f1,
#                 "grounded": grd,
#                 "abstained": absn,
#                 "hallucinated": hall,
#                 "tokens_out": tok,
#                 "latency_ms": sum(lats)//max(1, len(lats)),
#             })

#     res = pd.DataFrame(rows)
#     res.to_csv(results_csv, index=False)

#     summary = (
#         res.groupby("method")
#           .agg(
#               em=("correct_em","mean"),
#               f1=("f1","mean"),
#               grounded=("grounded","mean"),
#               abstained=("abstained","mean"),
#               hallucinated=("hallucinated","mean"),
#               tokens_out=("tokens_out","median"),
#               latency_p50=("latency_ms","median"),
#               latency_p95=("latency_ms", lambda s: int(s.quantile(0.95))),
#           )
#           .reset_index()
#     )
#     summary.to_csv(summary_csv, index=False)
#     print("[done] wrote:", results_csv, "and", summary_csv)


# if __name__ == "__main__":
#     main()




# # app/run_eval.py
# import re
# from pathlib import Path
# import pandas as pd
# from tqdm import tqdm

# from app.config import cfg
# from app.prompt import direct_prompt, cot_prompt
# from app.metrics import (
#     exact_match, f1_tokens, grounded, abstained, hallucinated, token_guess, Timer
# )
# from app.load_model import call_model

# # ---- helpers ----
# _LABEL_RE = re.compile(r"\b(yes|no|maybe|not in context)\b", re.IGNORECASE)

# def extract_label(text: str) -> str:
#     """Pull yes|no|maybe|not in context from 'Final Answer:' (fallback: anywhere)."""
#     if not text:
#         return ""
#     s = text.split("Final Answer:")[-1].strip()
#     m = _LABEL_RE.search(s) or _LABEL_RE.search(text)
#     return m.group(1).lower() if m else (s.split()[0].lower() if s else "")

# def canonicalize(s: str) -> str:
#     """Normalize to exactly one of: yes|no|maybe|not in context."""
#     if not s: return ""
#     s = s.strip().lower()
#     if "not in context" in s or s in {"nic", "none", "na", "n/a"}:
#         return "not in context"
#     if s.startswith("y"): return "yes"
#     if s.startswith("n") and s != "na": return "no"
#     if s.startswith("m"): return "maybe"
#     return s if s in {"yes","no","maybe","not in context"} else s

# # (Optional) only used if you set cot_self_consistency_k > 1
# def vote_majority(labels):
#     return max(set(labels), key=labels.count) if labels else ""

# def main():
#     # paths
#     results_csv = cfg.paths.results_csv
#     summary_csv = cfg.paths.summary_csv
#     Path(results_csv).parent.mkdir(parents=True, exist_ok=True)
#     Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)

#     # data
#     df = pd.read_csv(cfg.paths.data_csv)  # id,question,context_id,context,gold_answer,has_answer_in_context

#     # experiment config
#     methods        = cfg.experiment.methods
#     force_cite     = cfg.experiment.force_citation
#     abstain_phrase = cfg.experiment.abstain_phrase

#     # inference config (with safe fallbacks)
#     max_new_direct = getattr(cfg.inference, "max_new_tokens_direct", None)
#     max_new_cot    = getattr(cfg.inference, "max_new_tokens_cot", None)
#     # fall back to single max_new_tokens if the split keys aren’t present
#     fallback_max   = cfg.inference.max_new_tokens if hasattr(cfg.inference, "max_new_tokens") else 128
#     max_new_direct = max_new_direct or fallback_max
#     max_new_cot    = max_new_cot or fallback_max

#     t_direct       = cfg.inference.temperature_direct
#     t_cot          = cfg.inference.temperature_cot
#     k_sc           = int(getattr(cfg.inference, "cot_self_consistency_k", 0) or 0)

#     rows = []
#     for _, r in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
#         gold_label = canonicalize(str(r.gold_answer))

#         for method in methods:
#             if method == "direct":
#                 prompt  = direct_prompt(r.context_id, r.context, r.question, force_cite, abstain_phrase)
#                 prompts = [prompt]
#                 temps   = [t_direct]
#                 max_tok = [max_new_direct]
#             else:
#                 base    = cot_prompt(r.context_id, r.context, r.question, force_cite, abstain_phrase)
#                 k       = max(1, k_sc)
#                 prompts = [base] * k
#                 temps   = [t_cot] * k
#                 max_tok = [max_new_cot] * k

#             outs, lats = [], []
#             for p, temp, mx in zip(prompts, temps, max_tok):
#                 with Timer() as t:
#                     # call_model returns ONLY the generated continuation
#                     outs.append(call_model(p, mx, temp))
#                 lats.append(t.ms)

#             # choose final label (self-consistency if enabled)
#             if len(outs) == 1:
#                 raw = outs[0] or ""
#                 label = canonicalize(extract_label(raw))
#             else:
#                 labels = [canonicalize(extract_label(o)) for o in outs if o]
#                 label  = vote_majority(labels) if labels else ""
#                 raw    = outs[0] if outs else ""

#             # metrics
#             em   = exact_match(label, gold_label)
#             f1   = f1_tokens(label, gold_label)
#             grd  = grounded(raw, r.context_id) if force_cite else 0   # check citation in raw text
#             absn = int(label == "not in context")
#             hall = hallucinated(em, grd, int(r.has_answer_in_context))
#             tok  = token_guess(raw)                                   # token cost of full output

#             rows.append({
#                 "id": r.id,
#                 "method": method,
#                 "gold_answer": gold_label,    # canonical gold
#                 "model_raw": raw,             # full continuation text
#                 "model_answer": label,        # canonical label
#                 "correct_em": em,
#                 "f1": f1,
#                 "grounded": grd,
#                 "abstained": absn,
#                 "hallucinated": hall,
#                 "tokens_out": tok,
#                 "latency_ms": sum(lats)//max(1, len(lats)),
#             })

#     res = pd.DataFrame(rows)
#     res.to_csv(results_csv, index=False)

#     summary = (
#         res.groupby("method")
#           .agg(
#               em=("correct_em","mean"),
#               f1=("f1","mean"),
#               grounded=("grounded","mean"),
#               abstained=("abstained","mean"),
#               hallucinated=("hallucinated","mean"),
#               tokens_out=("tokens_out","median"),
#               latency_p50=("latency_ms","median"),
#               latency_p95=("latency_ms", lambda s: int(s.quantile(0.95))),
#           )
#           .reset_index()
#     )
#     summary.to_csv(summary_csv, index=False)
#     print("[done] wrote:", results_csv, "and", summary_csv)

# if __name__ == "__main__":
#     main()




# # app/run_eval.py
# import re
# from pathlib import Path
# import pandas as pd
# from tqdm import tqdm

# from app.config import cfg
# from app.prompt import direct_prompt, cot_prompt
# from app.metrics import exact_match, f1_tokens, grounded, abstained, hallucinated, token_guess, Timer
# from app.load_model import call_model

# # ---- helpers ----
# _label_pat = re.compile(r"\b(yes|no|maybe|not in context)\b", re.IGNORECASE)

# def extract_label(text: str) -> str:
#     """Pull yes|no|maybe|not in context from 'Final Answer:' line (fallback: anywhere)."""
#     if not text:
#         return ""
#     s = text.split("Final Answer:")[-1].strip()
#     m = _label_pat.search(s) or _label_pat.search(text)
#     return m.group(1).lower() if m else (s.split()[0].lower() if s else "")

# def canonicalize(s: str) -> str:
#     """Normalize to exactly one of: yes|no|maybe|not in context."""
#     if not s: return ""
#     s = s.strip().lower()
#     if "not in context" in s or s in {"nic", "none", "na", "n/a"}:
#         return "not in context"
#     if s.startswith("y"): return "yes"
#     if s.startswith("n") and s != "na": return "no"
#     if s.startswith("m"): return "maybe"
#     # last resort: keep only exact match if already valid
#     return s if s in {"yes","no","maybe","not in context"} else s

# # (Optional) only needed if you enable k>1 self-consistency later
# def vote_majority(labels):
#     return max(set(labels), key=labels.count) if labels else ""

# def main():
#     # paths
#     results_csv = cfg.paths.results_csv
#     summary_csv = cfg.paths.summary_csv
#     Path(results_csv).parent.mkdir(parents=True, exist_ok=True)
#     Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)

#     # data
#     df = pd.read_csv(cfg.paths.data_csv)  # id,question,context_id,context,gold_answer,has_answer_in_context

#     # config
#     methods        = cfg.experiment.methods
#     force_cite     = cfg.experiment.force_citation
#     abstain_phrase = cfg.experiment.abstain_phrase
#     max_new        = cfg.inference.max_new_tokens
#     t_direct       = cfg.inference.temperature_direct
#     t_cot          = cfg.inference.temperature_cot
#     k_sc           = int(cfg.inference.cot_self_consistency_k or 0)

#     rows = []
#     for _, r in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
#         gold_label = canonicalize(str(r.gold_answer))
#         for method in methods:
#             if method == "direct":
#                 prompt = direct_prompt(r.context_id, r.context, r.question, force_cite, abstain_phrase)
#                 prompts, temps = [prompt], [t_direct]
#             else:
#                 base = cot_prompt(r.context_id, r.context, r.question, force_cite, abstain_phrase)
#                 k = max(1, k_sc)
#                 prompts, temps = [base] * k, [t_cot] * k

#             # generate(s)
#             outs, lats = [], []
#             for p, temp in zip(prompts, temps):
#                 with Timer() as t:
#                     outs.append(call_model(p, max_new, temp))
#                 lats.append(t.ms)

#             # choose final label
#             if len(outs) == 1:
#                 raw = outs[0] or ""
#                 label = canonicalize(extract_label(raw))
#             else:
#                 labels = [canonicalize(extract_label(o)) for o in outs if o]
#                 label = vote_majority(labels) if labels else ""
#                 raw = outs[0] if outs else ""

#             # metrics (score on canonical labels)
#             em  = exact_match(label, gold_label)
#             f1  = f1_tokens(label, gold_label)
#             grd = grounded(label, r.context_id) if force_cite else 0
#             absn = int(label == "not in context")
#             hall = hallucinated(em, grd, int(r.has_answer_in_context))
#             tok  = token_guess(label)

#             rows.append({
#                 "id": r.id,
#                 "method": method,
#                 "gold_answer": gold_label,        # canonical gold
#                 "model_raw": raw,                 # full text
#                 "model_answer": label,            # canonical label (yes/no/maybe/not in context)
#                 "correct_em": em,
#                 "f1": f1,
#                 "grounded": grd,
#                 "abstained": absn,
#                 "hallucinated": hall,
#                 "tokens_out": tok,
#                 "latency_ms": sum(lats)//max(1, len(lats)),
#             })

#     res = pd.DataFrame(rows)
#     res.to_csv(results_csv, index=False)

#     summary = (
#         res.groupby("method")
#           .agg(
#               em=("correct_em","mean"),
#               f1=("f1","mean"),
#               grounded=("grounded","mean"),
#               abstained=("abstained","mean"),
#               hallucinated=("hallucinated","mean"),
#               tokens_out=("tokens_out","median"),
#               latency_p50=("latency_ms","median"),
#               latency_p95=("latency_ms", lambda s: int(s.quantile(0.95))),
#           )
#           .reset_index()
#     )
#     summary.to_csv(summary_csv, index=False)
#     print("[done] wrote:", results_csv, "and", summary_csv)

# if __name__ == "__main__":
#     main()





# # app/run_eval.py
# import os
# import pandas as pd
# from tqdm import tqdm
# from pathlib import Path

# from app.config import load_config, cfg
# from app.prompt import direct_prompt, cot_prompt
# from app.metrics import exact_match, f1_tokens, grounded, abstained, hallucinated, token_guess, Timer
# from app.load_model import call_model

# # (Optional) keep this only if you later set k>1 for self-consistency
# def vote_majority(final_answers):
#     return max(set(final_answers), key=final_answers.count) if final_answers else ""

# def main():
#     # You can use the global cfg, or reload; both return DotDict
#     c = cfg if cfg else load_config()

#     # paths
#     results_dir = Path(c.paths.results_csv).parent
#     plots_dir   = Path(c.paths.plots_dir)
#     results_dir.mkdir(parents=True, exist_ok=True)
#     plots_dir.mkdir(parents=True, exist_ok=True)

#     # data
#     df = pd.read_csv(c.paths.data_csv)  # id,question,context_id,context,gold_answer,has_answer_in_context

#     # config (NOTE: levels changed)
#     methods         = c.experiment.methods
#     force_citation  = c.experiment.force_citation
#     abstain_phrase  = c.experiment.abstain_phrase
#     max_new_tokens  = c.inference.max_new_tokens
#     temp_direct     = c.inference.temperature_direct
#     temp_cot        = c.inference.temperature_cot
#     k_sc            = int(c.inference.cot_self_consistency_k or 0)

#     rows = []
#     # added 9/18
#     for _, r in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
#     # END
#     # for _, r in df.iterrows():
#         for method in methods:
#             if method == "direct":
#                 prompt = direct_prompt(r.context_id, r.context, r.question, force_citation, abstain_phrase)
#                 prompts = [prompt]
#                 temps   = [temp_direct]
#             else:
#                 base = cot_prompt(r.context_id, r.context, r.question, force_citation, abstain_phrase)
#                 k = max(1, k_sc)
#                 prompts = [base] * k
#                 temps   = [temp_cot] * k

#             answers, latencies = [], []
#             for p, tval in zip(prompts, temps):
#                 with Timer() as t:
#                     # NOTE: load_model.call_model takes (prompt, max_new_tokens, temperature)
#                     ans = call_model(p, max_new_tokens, tval)
#                 answers.append(ans); latencies.append(t.ms)

#             # pick final (no self-consistency unless k>1)
#             if len(answers) == 1:
#                 final = (answers[0] or "").split("Final Answer:")[-1].strip()
#             else:
#                 finals = [(a or "").split("Final Answer:")[-1].strip() for a in answers if a]
#                 final = vote_majority(finals) if finals else ""

#             em  = exact_match(final, r.gold_answer)
#             f1  = f1_tokens(final, r.gold_answer)
#             grd = grounded(final, r.context_id) if force_citation else 0
#             absn = abstained(final, abstain_phrase)
#             hall = hallucinated(em, grd, int(r.has_answer_in_context))
#             tok_out = token_guess(final)

#             rows.append({
#                 "id": r.id, "method": method,
#                 "gold_answer": r.gold_answer, "model_answer": final,
#                 "correct_em": em, "f1": f1, "grounded": grd,
#                 "abstained": absn, "hallucinated": hall,
#                 "tokens_out": tok_out, "latency_ms": sum(latencies)//max(1,len(latencies))
#             })

#     res = pd.DataFrame(rows)
#     res.to_csv(c.paths.results_csv, index=False)

#     summary = (
#         res.groupby("method")
#           .agg(em=("correct_em","mean"),
#                f1=("f1","mean"),
#                grounded=("grounded","mean"),
#                abstained=("abstained","mean"),
#                hallucinated=("hallucinated","mean"),
#                tokens_out=("tokens_out","median"),
#                latency_p50=("latency_ms","median"),
#                latency_p95=("latency_ms", lambda s: int(s.quantile(0.95)))))
#     summary.to_csv(c.paths.summary_csv)

#     print("[done] wrote:", c.paths.results_csv, "and", c.paths.summary_csv)

# if __name__ == "__main__":
#     main()
