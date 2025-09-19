import csv, os, random
import pandas as pd
from config import load_config
from prompt import direct_prompt, cot_prompt
from metrics import exact_match, f1_tokens, grounded, abstained, hallucinated, token_guess, Timer
from app.load_model import call_model 

def vote_majority(final_answers):  # for self-consistency k>1
    return max(set(final_answers), key=final_answers.count)

def main():
    cfg = load_config()
    os.makedirs(os.path.dirname(cfg.paths.results_csv), exist_ok=True)
    os.makedirs(cfg.paths.plots_dir, exist_ok=True)

    df = pd.read_csv(cfg.paths.data_csv)  # cols: id,question,context_id,context,gold_answer,has_answer_in_context

    rows = []
    for _, r in df.iterrows():
        for method in cfg.methods:
            if method == "direct":
                prompt = direct_prompt(r.context_id, r.context, r.question, cfg.force_citation, cfg.abstain_phrase)
                prompts = [prompt]; k = 1
            else:
                base = cot_prompt(r.context_id, r.context, r.question, cfg.force_citation, cfg.abstain_phrase)
                k = max(1, cfg.k_sc)
                prompts = [base]*k

            answers, latencies = [], []
            for p in prompts:
                with Timer() as t:
                    ans = call_model(p, cfg.primary, cfg.temperature, cfg.max_new_tokens)
                answers.append(ans); latencies.append(t.ms)

            final = answers[0] if k == 1 else vote_majority([a.splitlines()[-1] if a else "" for a in answers])

            em  = exact_match(final, r.gold_answer)
            f1  = f1_tokens(final, r.gold_answer)
            grd = grounded(final, r.context_id) if cfg.force_citation else 0
            absn = abstained(final, cfg.abstain_phrase)
            hall = hallucinated(em, grd, int(r.has_answer_in_context))
            tok_out = token_guess(final)

            rows.append({
                "id": r.id, "method": method, "gold_answer": r.gold_answer, "model_answer": final,
                "correct_em": em, "f1": f1, "grounded": grd, "abstained": absn, "hallucinated": hall,
                "tokens_out": tok_out, "latency_ms": sum(latencies)//len(latencies)
            })

    res = pd.DataFrame(rows)
    res.to_csv(cfg.paths.results_csv, index=False)

    summary = (res.groupby("method")
                 .agg(em=("correct_em","mean"),
                      f1=("f1","mean"),
                      grounded=("grounded","mean"),
                      abstained=("abstained","mean"),
                      hallucinated=("hallucinated","mean"),
                      tokens_out=("tokens_out","median"),
                      latency_p50=("latency_ms","median"),
                      latency_p95=("latency_ms", lambda s: int(s.quantile(0.95)))))
    summary.to_csv(cfg.paths.summary_csv)

    print("[done] wrote:", cfg.paths.results_csv, "and", cfg.paths.summary_csv)

if __name__ == "__main__":
    main()
