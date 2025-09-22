# app/make_metrics_csv.py
from pathlib import Path
import numpy as np
import pandas as pd
from app.config import cfg

def _wc(s: str) -> int:
    if not isinstance(s, str) or not s.strip():
        return 0
    return len(s.split())

def _mcnemar_counts(df: pd.DataFrame) -> dict:
    # restrict to ids that have both methods
    both_ids = (
        df.groupby("id")["method"].nunique().reset_index()
          .query("method == 2")["id"]
    )
    piv = (
        df[df["id"].isin(both_ids)][["id","method","correct_em"]]
          .pivot(index="id", columns="method", values="correct_em")
          .rename(columns={"cot":"cot", "direct":"direct"})
          .dropna()
          .astype(int)
    )
    a = int(((piv["cot"]==1) & (piv["direct"]==1)).sum())  # both correct
    b = int(((piv["cot"]==1) & (piv["direct"]==0)).sum())  # CoT only
    c = int(((piv["cot"]==0) & (piv["direct"]==1)).sum())  # Direct only
    d = int(((piv["cot"]==0) & (piv["direct"]==0)).sum())  # both wrong
    return {"a_both_correct": a, "b_cot_only": b, "c_direct_only": c, "d_both_wrong": d}

def _mcnemar_stats(b: int, c: int):
    # Continuity-corrected chi-square (no SciPy dependency)
    import math
    if b + c == 0:
        return 0.0, 1.0
    stat = ((abs(b - c) - 1.0)**2) / (b + c)
    # chi2 survival with df=1 ≈ exp(-stat/2)
    p = math.exp(-stat/2.0)
    return float(stat), float(p)

def main():
    results_csv = Path(cfg.paths.results_csv)
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)

    res = pd.read_csv(results_csv)
    res["method"] = res["method"].str.lower()
    for col in ("gold_answer","model_answer"):
        if col in res.columns:
            res[col] = res[col].astype(str).str.strip().str.lower()
    if "model_raw" not in res.columns:
        res["model_raw"] = ""

    # Write into the same run folder as plots (…/outputs/<run>/)
    run_dir = Path(cfg.paths.plots_dir).parent
    out_dir = run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- method_metrics.csv ----------
    rows = []
    for m, dfm in res.groupby("method"):
        em = float(dfm["correct_em"].mean()) if "correct_em" in dfm else np.nan

        def count_right(label):
            return int(((dfm["gold_answer"]==label) & (dfm["correct_em"]==1)).sum())

        yes_right   = count_right("yes")
        no_right    = count_right("no")
        maybe_right = count_right("maybe")

        lat_mean = float(dfm.get("latency_ms", pd.Series(dtype=float)).mean())
        tok_mean = float(dfm.get("tokens_out", pd.Series(dtype=float)).mean())
        word_avg = float(dfm["model_raw"].astype(str).map(_wc).mean())

        rows.append({
            "method": m,
            "em": round(em, 4),
            "yes_correct": yes_right,
            "no_correct": no_right,
            "maybe_correct": maybe_right,
            "latency_ms_mean": round(lat_mean, 1) if not np.isnan(lat_mean) else np.nan,
            "tokens_out_mean": round(tok_mean, 1) if not np.isnan(tok_mean) else np.nan,
            "model_raw_word_avg": round(word_avg, 1),
        })
    pd.DataFrame(rows).to_csv(out_dir / "method_metrics.csv", index=False)

    # ---------- McNemar CSVs ----------
    counts = _mcnemar_counts(res)
    pd.DataFrame([counts]).to_csv(out_dir / "mcnemar_counts.csv", index=False)

    stat, p = _mcnemar_stats(counts["b_cot_only"], counts["c_direct_only"])
    pd.DataFrame([{
        "b_cot_only": counts["b_cot_only"],
        "c_direct_only": counts["c_direct_only"],
        "statistic_chi2_cc": round(stat, 3),
        "p_value": f"{p:.6g}",
    }]).to_csv(out_dir / "mcnemar_stats.csv", index=False)

    print(f"[ok] wrote:\n  - {out_dir/'method_metrics.csv'}\n  - {out_dir/'mcnemar_counts.csv'}\n  - {out_dir/'mcnemar_stats.csv'}")

if __name__ == "__main__":
    main()
