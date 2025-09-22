# app/make_tables.py
from pathlib import Path
import pandas as pd
from app.config import cfg

def _markdown_table(df: pd.DataFrame) -> str:
    # Simple GitHub-flavored markdown table builder (no external deps)
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows   = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows) + "\n"

def main():
    run_dir = Path(cfg.paths.plots_dir).parent  # same folder where make_metrics_csv wrote files
    method_csv  = run_dir / "method_metrics.csv"
    counts_csv  = run_dir / "mcnemar_counts.csv"
    stats_csv   = run_dir / "mcnemar_stats.csv"

    if not method_csv.exists():
        raise FileNotFoundError(f"missing: {method_csv} — run `python -m app.make_metrics_csv` first")

    out_dir = run_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Table 1: per-method summary (for README) ----------
    method_df = pd.read_csv(method_csv)
    # reorder & pretty rounding
    nice = method_df.copy()
    if "em" in nice:
        nice["em"] = (nice["em"] * 100).round(1)
        nice.rename(columns={"em":"EM (%)"}, inplace=True)
    for col in ("latency_ms_mean","tokens_out_mean","model_raw_word_avg"):
        if col in nice:
            nice[col] = nice[col].round(1)

    # consistent column order
    want_cols = [
        "method", "EM (%)",
        "yes_correct", "no_correct", "maybe_correct",
        "latency_ms_mean", "tokens_out_mean", "model_raw_word_avg"
    ]
    have_cols = [c for c in want_cols if c in nice.columns]
    nice = nice[have_cols]
    md1 = _markdown_table(nice)
    (out_dir / "README_table_method_summary.md").write_text(md1, encoding="utf-8")

    # ---------- Table 2: McNemar 2×2 + stats ----------
    if counts_csv.exists() and stats_csv.exists():
        counts = pd.read_csv(counts_csv).iloc[0].to_dict()
        stats  = pd.read_csv(stats_csv).iloc[0].to_dict()

        mcnemar_mat = pd.DataFrame({
            "": ["CoT correct", "CoT wrong", "Totals"],
            "Direct correct": [
                counts.get("a_both_correct", 0),
                counts.get("c_direct_only", 0),
                counts.get("a_both_correct", 0) + counts.get("c_direct_only", 0)
            ],
            "Direct wrong": [
                counts.get("b_cot_only", 0),
                counts.get("d_both_wrong", 0),
                counts.get("b_cot_only", 0) + counts.get("d_both_wrong", 0)
            ],
        })
        md2 = _markdown_table(mcnemar_mat)
        md2 += f"\n**McNemar (CC)**: b={int(stats.get('b_cot_only',0))}, c={int(stats.get('c_direct_only',0))}, "\
               f"χ²={stats.get('statistic_chi2_cc',0)}, p={stats.get('p_value','NA')}\n"
        (out_dir / "README_table_mcnemar.md").write_text(md2, encoding="utf-8")

    print(f"[ok] README tables:\n  - {out_dir/'README_table_method_summary.md'}\n  - {out_dir/'README_table_mcnemar.md'}")

if __name__ == "__main__":
    main()
