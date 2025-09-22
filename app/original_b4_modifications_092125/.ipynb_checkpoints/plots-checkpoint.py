# app/plots.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from app.config import cfg


def _savefig(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    plots_dir = cfg.paths.plots_dir
    summary   = pd.read_csv(cfg.paths.summary_csv)   # per-method agg
    results   = pd.read_csv(cfg.paths.results_csv)   # per-row

    # # --- Plot 1: Accuracy bar (EM %) ---
    # plt.figure(figsize=(4.2, 3.2))
    # bars = plt.bar(summary["method"], (summary["em"] * 100).round(1))
    # plt.ylabel("EM (%)")
    # plt.title("Clinical QA Accuracy — CoT vs Direct")
    # try: plt.bar_label(bars, fmt="%.1f")
    # except Exception: pass
    # _savefig(os.path.join(plots_dir, "accuracy_bar.png"))

    # --- Plot 1: Accuracy bar (EM %) ---
    plt.figure(figsize=(4.2, 3.2))
    bars = plt.bar(summary["method"], (summary["em"] * 100).round(1))
    plt.ylabel("EM (%)")
    plt.title("Clinical QA Accuracy — CoT vs Direct")

    # Add labels with padding
    try:
        plt.bar_label(bars, fmt="%.1f", label_type="edge", padding=4)
    except Exception:
        pass

    # Add a bit more space above bars
    ymax = (summary["em"] * 100).max()
    plt.ylim(0, ymax + 10)

    _savefig(os.path.join(plots_dir, "accuracy_bar.png"))

    # --- Plot 2: Token cost bar (median tokens_out) ---
    plt.figure(figsize=(4.2, 3.2))
    bars = plt.bar(summary["method"], summary["tokens_out"])
    plt.ylabel("Median tokens (out)")
    plt.title("Token Cost — CoT vs Direct")
    try: plt.bar_label(bars, fmt="%.0f")
    except Exception: pass
    _savefig(os.path.join(plots_dir, "tokens_bar.png"))

    # --- Plot 3: Accuracy vs Token Frontier (scatter + method means) ---
    plt.figure(figsize=(5.2, 4.0))
    for m, dfm in results.groupby("method"):
        x = dfm["tokens_out"].values
        y = (dfm["correct_em"].values * 100.0)
        plt.scatter(x, y, alpha=0.28, s=14, label=f"{m} (rows)")
    means = results.assign(em_pct=results["correct_em"] * 100).groupby("method").agg(
        mean_tokens=("tokens_out", "mean"),
        mean_em=("em_pct", "mean"),
    )
    plt.scatter(means["mean_tokens"], means["mean_em"], s=140, marker="D", label="method means")
    for m, row in means.iterrows():
        plt.annotate(m, (row["mean_tokens"], row["mean_em"]), xytext=(6, 6),
                     textcoords="offset points", fontsize=8)
    plt.xlabel("Tokens (out)")
    plt.ylabel("EM (%)")
    plt.title("Accuracy vs Token Cost (Pareto Frontier)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, loc="lower right")
    _savefig(os.path.join(plots_dir, "accuracy_vs_tokens.png"))

    # --- Plot 4: Latency p50 / p95 (side-by-side bars) ---
    methods = summary["method"].tolist()
    x = np.arange(len(methods))
    width = 0.36
    plt.figure(figsize=(5.2, 3.4))
    p50 = plt.bar(x - width/2, summary["latency_p50"], width, label="p50")
    p95 = plt.bar(x + width/2, summary["latency_p95"], width, label="p95")
    plt.xticks(x, methods)
    plt.ylabel("Latency (ms)")
    plt.title("Latency by Method (p50 / p95)")
    try:
        plt.bar_label(p50, fmt="%.0f")
        plt.bar_label(p95, fmt="%.0f")
    except Exception:
        pass
    plt.legend()
    _savefig(os.path.join(plots_dir, "latency_p50_p95.png"))

    # --- Optional: Groundedness vs Hallucination (grouped bars) ---
    if {"grounded", "hallucinated"}.issubset(set(results.columns)):
        rates = (results.groupby("method")[["grounded", "hallucinated"]].mean() * 100).reset_index()
        plt.figure(figsize=(5.4, 3.6))
        bw = 0.36
        x = np.arange(len(rates["method"]))
        g = plt.bar(x - bw/2, rates["grounded"], bw, label="Grounded (%)")
        h = plt.bar(x + bw/2, rates["hallucinated"], bw, label="Hallucinated (%)")
        plt.xticks(x, rates["method"])
        plt.ylabel("Rate (%)")
        plt.title("Groundedness vs Hallucination")
        try:
            plt.bar_label(g, fmt="%.1f")
            plt.bar_label(h, fmt="%.1f")
        except Exception:
            pass
        plt.legend()
        _savefig(os.path.join(plots_dir, "grounded_vs_hallucination.png"))

    # --- Optional: Reasoning Coverage (CoT only) ---
    # Use 'reasoning_coverage' in results if present; else summary column 'reasoning_cov'
    if "reasoning_coverage" in results.columns:
        cov = (results[results["method"] == "cot"]["reasoning_coverage"].mean() * 100.0) if "cot" in results["method"].unique() else 0.0
    elif "reasoning_cov" in summary.columns:
        cov = float(summary.loc[summary["method"] == "cot", "reasoning_cov"].values[0]) * 100.0 if "cot" in summary["method"].values else 0.0
    else:
        cov = None

    if cov is not None:
        plt.figure(figsize=(3.6, 3.2))
        bars = plt.bar(["CoT"], [cov])
        plt.ylabel("Coverage (%)")
        plt.title("Reasoning Coverage (CoT)")
        try: plt.bar_label(bars, fmt="%.1f")
        except Exception: pass
        _savefig(os.path.join(plots_dir, "reasoning_coverage.png"))

    print("[done] plots in", plots_dir)


if __name__ == "__main__":
    main()
