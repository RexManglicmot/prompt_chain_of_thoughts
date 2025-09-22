# app/plot_2.py
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from app.config import cfg

METHOD_ORDER = ["direct", "cot"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["#d1eb81", "#8f0583", "#2ca02c", "#d62728"])


# ---------- helpers ----------
def _savefig(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=1.2)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

def _ensure_method_order(df, col="method"):
    present = [m for m in METHOD_ORDER if m in df[col].unique().tolist()]
    return df[df[col].isin(present)], present

def _bar_values(bars, pad_frac=0.06, fmt=None):
    if not bars: return
    heights = [b.get_height() for b in bars]
    ymax = max(heights) if heights else 1
    for b in bars:
        v = b.get_height()
        if fmt:
            text = fmt % v
        else:
            text = f"{v:.1f}" if isinstance(v, float) and v % 1 else f"{int(v)}"
        plt.text(b.get_x() + b.get_width()/2, v + ymax*pad_frac, text,
                 ha="center", va="bottom", fontsize=9)
    plt.ylim(0, ymax * (1 + pad_frac*6))

def _prf(df, labels=("yes","no","maybe")):
    rows = []
    for m, chunk in df.groupby("method"):
        for y in labels:
            tp = ((chunk["gold_answer"]==y) & (chunk["model_answer"]==y)).sum()
            fp = ((chunk["gold_answer"]!=y) & (chunk["model_answer"]==y)).sum()
            fn = ((chunk["gold_answer"]==y) & (chunk["model_answer"]!=y)).sum()
            prec = tp / max(1, tp+fp)
            rec  = tp / max(1, tp+fn)
            f1   = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
            rows.append({"method": m, "label": y, "precision": prec, "recall": rec, "f1": f1})
    return pd.DataFrame(rows)

_reason_re = re.compile(r"reasoning\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
def _reasoning_length_words(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    m = _reason_re.search(text)
    seg = m.group(1) if m else text
    # one line policy: split off any 'Final Answer:' if it appears after
    seg = seg.split("Final Answer:")[0]
    # crude word count
    return len(seg.strip().split())

# ---------- main ----------
def main():
    plots_dir = cfg.paths.plots_dir
    results   = pd.read_csv(cfg.paths.results_csv)
    summary   = pd.read_csv(cfg.paths.summary_csv)

    results, methods = _ensure_method_order(results, "method")
    summary, _ = _ensure_method_order(summary, "method")

    # ===== 0) EM accuracy bar =====
    plt.figure(figsize=(4.8, 3.4))
    em_pct = (summary.set_index("method").loc[methods, "em"] * 100).values
    bars = plt.bar(methods, em_pct, edgecolor="black", linewidth=0.5)
    _bar_values(list(bars), pad_frac=0.06)
    plt.ylabel("EM (%)")
    plt.title("Clinical QA Accuracy — CoT vs Direct")
    _savefig(os.path.join(plots_dir, "accuracy_bar.png"))

    # ===== 1) Latency p50/p95 =====
    x = np.arange(len(methods))
    width = 0.38
    p50_vals = summary.set_index("method").loc[methods, "latency_p50"].tolist()
    p95_vals = summary.set_index("method").loc[methods, "latency_p95"].tolist()
    plt.figure(figsize=(6.6, 4.0))
    b50 = plt.bar(x - width/2, p50_vals, width, label="p50",edgecolor="black", linewidth=0.5)
    b95 = plt.bar(x + width/2, p95_vals, width, label="p95", edgecolor="black", linewidth=0.5)
    plt.xticks(x, methods)
    plt.ylabel("Latency (ms)")
    plt.title("Latency by Method (p50 / p95)")
    tallest = max(p50_vals + p95_vals) if (p50_vals or p95_vals) else 1
    for bars in (b50, b95):
        for b in bars:
            v = b.get_height()
            plt.text(b.get_x()+b.get_width()/2, v + tallest*0.02, f"{int(v)}",
                     ha="center", va="bottom", fontsize=9)
    plt.ylim(0, tallest * 1.14)
    plt.legend()
    _savefig(os.path.join(plots_dir, "latency_p50_p95.png"))

    # ===== 2) Tokens: boxplot per method =====

    # ===== 2) Token cost distribution — box + jitter (match CoT plot style) =====
    plt.figure(figsize=(6.2, 4.0))

    # prepare series per method in fixed order
    data = [results.loc[results["method"] == m, "tokens_out"].dropna().values for m in methods]

    # boxplot (no fliers), consistent styling
    bp = plt.boxplot(data, tick_labels=methods, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_edgecolor("#333333"); patch.set_linewidth(1.0)
    for med in bp['medians']:
        med.set_color("#222222"); med.set_linewidth(1.4)

    # jitter overlay (all orange), same look as reasoning plot
    rng = np.random.default_rng(0)
    for i, g in enumerate(data, start=1):
        if len(g) == 0: 
            continue
        xj = rng.normal(loc=i, scale=0.06, size=len(g))
        plt.scatter(xj, g, s=10, alpha=0.25, color="orange")

    # median labels on the side of each box (left→right order of `methods`)
    for i, g in enumerate(data, start=1):
        if len(g) == 0:
            continue
        med = float(np.median(g))
        plt.text(i + 0.20, med, f"median={int(med)}",
                ha="left", va="center", fontsize=9)

    plt.ylabel("Tokens (out)")
    plt.title("Token Cost Distribution — Direct vs CoT (Boxplot)")
    plt.ylim(bottom=-6)
    plt.tight_layout(pad=2.0)      # increase padding
    _savefig(os.path.join(plots_dir, "tokens_boxplot.png"))


    # plt.figure(figsize=(6.2, 3.8))
    # data = [results[results["method"]==m]["tokens_out"].dropna().values for m in methods]
    # bp = plt.boxplot(data, labels=methods, patch_artist=True, showfliers=False)
    # for patch in bp['boxes']:
    #     patch.set_edgecolor("#333333"); patch.set_linewidth(1.0)
    # for med in bp['medians']:
    #     med.set_color("#222222"); med.set_linewidth(1.4)
    # plt.ylabel("Tokens (out)")
    # plt.title("Token Cost Distribution — Direct vs CoT (Boxplot)")
    # medians = [np.median(d) if len(d) else 0 for d in data]
    # ymax = max([max(d) if len(d) else 1 for d in data]) if data else 1
    # for i, med in enumerate(medians, start=1):
    #     plt.text(i, med + ymax*0.03, f"median={int(med)}", ha="center", va="bottom", fontsize=9)
    # plt.ylim(bottom=0)
    # _savefig(os.path.join(plots_dir, "tokens_boxplot.png"))

    # ===== 3) Per-label F1 =====
    prf = _prf(results)
    prf["f1"] = prf["f1"] * 100
    labels = ["yes","no","maybe"]
    x = np.arange(len(labels))
    bw = 0.36
    plt.figure(figsize=(6.4, 3.9))
    for i, m in enumerate(methods):
        vals = [float(prf[(prf["method"]==m) & (prf["label"]==L)]["f1"].values[0]) for L in labels]
        bars = plt.bar(x + (i-0.5)*bw, vals, bw, label=m,edgecolor="black", linewidth=0.5)
        _bar_values(list(bars), pad_frac=0.06)
    plt.xticks(x, labels)
    plt.ylabel("F1 (%)")
    plt.title("Per-Label F1 — Direct vs CoT")
    plt.legend()
    _savefig(os.path.join(plots_dir, "per_label_f1.png"))

    # ===== 4) Accuracy vs Token (Pareto scatter + method means) =====
    plt.figure(figsize=(6.4, 4.4))
    rng = np.random.default_rng(42)   # reproducible jitter

    for m in methods:
        dfm = results[results["method"]==m]
        xv = dfm["tokens_out"].values
        yv = (dfm["correct_em"].values * 100.0)

        # add jitter only to x
        # jitter = rng.normal(loc=0, scale=1.5, size=len(xv))  
        # xv_j = xv + jitter

        jitter_x = rng.normal(loc=0, scale=1.5, size=len(xv))  
        jitter_y = rng.normal(loc=0, scale=2.0, size=len(yv))  # small vertical spread
        xv_j, yv_j = xv + jitter_x, yv + jitter_y

        # light alpha for rows
        plt.scatter(xv_j, yv_j, alpha=0.28, s=14, label=f"{m} (rows)")

    means = results.assign(em_pct=results["correct_em"]*100).groupby("method").agg(
        mean_tokens=("tokens_out","mean"),
        mean_em=("em_pct","mean"),
    )
    for m in methods:
        if m in means.index:
            px, py = means.loc[m, "mean_tokens"], means.loc[m, "mean_em"]
            plt.scatter([px], [py], s=180, marker="D", label=f"{m} mean")
            plt.annotate(m, (px, py), xytext=(6,6), textcoords="offset points", fontsize=9)
    
    plt.xlabel("Tokens (out)")
    plt.ylabel("EM (%)")
    plt.title("Accuracy vs Token Cost (Pareto Frontier)")
    plt.grid(alpha=0.25)

    handles, labels = plt.gca().get_legend_handles_labels()

    plt.legend(
        handles, labels,
        fontsize=8,
        loc="center right",     # middle right side
        handletextpad=1.0,      # extra space between symbol and text
        labelspacing=1.0,       # extra vertical spacing between entries
        borderaxespad=1.0,      # push away from plot edge
        scatterpoints=1,        # ensure scatter markers don't stack weird
        markerscale=0.5,        # scale small markers so they don’t shrink in legend
        frameon=True,           # put box around
        fancybox=True,
        framealpha=0.8          # semi-transparent background
    )

    # plt.legend(fontsize=8, 
    #            loc="center right",
    #            handletextpad=5.0,  # space between symbol and text
    #            labelspacing=0.6,   # space between rows
    #            borderaxespad=0.8   # space from the plot edge
    #            )
    
    _savefig(os.path.join(plots_dir, "accuracy_vs_tokens.png"))



    # ===== 5) CoT reasoning length vs correctness (box + jitter) =====
    cot = results[results["method"]=="cot"].copy()
    if "model_raw" in cot.columns and not cot.empty:
        cot["reason_len"] = cot["model_raw"].astype(str).map(_reasoning_length_words)
        cot["correct"] = cot["correct_em"].astype(int)
        groups = [cot[cot["correct"]==0]["reason_len"].values,
                cot[cot["correct"]==1]["reason_len"].values]
        labels = ["Wrong", "Correct"]

        plt.figure(figsize=(6.2, 4.0)) #6.2
        bp = plt.boxplot(groups, labels=labels, patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_edgecolor("#333333"); patch.set_linewidth(1.0)
        for med in bp['medians']:
            med.set_color("#222222"); med.set_linewidth(1.4)

        # jitter overlay (all orange)
        rng = np.random.default_rng(0)
        for i, g in enumerate(groups, start=1):
            if len(g) == 0: continue
            xj = rng.normal(loc=i, scale=0.06, size=len(g))
            plt.scatter(xj, g, s=10, alpha=0.25, color="orange")

        # median labels on the side of each box
        meds = [np.median(g) if len(g) else 0 for g in groups]
        for i, med in enumerate(meds, start=1):
            plt.text(i + 0.2, med, f"median={int(med)}",
                 ha="left", va="center", fontsize=9)

        plt.ylabel("Reasoning length (words)")
        plt.title("CoT Reasoning Length vs Correctness")
        plt.ylim(bottom=-3)
        _savefig(os.path.join(plots_dir, "cot_reason_len_vs_correct.png"))


    # # ===== 5) CoT reasoning length vs correctness (box + jitter) =====
    # cot = results[results["method"]=="cot"].copy()
    # if "model_raw" in cot.columns and not cot.empty:
    #     cot["reason_len"] = cot["model_raw"].astype(str).map(_reasoning_length_words)
    #     cot["correct"] = cot["correct_em"].astype(int)
    #     groups = [cot[cot["correct"]==0]["reason_len"].values,
    #             cot[cot["correct"]==1]["reason_len"].values]
    #     labels = ["Wrong", "Correct"]

    #     plt.figure(figsize=(6.2, 4.0))
    #     bp = plt.boxplot(groups, labels=labels, patch_artist=True, showfliers=False)
    #     for patch in bp['boxes']:
    #         patch.set_edgecolor("#333333"); patch.set_linewidth(1.0)
    #     for med in bp['medians']:
    #         med.set_color("#222222"); med.set_linewidth(1.4)

    #     # jitter overlay (all orange)
    #     rng = np.random.default_rng(0)
    #     for i, g in enumerate(groups, start=1):
    #         if len(g) == 0: continue
    #         xj = rng.normal(loc=i, scale=0.06, size=len(g))
    #         plt.scatter(xj, g, s=10, alpha=0.25, color="orange")

    #     # median labels
    #     ymax = max([max(g) if len(g) else 1 for g in groups]) if groups else 1
    #     meds = [np.median(g) if len(g) else 0 for g in groups]
    #     for i, med in enumerate(meds, start=1):
    #         plt.text(i, med + ymax*0.03, f"median={int(med)}",
    #                 ha="center", va="bottom", fontsize=9)

    #     plt.ylabel("Reasoning length (words)")
    #     plt.title("CoT Reasoning Length vs Correctness")
    #     plt.ylim(bottom=0)
    #     _savefig(os.path.join(plots_dir, "cot_reason_len_vs_correct.png"))


    # # ===== 5) CoT reasoning length vs correctness (box + jitter) =====
    # cot = results[results["method"]=="cot"].copy()
    # if "model_raw" in cot.columns and not cot.empty:
    #     cot["reason_len"] = cot["model_raw"].astype(str).map(_reasoning_length_words)
    #     cot["correct"] = cot["correct_em"].astype(int)
    #     groups = [cot[cot["correct"]==0]["reason_len"].values,
    #               cot[cot["correct"]==1]["reason_len"].values]
    #     labels = ["Wrong", "Correct"]

    #     plt.figure(figsize=(6.2, 4.0))
    #     bp = plt.boxplot(groups, labels=labels, patch_artist=True, showfliers=False)
    #     for patch in bp['boxes']:
    #         patch.set_edgecolor("#333333"); patch.set_linewidth(1.0)
    #     for med in bp['medians']:
    #         med.set_color("#222222"); med.set_linewidth(1.4)

    #     # jitter overlay
    #     rng = np.random.default_rng(0)
    #     for i, g in enumerate(groups, start=1):
    #         if len(g) == 0: continue
    #         xj = rng.normal(loc=i, scale=0.06, size=len(g))
    #         plt.scatter(xj, g, s=10, alpha=0.25)

    #     # median labels
    #     ymax = max([max(g) if len(g) else 1 for g in groups]) if groups else 1
    #     meds = [np.median(g) if len(g) else 0 for g in groups]
    #     for i, med in enumerate(meds, start=1):
    #         plt.text(i, med + ymax*0.03, f"median={int(med)}", ha="center", va="bottom", fontsize=9)

    #     plt.ylabel("Reasoning length (words)")
    #     plt.title("CoT Reasoning Length vs Correctness")
    #     plt.ylim(bottom=0)
    #     _savefig(os.path.join(plots_dir, "cot_reason_len_vs_correct.png"))

    # ===== 6) Wins/Losses by gold label (stacked) =====
    # pivot to compare methods per id
    wide = (results
            .pivot_table(index="id",
                         columns="method",
                         values="correct_em",
                         aggfunc="first")
            .reset_index())
    # attach gold labels
    gold = results.drop_duplicates(subset=["id"])[["id","gold_answer"]]
    wide = wide.merge(gold, on="id", how="left")

    # compute wins (cot=1, direct=0) and losses (cot=0, direct=1)
    win = wide[(wide.get("cot")==1) & (wide.get("direct")==0)]
    loss= wide[(wide.get("cot")==0) & (wide.get("direct")==1)]

    win_counts  = win["gold_answer"].value_counts().reindex(["yes","no","maybe"], fill_value=0)
    loss_counts = loss["gold_answer"].value_counts().reindex(["yes","no","maybe"], fill_value=0)

    plt.figure(figsize=(6.0, 3.8))
    idx = np.arange(3)
    b1 = plt.bar(idx, win_counts.values, label="CoT wins",edgecolor="black", linewidth=0.5)
    b2 = plt.bar(idx, loss_counts.values, bottom=win_counts.values, label="CoT losses",edgecolor="black", linewidth=0.5)
    plt.xticks(idx, ["yes","no","maybe"])
    plt.ylabel("Count")
    plt.title("Wins / Losses by Gold Label (CoT vs Direct)")
    # annotate tops
    tops = win_counts.values + loss_counts.values
    for i, (w, l, t) in enumerate(zip(win_counts.values, loss_counts.values, tops)):
        plt.text(i, t + max(tops)*0.02 if max(tops)>0 else 0.5, f"{int(t)}", ha="center", va="bottom", fontsize=9)
        if w>0:
            plt.text(i, w/2, f"{int(w)}", ha="center", va="center", color="white", fontsize=8)
        if l>0:
            plt.text(i, w + l/2, f"{int(l)}", ha="center", va="center", color="white", fontsize=8)
    plt.ylim(0, max(tops)*1.18 if len(tops) and max(tops)>0 else 1)
    plt.legend()
    _savefig(os.path.join(plots_dir, "wins_losses_by_gold.png"))

    print("[done] wrote plots to", plots_dir)

if __name__ == "__main__":
    main()
