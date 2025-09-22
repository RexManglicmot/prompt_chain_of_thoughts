# app/stat_test.py
import pandas as pd
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar
from app.config import cfg

def main():
    # Load results
    results_csv = Path(cfg.paths.results_csv)
    df = pd.read_csv(results_csv)

    # Pivot to wide format: one row per id, cols = direct vs cot
    wide = df.pivot(index="id", columns="method", values="correct_em")

    # Count discordant pairs
    n10 = int(((wide["direct"] == 1) & (wide["cot"] == 0)).sum())  # Direct only correct
    n01 = int(((wide["direct"] == 0) & (wide["cot"] == 1)).sum())  # CoT only correct
    n00 = int(((wide["direct"] == 0) & (wide["cot"] == 0)).sum())
    n11 = int(((wide["direct"] == 1) & (wide["cot"] == 1)).sum())

    table = [[n00, n01],
             [n10, n11]]

    # Run McNemar’s test (exact if small sample, else chi2)
    result = mcnemar(table, exact=(n01 + n10 < 25), correction=True)

    print("=== McNemar’s Test ===")
    print(f"n01 (CoT only correct): {n01}")
    print(f"n10 (Direct only correct): {n10}")
    print(f"statistic = {result.statistic:.3f}, p-value = {result.pvalue:.4f}")
    if n01 > n10:
        print("Interpretation: CoT wins more often on discordant cases.")
    elif n10 > n01:
        print("Interpretation: Direct wins more often on discordant cases.")
    else:
        print("Interpretation: Tie on discordant cases.")

if __name__ == "__main__":
    main()
