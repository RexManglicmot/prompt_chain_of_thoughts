import pandas as pd
from datasets import load_dataset
from pathlib import Path
from app.config import cfg 

# Load cfg
cfg

def main(n: int = 100, n_decoys: int = 10):
    # load pubmed_qa (labeled split)
    ds = load_dataset("pubmed_qa", "pqa_labeled")["train"]
    df = ds.to_pandas()[["pubid", "question", "context", "final_decision"]]

    # sample n rows
    df = df.sample(n=n, random_state=42).reset_index(drop=True)

    # Added 9/18
    def flatten_context(x):
        if isinstance(x, dict) and "contexts" in x:
            return " ".join(x["contexts"])
        if isinstance(x, list):
            return " ".join(x)
        return str(x)
    # End

    # build normal rows
    out = pd.DataFrame({
        "id": range(1, n + 1),
        "question": df["question"],
        "context_id": "PubMedQA-" + df["pubid"].astype(str),
        # "context": df["context"].apply(lambda c: " ".join(c) if isinstance(c, list) else str(c)),
        "context": df["context"].apply(flatten_context),
        "gold_answer": df["final_decision"].str.lower(),
        "has_answer_in_context": 1
    })

    # add decoys
    decoys = []
    for i in range(n_decoys):
        q = f"What is the recommended treatment for condition X{i+1}?"
        ctx = f"This is a random guideline-like passage about nutrition and exercise, not related to condition X{i+1}."
        decoys.append({
            "id": n + i + 1,
            "question": q,
            "context_id": f"Decoy-{i+1}",
            "context": ctx,
            "gold_answer": "not in context",
            "has_answer_in_context": 0
        })
    out = pd.concat([out, pd.DataFrame(decoys)], ignore_index=True)

    # resolve save path from config
    out_path = Path(cfg["paths"].get("data_csv", "data/clinical_qa.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f" Saved {len(out)} examples (including {n_decoys} decoys) to {out_path}")

if __name__ == "__main__":
    main(n=100, n_decoys=10)
