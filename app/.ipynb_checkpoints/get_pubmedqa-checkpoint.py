import pandas as pd
from datasets import load_dataset
from pathlib import Path
from app.config import cfg 

# Load cfg
cfg

def main(n: int = 1000):
    # load pubmed_qa (labeled split)
    # Use load_dataset from the dataset module to obtain labeled train datasets
    ds = load_dataset("pubmed_qa", "pqa_labeled")["train"]
    # turn df into pandas DataFrame and then column-selects
    df = ds.to_pandas()[["pubid", "question", "context", "final_decision"]]

 
    # Normalizes a “context” field into one flat string
    # Impt and useful for combining strings
    def flatten_context(x):
        # Check if x is a dict and there exists a "contexts" key
        if isinstance(x, dict) and "contexts" in x:
            # It takes the list (or other iterable) stored at x["contexts"] 
            # and concatenates all its elements into one string, 
            # inserting a single space between each item.
            # Example: x["contexts"] = ["sent A", "sent B"] → returns "sent A sent B".
            """
            Order of evaluation:
                spaces = " "
                join_fn = spaces.join          # step 1: get the callable
                seq = x["contexts"]            # step 2: evaluate the argument
                result = join_fn(seq)          # step 3: call it
            """
            return " ".join(x["contexts"])
        if isinstance(x, list):
            return " ".join(x)
        return str(x)

    # Build normal rows
    out = pd.DataFrame({
        "id": range(1, n + 1),
        "question": df["question"],
        "context_id": "PubMedQA-" + df["pubid"].astype(str),
        "context": df["context"].apply(flatten_context),            # call function created earlier
        "gold_answer": df["final_decision"].str.lower(),
    })

    # Build the output file path
        # Order of evaluation....Path, cfg.paths.data_csv
        # So, it is calling it to retrieve the value, 
        # then passing that value into Path(...)
        # then return the value for "data_csv"
    out_path = Path(cfg.paths.data_csv)
    
    # Ensure the destination folder exists:
    # .parent is the directory portion of the file path
    # parents=True creates any missing intermediate directories
    # exist_ok=True prevents errors if the directory already exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the DataFrame `out` to CSV at the resolved path with no pandas index column
    out.to_csv(out_path, index=False)

    # print output in the terminal
    print(f" Saved {len(out)} examples to {out_path}")

if __name__ == "__main__":
    main()
