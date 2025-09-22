import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Path to config.yaml (root of project)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

class DotDict(dict):
    """
    dict → object with attribute access
    In other words, dot-style access (cfg.project.input_dir).
    """
    def __getattr__(self, k):
        # Called only if normal attribute lookup fails (i.e., not a real attr like .items)
        v = self.get(k)                         # Look up the key; returns None if the key doesn't exist
        if isinstance(v, dict):
            # Wrap nested dicts so chaining works: cfg.project.input_dir
            return DotDict(v)
        # Could be any type (or None if missing key)
        return v
        
    # Redirect attribute assignment/deletion to item assignment/deletion:
    __setattr__ = dict.__setitem__              # cfg.foo = 1    -> cfg["foo"] = 1
    __delattr__ = dict.__delitem__              # del cfg.foo    -> del cfg["foo"]

def load_config(path: Path = CONFIG_PATH):
    """Load YAML config into a dictionary and normalize paths."""
    
    # Open and read the YAML config and save as f
    with open(path, "r") as f:
        # Specifically, tt reads a YAML document and returns only basic Python types 
        # (dict, list, str, int, float, bool, None, etc.) using the SafeLoader.
        # yaml.safe_load returns whatever the document represents, for example:
            # Top-level mapping → dict
            # Top-level sequence → list
            # Top-level scalar → str/int/float/bool/None
        # In this case, this project, where the top level is a mapping, it returns a dict
        cfg = yaml.safe_load(f)

    # Root folder to resolve any relative paths from (the config file's directory)
    # path.parent means “the folder that contains the file at path.”
    root = path.parent
    
    # normalize "project" paths for input and output directories
    if "project" in cfg:
        # Convert the YAML's project.input_dir and .output_dir to an absolute paths and write it back:
        # uses `root` (the folder containing config.yaml) as the base for relative paths
        if "input_dir" in cfg["project"]:
            cfg["project"]["input_dir"] = str((root / cfg["project"]["input_dir"]).resolve())
        if "output_dir" in cfg["project"]:
            cfg["project"]["output_dir"] = str((root / cfg["project"]["output_dir"]).resolve())

    # Normalize "paths" section, normalize each entry to an absolute path
    if "paths" in cfg:
        # Iterate over key (k) and path string (v) in the "paths" mapping
        for k, v in cfg["paths"].items():
            # Join the config folder (root) with the path value (v) and resolve:
            #  - If v is relative (e.g., "outputs/plots"), it becomes rooted at `root`
            #  - If v is already absolute (e.g., "/mnt/data"), `root / v` yields v unchanged
            #  - .resolve() makes it absolute, collapses "..", and follows symlinks
            #  - str(...) stores a plain string back into the config
            cfg["paths"][k] = str((root / v).resolve())
    
    # DotDict only for ergonomics: it lets you write cfg.project.input_dir instead of cfg["project"]["input_dir"].
        # Reasons:
        # Cleaner access. dot notation is shorter and reads nicer.
        # Recursive access: nested dicts also become dot-accessible.
    return DotDict(cfg)

# Global config object
cfg = load_config()


# Hugging Face API key 
HF_TOKEN = os.getenv("HF_TOKEN")

# Debug mode / smoke test
if __name__ == "__main__":
    print("Config loaded successfully")
    print("Project:", cfg.project.name)
    print("Input dir:", cfg.project.input_dir)
    print("Output dir:", cfg.project.output_dir)
    print("Primary model:", cfg.models.primary)
    print("Device:", cfg.models.device)
    print("HF_TOKEN found:", HF_TOKEN is not None)
