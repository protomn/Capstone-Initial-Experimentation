"""
Shared script for all experimental scripts

--quick shrinks the seed count and problem sizes so the suite can finish
in a short period of time. Full mode reproduces complete results and can
take a long period of time.
"""

import argparse
import json
import pathlib
import sys
import time

# all results are saved at the repo root, along with experiments/ and src/
RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

def parse_args(description: str | None):
    """
    Standard argument set for every experiment.
    """

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--quick", action="store_true",
                    help="fewer seeds and smaller instances; same qualitative conclusions, numbers will differ.")
    ap.add_argument("--seeds", type=int, default=None,
                    help="override number of random seeds per experiment.")
    
    return ap.parse_args()

def n_seeds(args, full: int = 6, quick: int = 2):
    """
    Number of seeds to be run. 
    Explicit override wins, then --quick, then default.
    """

    if args.seeds is not None:
        return args.seeds
    return quick if args.quick else full

def save(name: str, payload):
    """
    Write a results dict to results/<name>.json

    Every result is written as JSON with default as str.
    numpy arrays and scalars degrade gracefully rather than raising.
    Results are additive, no overwriting.
    """
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[saved] {path.relative_to(RESULTS.parent)}")

class Timer:
    """
    Context manager that prints elapsed time.
    """

    def __init__(self, label):
        self.label = label
    
    def __enter__(self):
        self.t0 = time.time()
        print(f"-----{self.label}-----", flush=True)
        return self
    
    def __exit__(self, *exc):
        print(f"---{self.label}: {time.time() - self.t0:.1f}s", flush=True)
        return False
    
def banner(text: str):
    """
    Section header, flushed immediately
    """
    print(f"\n{text}", flush=True)

def rule_line(width: int = 78):
    """
    Horizontal separator for tables.
    """
    print("-" * width, flush=True)

def require(cond, msg):
    """
    Assertion that exits with a clear message, no traceback.
    """

    if not cond:
        print(f"INVARIANT VIOLATED: {msg}", file=sys.stderr)
        sys.exit(1)