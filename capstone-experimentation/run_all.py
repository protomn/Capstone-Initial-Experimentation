"""
Run the whole study end to end.

    uv run python run_all.py --quick     ~2 minutes, reduced settings
    uv run python run_all.py             full reproduction, ~30-45 minutes

Order matters and is not arbitrary:
  tests first    -- if the solver is wrong, nothing after it means anything
  01 before 02   -- 01 finds the regime in which 02's comparison is meaningful
  05 before 06   -- 05 establishes the bound is valid and how loose, which is what
                    makes 06's numbers interpretable
  08 last        -- the figure reads the JSON the others write
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"

# (label, argv). None means "run pytest instead of a script".
STAGES = [
    ("correctness suite (pytest)", None),
    ("01 regime search", "exp1_regime_search.py"),
    ("02 policy ladder", "exp2_policy_ladder.py"),
    ("03 diagnostics", "exp3_diagnostics.py"),
    ("04 exact scaling", "exp4_exact_scaling.py"),
    ("05 LP validation", "exp5_lp_validation.py"),
    ("06 asymptotic", "exp6_asymptotic.py"),
    ("07 staleness", "exp7_staleness.py"),
    ("08 figure", "exp8_figure.py"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="reduced settings; ~2 minutes instead of ~40")
    ap.add_argument("--skip-tests", action="store_true",
                    help="skip the correctness suite (not recommended)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only stages whose label contains one of these "
                         "strings, e.g. --only 06 07")
    args = ap.parse_args()

    t0 = time.time()
    failures = []

    for label, script in STAGES:
        if args.only and not any(tok in label for tok in args.only):
            continue
        if script is None:
            if args.skip_tests:
                continue
            cmd = [sys.executable, "-m", "pytest"]
            cwd = ROOT
        else:
            cmd = [sys.executable, script]
            if args.quick:
                cmd.append("--quick")
            # Run from experiments/ so the scripts' `from _common import ...`
            # resolves without any sys.path manipulation.
            cwd = EXP

        print("\n" + "=" * 78)
        print(f"### {label}")
        print("=" * 78, flush=True)

        rc = subprocess.call(cmd, cwd=cwd)
        if rc != 0:
            failures.append(label)
            print(f"\n!!! {label} exited with code {rc}", file=sys.stderr)
            # Correctness failing invalidates everything downstream, so stop.
            if script is None:
                print("Correctness suite failed -- stopping. Fix this first.",
                      file=sys.stderr)
                break

    dt = time.time() - t0
    print("\n" + "=" * 78)
    if failures:
        print(f"FINISHED WITH FAILURES ({dt:.0f}s): {', '.join(failures)}")
        sys.exit(1)
    print(f"All stages completed in {dt:.0f}s.")
    print("Results in results/  (JSON per experiment, plus spike_findings.png)")
    if args.quick:
        print("\nNOTE: --quick uses fewer seeds and smaller instances. Conclusions")
        print("hold, but the numbers will not match FINDINGS.md exactly.")


if __name__ == "__main__":
    main()
