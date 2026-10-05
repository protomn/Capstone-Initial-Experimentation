"""
Validate the LP relaxation against the exact optimum.

Experiment 06 measures deviation from the LP bound at scales where the exact
optimum is unavailable. That measurement is only interpretable if we know (a) the
bound is genuinely a lower bound, and (b) roughly how loose it is.

EXPECTED RESULT (full mode)
    0 violations; looseness from 0.62% to 38.57%. Tightest when the fleet is
    generous relative to the number of zones (N=6, K=3 -> 0.80%); loosest when the
    fleet is scarce and the graph is sparse (N=6, K=2, ring -> 38.57%).
"""

import numpy as np

from _common import banner, n_seeds, parse_args, require, rule_line, save, Timer
from capstone_experimentation import (Instance, action_visited_masks, backward, initial_value,
                         solve_lp_bound)

args = parse_args(__doc__)
SEEDS = n_seeds(args, full=2, quick=1)

# Spread deliberately over fleet ratios and graph densities, because that is what
# looseness turns out to depend on.
CASES = [(4, 2, 3, 8, "ring"), (4, 2, 3, 8, "ring2"),
         (5, 2, 3, 10, "ring"), (5, 2, 3, 10, "ring2"),
         (5, 2, 4, 10, "ring2"), (6, 2, 3, 8, "ring"),
         (6, 3, 3, 8, "ring2"), (6, 2, 4, 10, "ring2")]
if args.quick:
    CASES = [(4, 2, 3, 8, "ring"), (5, 2, 3, 10, "ring2"), (6, 3, 3, 8, "ring2")]

rows, violations = [], 0

with Timer("experiment 05: LP validation"):
    banner("LP relaxation vs exact optimum (LP must be <= OPT)")
    print(f"{'N':>3} {'K':>3} {'L':>3} {'T':>3} {'graph':>6} "
          f"{'LP':>9} {'OPT':>9} {'looseness%':>11} {'K/N':>5}")
    rule_line(78)

    for (N, K, L, T, g) in CASES:
        for s in range(SEEDS):
            inst = Instance(N=N, K=K, L=L, T=T, seed=s, graph=g,
                            risk_spread=0.0, esc_lo=0.3, esc_hi=0.95,
                            cost_pow=2.0)
            # The bound. Default init_pos (vehicles on zones 0..K-1) is fine here
            # because N is small enough that nothing is unreachable.
            lp = solve_lp_bound(inst)
            # Ground truth.
            opt = initial_value(
                backward(inst, action_visited_masks(inst), rule=None), inst)

            loose = 100 * (opt - lp) / opt
            bad = lp > opt + 1e-6
            violations += int(bad)
            rows.append(dict(N=N, K=K, L=L, T=T, graph=g, seed=s,
                             lp=lp, opt=opt, looseness_pct=float(loose)))
            print(f"{N:>3} {K:>3} {L:>3} {T:>3} {g:>6} {lp:9.4f} {opt:9.4f} "
                  f"{loose:11.2f} {K/N:5.2f}"
                  + ("   <-- VIOLATION" if bad else ""))

    rule_line(78)
    require(violations == 0, f"{violations} instances had LP > OPT")
    lo = min(r["looseness_pct"] for r in rows)
    hi = max(r["looseness_pct"] for r in rows)
    print(f"violations: 0    looseness range: {lo:.2f}% to {hi:.2f}%")
    print()
    print("Reading: the bound is valid, but its slack depends strongly on the")
    print("fleet ratio and graph density. Carry this forward -- it is why the")
    print("plateau in experiment 06 is evidence about the BOUND, not the policy.")

save("e05_lp_validation", dict(seeds=SEEDS, quick=args.quick,
                               violations=violations, rows=rows,
                               looseness_min=lo, looseness_max=hi))
