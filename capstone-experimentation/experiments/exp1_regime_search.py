"""
Where does an index policy differ from greedy?

WHAT IT SWEEPS
    risk_spread  0.0 / 0.4 / 1.0        how much zones differ in importance
    escalation   deterministic / 0.3-0.95  how much zones differ in dynamics
    cost shape   linear / quadratic     how fast harm accumulates
    graph        ring (deg 3) / ring2 (deg 5)

EXPECTED RESULT (full mode)
    best separation ~1.86 pp at risk_spread=0, heterogeneous escalation,
    quadratic cost, ring2 -- and near zero, sometimes negative, elsewhere.
"""

import itertools
import numpy as np

from _common import banner, n_seeds, parse_args, rule_line, save, Timer
from capstone_experimentation import (Instance, action_visited_masks, backward, compute_index,
                         initial_value, make_greedy_rule,
                         make_index_optassign_rule)

args = parse_args(__doc__)
SEEDS = n_seeds(args, full=4, quick=2)

# In quick mode drop the middle risk level and the linear cost, keeping the four
# corners that carry the conclusion.

RISK = [0.0, 0.4, 1.0] if not args.quick else [0.0, 1.0]
ESC = [(1.0, 1.0), (0.3, 0.95)]
POW = [1.0, 2.0] if not args.quick else [2.0]
GRAPH = ["ring", "ring2"]

rows = []

with Timer("experiment 01: regime search"):
    banner("Does the index differ from greedy? Sweep of instance regimes")
    print(f"{'risk_sp':>7} {'esc':>12} {'pow':>4} {'graph':>6} "
          f"{'OPT':>8} {'GREEDY%':>8} {'IDX%':>8} {'sep(pp)':>8}")
    rule_line()

    for risk_spread, esc, cost_pow, graph in itertools.product(
            RISK, ESC, POW, GRAPH):

        gaps_g, gaps_i, opts = [], [], []
        for seed in range(SEEDS):
            # One instance. N=5, K=2 keeps the joint state space at
            # 4^5 * 20 = 20,480 states, so the exact optimum is cheap.
            inst = Instance(N=5, K=2, L=4, T=12, seed=seed, graph=graph,
                            risk_spread=risk_spread,
                            esc_lo=esc[0], esc_hi=esc[1], cost_pow=cost_pow)
            masks = action_visited_masks(inst)

            # Ground truth: exact optimum by backward induction.
            opt = initial_value(backward(inst, masks, rule=None), inst)

            # Myopic baseline.
            g = initial_value(
                backward(inst, masks, rule=make_greedy_rule(inst)), inst)

            # Index policy under the classical relaxation (uniform budget share
            # q = K/N), with optimal assignment so the comparison isolates the
            # SCORING rule rather than the assignment method.
            lam = compute_index(inst, np.full(inst.N, inst.K / inst.N))
            i_ = initial_value(
                backward(inst, masks, rule=make_index_optassign_rule(inst, lam)),
                inst)

            # Standing invariant: neither policy may beat the optimum.
            assert g >= opt - 1e-7 and i_ >= opt - 1e-7

            opts.append(opt)
            gaps_g.append(100 * (g - opt) / opt)
            gaps_i.append(100 * (i_ - opt) / opt)

        gg, gi = np.mean(gaps_g), np.mean(gaps_i)
        sep = gg - gi          # positive means the index helped
        rows.append(dict(risk_spread=risk_spread, esc=list(esc),
                         cost_pow=cost_pow, graph=graph,
                         opt=float(np.mean(opts)), greedy_gap=float(gg),
                         index_gap=float(gi), separation_pp=float(sep)))
        print(f"{risk_spread:7.1f} {str(esc):>12} {cost_pow:4.0f} {graph:>6} "
              f"{np.mean(opts):8.3f} {gg:8.2f} {gi:8.2f} {sep:8.2f}")

    rule_line()
    best = max(rows, key=lambda r: r["separation_pp"])
    print(f"\nLargest separation: risk_spread={best['risk_spread']}, "
          f"esc={best['esc']}, cost_pow={best['cost_pow']}, "
          f"graph={best['graph']}  ->  {best['separation_pp']:.2f} pp")

    # The three readings that matter, printed so a fresh reader gets them without
    # having to interpret the table.
    print("\nReadings:")
    print("  1. The index's best-case advantage is small (~2 pp), and it is")
    print("     negative in some regimes.")
    print("  2. Separation appears only when risk weights are UNIFORM and the")
    print("     dynamics are not. Heterogeneous risk swamps the distinction.")
    print("  3. The gaps themselves are dominated by the GRAPH, not the policy:")
    print("     sparse graphs give ~100% gaps, denser ones ~10%.")

save("e01_regime_search", dict(seeds=SEEDS, quick=args.quick, rows=rows,
                               best=best))
