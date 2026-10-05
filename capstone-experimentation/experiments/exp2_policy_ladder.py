"""
The main policy ladder across four regimes.

WHAT THIS MEASURES
------------------
Every policy against the exact optimum, in four deliberately chosen regimes, so
each design factor can be isolated:

    A  index's best case      homogeneous risk, heterogeneous escalation,
                              quadratic cost, ring2, no travel charge
    B  heterogeneous risk     A but with risk weights spanning 7x
    C  tight movement         A but on 'ring' (degree 3)
    D  travel charged         A but with beta > 0

The ladder is a 2x2 factorial (index type x assignment method) plus three
reference points, which lets us read off:

    greedy   -> index          is a lookahead score worth anything?
    repair   -> optimal assign does the assignment method matter?
    naive    -> matching-aware is the proposed contribution any good?

HEADLINE RESULTS (full mode)
    A: RANDOM 442%, GREEDY 12.30%, IDX-naive 10.39%, IDX-match 11.29%
       index worth +1.91 pp; matching-awareness worth -0.90 pp
    B: index worth +0.02 pp (risk weights swamp it)
    C: gaps ~100%; assignment method worth +5.64 pp -- more than the index
    D: best heuristic anywhere, 7.75%

    Matching-awareness was negative on 22 of 22 seed-level comparisons.
"""

import numpy as np

from _common import banner, n_seeds, parse_args, rule_line, save, Timer
from capstone_experimentation import ORDER, Instance, full_ladder, gap

args = parse_args(__doc__)
SEEDS = n_seeds(args, full=6, quick=2)

# Regime A. Chosen by experiment 01 as the setting where the index has the best
# chance of differing from greedy at all.
BASE = dict(N=5, K=2, L=4, T=12, graph="ring2", risk_spread=0.0,
            esc_lo=0.3, esc_hi=0.95, cost_pow=2.0)

RESULTS = {}


def report(title, insts):
    """Run the full ladder on a list of instances and print mean gaps plus the
    three isolated contrasts."""
    banner(title)
    acc = {k: [] for k in ORDER}
    for inst in insts:
        res, _q = full_ladder(inst)     # asserts internally that OPT bounds all
        for k in ORDER:
            acc[k].append(gap(res, k))

    print(f"{'policy':>20} {'mean gap %':>11} {'sd':>7} {'min':>7} {'max':>7}")
    out = {}
    for k in ORDER:
        a = np.array(acc[k])
        out[k] = dict(mean=float(a.mean()), sd=float(a.std()),
                      lo=float(a.min()), hi=float(a.max()))
        print(f"{k:>20} {a.mean():11.2f} {a.std():7.2f} "
              f"{a.min():7.2f} {a.max():7.2f}")

    # The three contrasts. Sign convention: POSITIVE means the second thing named
    # is better, i.e. the change helped.
    d_idx = np.array(acc["IDX-naive/OPT-A"]) - np.array(acc["IDX-match/OPT-A"])
    d_asg = np.array(acc["IDX-naive/GREEDY"]) - np.array(acc["IDX-naive/OPT-A"])
    d_gre = np.array(acc["GREEDY"]) - np.array(acc["IDX-naive/OPT-A"])
    print(f"  index naive->matched  : {d_idx.mean():+6.2f} pp  "
          f"per-seed {np.round(d_idx, 2).tolist()}")
    print(f"  repair->opt assignment: {d_asg.mean():+6.2f} pp  "
          f"per-seed {np.round(d_asg, 2).tolist()}")
    print(f"  greedy->index         : {d_gre.mean():+6.2f} pp  "
          f"per-seed {np.round(d_gre, 2).tolist()}")

    return dict(policies=out,
                d_index_naive_to_matched=d_idx.tolist(),
                d_repair_to_optassign=d_asg.tolist(),
                d_greedy_to_index=d_gre.tolist())


with Timer("experiment 02: policy ladder"):

    # --- A: the index's most favourable regime --------------------------
    RESULTS["A_separating"] = report(
        "A. Index's best case (homogeneous risk, heterogeneous escalation, "
        "quadratic cost)",
        [Instance(seed=s, **BASE) for s in range(SEEDS)])

    # --- B: heterogeneous risk weights ----------------------------------
    # Everything else identical, so the only change is r_i spanning 0.25-1.75.
    b = dict(BASE, risk_spread=1.0)
    RESULTS["B_hetero_risk"] = report(
        "B. Heterogeneous risk weights (otherwise identical to A)",
        [Instance(seed=s, **b) for s in range(SEEDS)])

    # --- C: tighter movement --------------------------------------------
    # 'ring' has degree 3 vs ring2's degree 5. NOTE: on N=5, ring2 is actually the
    # COMPLETE graph, so this contrast is really "constrained vs unconstrained".
    c = dict(BASE, graph="ring")
    RESULTS["C_sparse_graph"] = report(
        "C. Sparser graph, degree 3 -- movement genuinely constrains",
        [Instance(seed=s, **c) for s in range(SEEDS)])

    # --- D: charging for travel -----------------------------------------
    # beta enters only the POLICY's scoring rule, never the environment, so the
    # optimum is unchanged and the comparison is fair.
    for beta in ([0.5, 1.5] if not args.quick else [0.5]):
        d = dict(BASE, beta=beta)
        RESULTS[f"D_beta{beta}"] = report(
            f"D. Travel charged inside the index score, beta={beta}",
            [Instance(seed=s, **d) for s in range(max(SEEDS - 2, 2))])

    rule_line()
    print("Reading: the index is worth ~2 pp at best; the assignment method is")
    print("worth ~5.6 pp when movement is tight; charging for travel is worth")
    print("more than either; and making the index matching-aware makes it WORSE.")

save("e02_policy_ladder", dict(seeds=SEEDS, quick=args.quick, base=BASE,
                               results=RESULTS))
