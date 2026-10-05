"""
Multi-epoch travel

THE PROBLEM
-----------
Everywhere else a vehicle arrives within one step. Reality: you dispatch across
town, it takes eight minutes, and during that time the vehicle covers NOTHING --
not the zone it left, not the zone it is heading to. The score that justified the
dispatch may be out of date on arrival.

TWO CANDIDATE FIXES
    charge for travel   subtract beta * travel time. Treats delay as a COST.
    project to arrival  score the zone at its EXPECTED state when you get there.
                        Treats delay as an ESTIMATION problem.

THE METHODOLOGICAL POINT OF THIS EXPERIMENT
-------------------------------------------
Part A runs both at a SHARED beta, which is the obvious thing to do and gives the
WRONG answer: projection looks actively harmful. Part B sweeps beta separately for
each variant and reverses the conclusion.

Why: forward projection inflates the score of distant zones, because a zone
escalates while you travel toward it, so the benefit of resetting it on arrival
looks larger. Projection therefore REWARDS LONG TRIPS unless the travel charge is
raised to compensate. The stale-aware variant needs roughly a third more beta.

Both parts are kept because the contrast is the lesson: a shared hyperparameter can
invert a comparison, and this nearly went into the writeup with the wrong sign.
"""

import numpy as np

from _common import banner, n_seeds, parse_args, rule_line, save, Timer
from capstone_experimentation import compute_index
from capstone_experimentation import staleness as st

args = parse_args(__doc__)
SEEDS = n_seeds(args, full=6, quick=2)

OUT = {}

with Timer("experiment 07: staleness"):

    # ==================================================================
    # PART A -- both variants at a shared beta. The naive comparison.
    # ==================================================================
    banner("A. Shared beta=1 for both variants (the naive, misleading comparison)")
    # N=5 gives max trip 2 epochs and 43,740 states; N=6 gives max trip 3 and
    # 349,920 states. Both exactly solvable, the second slowly.
    CASES = ([(5, 2, 3, 10), (6, 2, 3, 9)] if not args.quick
             else [(5, 2, 3, 8)])
    print(f"{'N':>3} {'K':>3} {'maxtrip':>8} {'states':>9} {'OPT':>8} "
          f"{'GREEDY%':>9} {'now%':>8} {'now+chg%':>10} {'arr%':>8} "
          f"{'arr+chg%':>10}")
    agg = {k: [] for k in ["GREEDY", "now", "now+chg", "arr", "arr+chg"]}
    a_rows = []
    for (N, K, L, T) in CASES:
        for s in range(min(SEEDS, 3)):
            inst = st.StaleInstance(N=N, K=K, L=L, T=T, seed=s)
            # Ground truth for the multi-epoch model.
            opt = st.initial_value(st.backward(inst), inst)
            lam = compute_index(inst, np.full(N, K / N))

            vals = {}
            for nm, mode, b in [("GREEDY", "greedy", 0.0),
                                ("now", "now", 0.0),
                                ("now+chg", "now", 1.0),
                                ("arr", "arrival", 0.0),
                                ("arr+chg", "arrival", 1.0)]:
                v = st.initial_value(
                    st.backward(inst, st.make_rule(inst, lam, mode, b)), inst)
                assert v >= opt - 1e-7, f"{nm} beat the optimum"
                vals[nm] = 100 * (v - opt) / opt
                agg[nm].append(vals[nm])

            n_states = L ** N * inst.n_v
            a_rows.append(dict(N=N, K=K, seed=s, opt=opt, **vals))
            print(f"{N:>3} {K:>3} {inst.D:>8} {n_states:>9} {opt:8.3f} "
                  f"{vals['GREEDY']:9.2f} {vals['now']:8.2f} "
                  f"{vals['now+chg']:10.2f} {vals['arr']:8.2f} "
                  f"{vals['arr+chg']:10.2f}")

    print("\n  means:")
    for k, v in agg.items():
        print(f"    {k:>9}: {np.mean(v):7.2f}%")
    print(f"\n  travel charge alone         : "
          f"{np.mean(agg['now']) - np.mean(agg['now+chg']):+.2f} pp")
    print(f"  projection alone            : "
          f"{np.mean(agg['now']) - np.mean(agg['arr']):+.2f} pp   "
          f"<-- looks HARMFUL at shared beta")
    OUT["A_shared_beta"] = dict(rows=a_rows,
                                means={k: float(np.mean(v))
                                       for k, v in agg.items()})

    # ==================================================================
    # PART B -- sweep beta separately for each variant.
    # ==================================================================
    banner("B. beta swept separately per variant (the fair comparison)")
    BETAS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0] if not args.quick else [0.0, 2.0, 4.0]
    tab = {m: {b: [] for b in BETAS} for m in ("now", "arrival")}

    for s in range(SEEDS):
        inst = st.StaleInstance(N=5, K=2, L=3, T=10 if not args.quick else 8,
                                seed=s)
        opt = st.initial_value(st.backward(inst), inst)
        lam = compute_index(inst, np.full(5, 0.4))
        for m in ("now", "arrival"):
            for b in BETAS:
                v = st.initial_value(
                    st.backward(inst, st.make_rule(inst, lam, m, b)), inst)
                assert v >= opt - 1e-7
                tab[m][b].append(100 * (v - opt) / opt)

    print(f"{'beta':>6} {'stale-blind':>20} {'stale-aware':>20}")
    b_rows = []
    for b in BETAS:
        a = np.array(tab["now"][b])
        c = np.array(tab["arrival"][b])
        b_rows.append(dict(beta=b, blind=float(a.mean()), aware=float(c.mean()),
                           blind_se=float(a.std(ddof=1) / np.sqrt(len(a)))
                           if len(a) > 1 else 0.0,
                           aware_se=float(c.std(ddof=1) / np.sqrt(len(c)))
                           if len(c) > 1 else 0.0))
        se_a = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        se_c = c.std(ddof=1) / np.sqrt(len(c)) if len(c) > 1 else 0.0
        print(f"{b:6.1f} {a.mean():12.2f}+-{se_a:4.2f}     "
              f"{c.mean():12.2f}+-{se_c:4.2f}")

    # Each variant's own best beta.
    bn = min(BETAS, key=lambda b: np.mean(tab["now"][b]))
    ba = min(BETAS, key=lambda b: np.mean(tab["arrival"][b]))
    # Paired difference at each variant's optimum: the fair measure of what
    # projection is worth.
    d = np.array(tab["now"][bn]) - np.array(tab["arrival"][ba])
    print(f"\n  best stale-blind: beta={bn} -> {np.mean(tab['now'][bn]):.2f}%")
    print(f"  best stale-aware: beta={ba} -> {np.mean(tab['arrival'][ba]):.2f}%")
    print(f"  paired difference: {d.mean():+.2f} pp  "
          f"per-seed {np.round(d, 2).tolist()}")
    print(f"  seeds favouring stale-aware: {(d > 0).sum()}/{len(d)}")
    print(f"\n  travel charge alone (beta 0 -> best): "
          f"{np.mean(tab['now'][0.0]) - np.mean(tab['now'][bn]):+.2f} pp")
    OUT["B_tuned_beta"] = dict(rows=b_rows, best_blind_beta=bn,
                               best_aware_beta=ba,
                               paired_diff=d.tolist())

    rule_line()
    print("Reading: charging for travel is the dominant fix by an order of")
    print("magnitude. Projecting to arrival helps too, but ONLY with its own")
    print("larger beta -- at a shared beta it looks harmful. The two corrections")
    print("must be tuned jointly, and beta has a cliff past its optimum.")
    print("\nCAVEAT: beta here is tuned on the same instances it is scored on.")
    print("A proper study needs held-out instances.")

save("e07_staleness", dict(seeds=SEEDS, quick=args.quick, results=OUT))
