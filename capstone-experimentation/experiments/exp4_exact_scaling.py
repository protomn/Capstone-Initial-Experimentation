"""
Scaling, using the exact optimum (and hitting its ceiling).

The joint state space is L^N * N!/(N-K)!. Watch the `states` column: 972 at N=4,
K=2, then 87,480 at N=6, K=3. A 90x explosion for two extra zones. That is the
curse of dimensionality, and it is why exact methods stop at N~7 -- which is why
experiment 06 has to switch to an LP bound.

Two data points cannot establish a trend.
"""

import numpy as np

from _common import banner, n_seeds, parse_args, rule_line, save, Timer
from capstone_experimentation import (Instance, action_visited_masks, backward, compute_index,
                         initial_value, make_greedy_rule,
                         make_index_optassign_rule)

args = parse_args(__doc__)
SEEDS = n_seeds(args, full=3, quick=1)

# L=3 and T=10 rather than the L=4, T=12 used elsewhere: the state space grows like
# L^N, so trimming L is what makes N=6, K=3 reachable at all.
BASE = dict(L=3, T=10, graph="ring2", risk_spread=0.0,
            esc_lo=0.3, esc_hi=0.95, cost_pow=2.0)

OUT = {}


def run(N, K):
    """Exact optimum plus three policies for one (N, K), averaged over seeds."""
    g = {"greedy": [], "index": [], "index_travel": []}
    opts, n_states = [], None
    for s in range(SEEDS):
        # beta=0 version: no travel charge.
        inst = Instance(N=N, K=K, seed=s, beta=0.0, **BASE)
        # beta=1 version: identical environment, policy charges for travel.
        instT = Instance(N=N, K=K, seed=s, beta=1.0, **BASE)
        n_states = inst.n_cfg * inst.n_v

        masks = action_visited_masks(inst)
        mT = action_visited_masks(instT)

        opt = initial_value(backward(inst, masks, rule=None), inst)
        opts.append(opt)

        gr = initial_value(backward(inst, masks, rule=make_greedy_rule(inst)),
                           inst)
        lam = compute_index(inst, np.full(N, K / N))
        ix = initial_value(
            backward(inst, masks, rule=make_index_optassign_rule(inst, lam)), inst)
        lamT = compute_index(instT, np.full(N, K / N))
        ixt = initial_value(
            backward(instT, mT, rule=make_index_optassign_rule(instT, lamT)),
            instT)

        # Note ixt is compared to the SAME opt: beta changes only the policy.
        for nm, v in [("greedy", gr), ("index", ix), ("index_travel", ixt)]:
            assert v >= opt - 1e-7, nm
            g[nm].append(100 * (v - opt) / opt)

    return (n_states, float(np.mean(opts)),
            {k: float(np.mean(v)) for k, v in g.items()})


HDR = (f"{'N':>3} {'K':>3} {'states':>10} {'OPT':>8} {'greedy%':>9} "
       f"{'index%':>8} {'index+travel%':>14}")

with Timer("experiment 04: exact scaling"):

    banner("Proportional scaling, K/N = 0.5 -- the asymptotic regime")
    print(HDR)
    rows = []
    # (6,3) is the largest exactly solvable point at these settings: 3^6 * 120
    # vehicle configurations = 87,480 states, each with ~60 legal joint actions.
    for N, K in ([(4, 2), (6, 3)] if not args.quick else [(4, 2)]):
        ns, opt, g = run(N, K)
        rows.append(dict(N=N, K=K, states=ns, opt=opt, **g))
        print(f"{N:>3} {K:>3} {ns:>10} {opt:8.3f} {g['greedy']:9.2f} "
              f"{g['index']:8.2f} {g['index_travel']:14.2f}")
    OUT["proportional"] = rows

    banner("Fixed fleet K=2, growing city -- the fleet becomes scarcer")
    print(HDR)
    rows2 = []
    for N in ([4, 5, 6] if not args.quick else [4, 5]):
        ns, opt, g = run(N, 2)
        rows2.append(dict(N=N, K=2, states=ns, opt=opt, **g))
        print(f"{N:>3} {2:>3} {ns:>10} {opt:8.3f} {g['greedy']:9.2f} "
              f"{g['index']:8.2f} {g['index_travel']:14.2f}")
    OUT["fixed_fleet"] = rows2

    rule_line()
    print("Reading: the state count explodes ~90x for two extra zones, so exact")
    print("methods cannot reach the scales an asymptotic claim concerns. Two")
    print("points are not a trend. Experiment 06 switches to an LP lower bound.")

save("e04_exact_scaling", dict(seeds=SEEDS, quick=args.quick, results=OUT))
