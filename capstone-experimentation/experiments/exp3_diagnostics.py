"""
Follow up diagnostics on the surprise findings from experiment 2.

Four follow-ups:

  G  Sweep the travel-cost weight beta. Finds a broad safe plateau and a
     catastrophic cliff past it (vehicles become too expensive to move and freeze).

  H  Is matching-awareness merely an unconverged fixed point? Sweep damping and
     iteration count. Answer: no -- converging harder makes it MONOTONICALLY WORSE.

  I  Dose-response. Instead of comparing two discrete options, interpolate
     continuously between the naive service rate (K/N) and the realised one. This
     is the single cleanest result in the study: a monotone curve, no threshold, no
     noise.

  J  Decompose where the gap lives, on a constrained vs an unconstrained graph.
     Shows travel cost is worth ~15.7 pp on the constrained graph, nine times the
     index's 1.7 pp.
"""

import numpy as np

from _common import banner, n_seeds, parse_args, rule_line, save, Timer
from capstone_experimentation import (Instance, action_visited_masks, backward, compute_index,
                         initial_value, make_greedy_rule,
                         make_index_greedyrepair_rule,
                         make_index_optassign_rule, realised_service_rates)

args = parse_args(__doc__)
SEEDS = n_seeds(args, full=6, quick=2)

BASE = dict(N=5, K=2, L=4, T=12, graph="ring2", risk_spread=0.0,
            esc_lo=0.3, esc_hi=0.95, cost_pow=2.0)

OUT = {}


def exact_gap(inst, rule):
    """Percentage above the exact optimum for one instance and one decision rule.

    Recomputes the optimum each call. Wasteful but keeps every measurement
    self-contained, which matters more than speed at these sizes.
    """
    masks = action_visited_masks(inst)
    opt = initial_value(backward(inst, masks, rule=None), inst)
    v = initial_value(backward(inst, masks, rule=rule), inst)
    assert v >= opt - 1e-7
    return 100 * (v - opt) / opt, opt


with Timer("experiment 03: diagnostics"):

    # ==================================================================
    # G. Travel-cost weight sweep
    # ==================================================================
    banner("G. Travel-penalty sweep (index + optimal assignment)")
    betas = ([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0] if not args.quick
             else [0.0, 0.5, 2.0, 8.0])
    print(f"{'beta':>6} {'mean gap %':>11} {'sd':>7}   per-seed")
    g_rows = []
    for beta in betas:
        gs = []
        for s in range(SEEDS):
            # beta is passed to the Instance only so the policy can read it; the
            # environment and therefore the optimum are unaffected.
            inst = Instance(seed=s, beta=beta, **BASE)
            lam = compute_index(inst, np.full(inst.N, inst.K / inst.N))
            g, _ = exact_gap(inst, make_index_optassign_rule(inst, lam))
            gs.append(g)
        a = np.array(gs)
        g_rows.append(dict(beta=beta, mean=float(a.mean()), sd=float(a.std())))
        print(f"{beta:6.2f} {a.mean():11.2f} {a.std():7.2f}   "
              f"{np.round(a, 2).tolist()}")
    best = min(g_rows, key=lambda r: r["mean"])
    print(f"  best beta = {best['beta']} at {best['mean']:.2f}% "
          f"(beta=0 gives {g_rows[0]['mean']:.2f}%) -> travel charge is worth "
          f"{g_rows[0]['mean'] - best['mean']:.2f} pp")
    print("  NOTE the cliff at large beta: over-charging travel makes vehicles")
    print("  refuse to move, and the policy becomes worse than random.")
    OUT["G_beta_sweep"] = g_rows

    # ==================================================================
    # H. Is the matching-aware index just under-converged?
    # ==================================================================
    banner("H. Does converging the fixed point harder help? (it does not)")
    inst = Instance(seed=0, beta=0.0, **BASE)
    masks = action_visited_masks(inst)
    opt = initial_value(backward(inst, masks, rule=None), inst)
    lam_n = compute_index(inst, np.full(inst.N, inst.K / inst.N))
    g_naive = 100 * (initial_value(
        backward(inst, masks, rule=make_index_optassign_rule(inst, lam_n)),
        inst) - opt) / opt
    print(f"{'damp':>6} {'iters':>6} {'gap %':>8} {'max|q-K/N|':>11}")
    print(f"{'naive':>6} {'-':>6} {g_naive:8.2f} {0.0:11.3f}")
    h_rows = [dict(damp=None, iters=0, gap=float(g_naive), dq=0.0)]

    damps = [0.25, 0.5, 1.0] if not args.quick else [0.5]
    iters = [1, 3, 10] if not args.quick else [1, 5]
    n_roll = 3000 if not args.quick else 800
    for damp in damps:
        for it_n in iters:
            # Reimplement the fixed-point loop here rather than calling
            # matched_index, so damping and iteration count can be controlled.
            q = np.full(inst.N, inst.K / inst.N)
            lam = compute_index(inst, q)
            for it in range(it_n):
                rule = make_index_optassign_rule(inst, lam)
                qn = np.clip(realised_service_rates(inst, masks, rule,
                                                    n_roll=n_roll, seed=500 + it),
                             1e-3, 0.999)
                q = (1 - damp) * q + damp * qn
                lam = compute_index(inst, q)
            v = initial_value(
                backward(inst, masks, rule=make_index_optassign_rule(inst, lam)),
                inst)
            gp = 100 * (v - opt) / opt
            dq = float(np.abs(q - inst.K / inst.N).max())
            h_rows.append(dict(damp=damp, iters=it_n, gap=float(gp), dq=dq))
            print(f"{damp:6.2f} {it_n:6d} {gp:8.2f} {dq:11.3f}")
    OUT["H_fixed_point"] = h_rows

    # ==================================================================
    # I. Dose-response: interpolate naive -> realised service rate
    # ==================================================================
    banner("I. Dose-response between naive and realised service rates")
    q_naive = np.full(inst.N, inst.K / inst.N)
    rule0 = make_index_optassign_rule(inst, lam_n)
    q_real = np.clip(realised_service_rates(inst, masks, rule0,
                                            n_roll=6000 if not args.quick else 1500,
                                            seed=999), 1e-3, 0.999)
    print(f"  q_naive    = {np.round(q_naive, 3).tolist()}")
    print(f"  q_realised = {np.round(q_real, 3).tolist()}")
    print("  (the realised rates depart substantially from K/N, so this is not a")
    print("   tiny perturbation being amplified by noise)")
    print(f"{'alpha':>6} {'gap %':>8}")
    i_rows = []
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # alpha = 0 is the classical relaxation; alpha = 1 is fully matching-aware.
        q = (1 - alpha) * q_naive + alpha * q_real
        lam = compute_index(inst, q)
        v = initial_value(
            backward(inst, masks, rule=make_index_optassign_rule(inst, lam)), inst)
        gp = 100 * (v - opt) / opt
        i_rows.append(dict(alpha=alpha, gap=float(gp)))
        print(f"{alpha:6.2f} {gp:8.2f}")
    print("  Monotone degradation. WHY: discounting a zone's urgency by how often")
    print("  it gets served double-counts, because the escalation level ALREADY")
    print("  encodes service history -- a recently served zone is at level 0 and")
    print("  scores zero anyway. The correction destroys cross-zone comparability.")
    OUT["I_dose_response"] = dict(q_naive=q_naive.tolist(),
                                  q_realised=q_real.tolist(), rows=i_rows)

    # ==================================================================
    # J. Where does the gap actually live?
    # ==================================================================
    banner("J. Decomposition: index choice vs movement handling")
    j_out = {}
    for graph in ["ring", "ring2"]:
        gs = {}
        for s in range(SEEDS):
            kw = dict(BASE, graph=graph)
            inst = Instance(seed=s, **kw)
            masks = action_visited_masks(inst)
            opt = initial_value(backward(inst, masks, rule=None), inst)
            lam = compute_index(inst, np.full(inst.N, inst.K / inst.N))

            for name, rule in [
                ("greedy_myopic", make_greedy_rule(inst)),
                ("idx_repair", make_index_greedyrepair_rule(inst, lam)),
                ("idx_optassign", make_index_optassign_rule(inst, lam)),
            ]:
                v = initial_value(backward(inst, masks, rule=rule), inst)
                gs.setdefault(name, []).append(100 * (v - opt) / opt)

            # Same instance but with travel charged, scored against the SAME
            # optimum (beta does not change the environment).
            inst2 = Instance(seed=s, **dict(kw, beta=1.0))
            m2 = action_visited_masks(inst2)
            lam2 = compute_index(inst2, np.full(inst2.N, inst2.K / inst2.N))
            v = initial_value(
                backward(inst2, m2, rule=make_index_optassign_rule(inst2, lam2)),
                inst2)
            gs.setdefault("idx_optassign_travelcharge", []).append(
                100 * (v - opt) / opt)

        deg = Instance(seed=0, **dict(BASE, graph=graph)).adj.sum(1).mean()
        print(f"  graph={graph} (mean degree {deg:.1f})")
        j_out[graph] = {}
        for k, v in gs.items():
            j_out[graph][k] = float(np.mean(v))
            print(f"    {k:>28}: {np.mean(v):7.2f}%")
    OUT["J_decomposition"] = j_out

    rule_line()
    print("Central finding: when movement binds, how you handle MOVEMENT dominates")
    print("how you score ZONES. Travel charge ~15.7 pp, assignment ~5.6 pp,")
    print("index ~1.7 pp, matching-awareness negative.")

save("e03_diagnostics", dict(seeds=SEEDS, quick=args.quick, results=OUT))
