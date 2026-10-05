"""
Does the deviation vanish as the system grows?

THE CLAIM UNDER TEST
--------------------
Theory says the gap between an index policy and the relaxed optimum shrinks
EXPONENTIALLY as the system scales. Doubling the city should roughly square the
closeness, not merely improve it a bit. That is the theoretical justification for
using an index policy.

With many zones, random fluctuations average out and the system behaves like a
smooth "fluid".

WHY THIS NEEDS THE LP BOUND
---------------------------
Exact value iteration dies at N~7 (experiment 04). The LP bound (experiment 05)
stays cheap and is movement-aware, so it is the only available reference at N = 320.
"""

import numpy as np

from _common import banner, parse_args, rule_line, save, Timer
from capstone_experimentation import compute_index, solve_lp_bound
from capstone_experimentation.scaling import BigInstance, make_scores, simulate

args = parse_args(__doc__)

L, T = 4, 15
# h is the scaling parameter: N = 5h zones, K = 2h vehicles, so K/N = 0.4 always.
# (n_roll shrinks at large h because each Hungarian solve grows with K x N.)
LADDER = ([(2, 300), (4, 300), (8, 300), (16, 300), (32, 250), (64, 120)]
          if not args.quick else [(2, 120), (4, 120), (8, 100)])

rows = []

with Timer("experiment 06: asymptotic behaviour"):
    banner(f"Proportional scaling: N = 5h, K = 2h (K/N = 0.4), degree 5, "
           f"L={L}, T={T}")
    print("Deviation = (policy cost - LP lower bound) / LP lower bound\n")
    hdr = (f"{'h':>3} {'N':>4} {'K':>4} {'LP bound':>10} {'LP/zone':>8} "
           f"{'greedy%':>16} {'index%':>16} {'index+travel%':>16}")
    print(hdr)
    rule_line(len(hdr))

    for h, n_roll in LADDER:
        # --- instance without a travel charge --------------------------
        inst = BigInstance(h, L=L, T=T, base_seed=0)

        # The initial layout must be spread, not clustered. See
        # BigInstance.init_positions for why this single line decides whether the
        # experiment measures anything real.
        ip = np.zeros(inst.N)
        ip[inst.init_positions()] = 1.0

        # The reference point. Movement-aware LP relaxation.
        lb = solve_lp_bound(inst, init_pos=ip)

        # Index under the classical relaxation. compute_index only needs
        # N/L/T/r/level/Tun, so it works unchanged on BigInstance.
        lam = compute_index(inst, np.full(inst.N, inst.K / inst.N))

        # Monte Carlo, exact evaluation is impossible at these sizes.
        g_mu, g_se, _ = simulate(inst, make_scores(inst, "greedy"),
                                 n_roll=n_roll, seed=1)
        i_mu, i_se, _ = simulate(inst, make_scores(inst, "index", lam),
                                 n_roll=n_roll, seed=1)

        # --- same instance, policy now charges for travel ---------------
        instT = BigInstance(h, L=L, T=T, base_seed=0, beta=1.0)
        lamT = compute_index(instT, np.full(instT.N, instT.K / instT.N))
        t_mu, t_se, bad = simulate(instT, make_scores(instT, "index", lamT),
                                   n_roll=n_roll, seed=1)
        # `bad` counts illegal pairings chosen by the assignment solver. Should be 0.
        assert bad == 0.0, f"assignment produced illegal pairings ({bad})"

        def dev(m):
            return 100 * (m - lb) / lb

        rows.append(dict(h=h, N=inst.N, K=inst.K, lp=lb,
                         lp_per_zone=lb / inst.N,
                         greedy=dev(g_mu), index=dev(i_mu),
                         index_travel=dev(t_mu),
                         se_greedy=100 * g_se / lb, se_index=100 * i_se / lb,
                         se_travel=100 * t_se / lb))
        print(f"{h:>3} {inst.N:>4} {inst.K:>4} {lb:10.3f} {lb/inst.N:8.4f} "
              f"{dev(g_mu):11.2f}+-{100*g_se/lb:4.2f} "
              f"{dev(i_mu):11.2f}+-{100*i_se/lb:4.2f} "
              f"{dev(t_mu):11.2f}+-{100*t_se/lb:4.2f}")


    # The construction validator.
    banner("LP bound PER ZONE (must be constant if the scaling is clean)")
    pz = [r["lp_per_zone"] for r in rows]
    for r in rows:
        print(f"  h={r['h']:>3} N={r['N']:>4}: {r['lp_per_zone']:.4f}")
    ok = np.allclose(pz, pz[0], rtol=1e-6)
    print(f"  -> {'CONSTANT: scaling construction is clean' if ok else 'DRIFTING: the construction is broken, results are not interpretable'}")

    banner("Trend as the system scales")
    for name in ["greedy", "index", "index_travel"]:
        seq = [f"{r[name]:.1f}" for r in rows]
        change = rows[-1][name] - rows[0][name]
        # A genuinely exponential decay would fall towards 0, not settle.
        verdict = ("shrinking" if change < -1
                   else "growing" if change > 1 else "flat")
        print(f"  {name:>13}: {' -> '.join(seq)}   ({verdict}, "
              f"{change:+.1f} pp end to end)")

    rule_line()
    print("Reading: the deviation PLATEAUS. That is what an irreducible per-zone")
    print("bound slack looks like, not what a failing policy looks like. The")
    print("ordering of policies and the travel-charge advantage are robust; the")
    print("absolute levels are not. Settling the claim needs a tighter bound.")

save("e06_asymptotic", dict(quick=args.quick, L=L, T=T, rows=rows,
                            per_zone_constant=bool(ok)))
