import itertools

import numpy as np
import pytest

from capstone_experimentation import (Instance, action_visited_masks, backward, compute_index,
                         evaluate, expect, initial_value,
                         make_greedy_rule, make_index_greedyrepair_rule,
                         make_index_optassign_rule, solve_lp_bound)

# A small, fast instance reused by several tests. Deliberately on 'ring' (degree 3)
# rather than 'ring2', because ring2 on N=5 is the COMPLETE graph and so exercises
# none of the movement-constraint logic.
BASE = dict(N=5, K=2, L=4, seed=3, graph="ring2", risk_spread=1.0,
            esc_lo=0.35, esc_hi=0.95)


@pytest.fixture(scope="module")
def inst():
    return Instance(T=12, **BASE)


@pytest.fixture(scope="module")
def masks(inst):
    return action_visited_masks(inst)


# ======================================================================
def test_kernels_are_stochastic(inst):
    """Every row of every transition matrix must sum to 1.

    Trivial, but it catches typos in the dynamics definition -- e.g. forgetting the
    absorbing top level, which would silently leak probability mass.
    """
    for i in range(inst.N):
        assert np.allclose(inst.Tun[i].sum(axis=1), 1.0), \
            f"unvisited kernel for zone {i} is not row-stochastic"
    assert np.allclose(inst.Tvis.sum(axis=1), 1.0)


# ======================================================================
def test_expectation_matches_brute_force():
    """THE critical test: the fast expectation equals the slow one.

    `expect` applies a Kronecker product of per-zone kernels as a sequence of
    mode-wise tensor contractions -- about 170x faster than summing over successors
    at N=6, and the cleverest code in the repo. Here it is checked against a
    literal enumeration of every successor configuration, with every probability
    multiplied out by hand, on an instance small enough to do that exhaustively.

    Checked for all 2^N possible visited-set patterns, since the kernel used per
    zone depends on whether that zone was visited.
    """
    inst = Instance(N=4, K=1, L=3, T=4, seed=11, graph="ring",
                    esc_lo=0.4, esc_hi=0.9)
    rng = np.random.default_rng(11)
    # An arbitrary value function. Using random values (rather than something
    # structured) means an axis-permutation bug cannot hide behind symmetry.
    V = rng.normal(size=inst.cfg_shape)

    for visited in itertools.product([False, True], repeat=inst.N):
        vm = np.array(visited)
        fast = expect(V, inst, vm)

        slow = np.zeros(inst.cfg_shape)
        for k in itertools.product(range(inst.L), repeat=inst.N):
            acc = 0.0
            for kp in itertools.product(range(inst.L), repeat=inst.N):
                pr = 1.0
                for i in range(inst.N):
                    Ti = inst.Tvis if vm[i] else inst.Tun[i]
                    pr *= Ti[k[i], kp[i]]
                    if pr == 0.0:
                        break          # early exit: most successors are impossible
                if pr:
                    acc += pr * V[kp]
            slow[k] = acc

        assert np.allclose(fast, slow, atol=1e-10), \
            f"contraction != brute force for visited={visited}"


# ======================================================================
def test_greedy_repair_matches_reference():
    """The vectorised greedy-repair rule equals a plain per-configuration loop.

    The vectorised implementation decides for all L^N configurations at once, which
    is necessary for speed and notoriously error-prone. This test compares it
    against an obvious, slow, readable reference.

    THIS TEST FOUND A REAL BUG: the original used np.putmask, which indexes by flat
    position rather than by count of selected entries, silently writing wrong values
    and producing gaps above 1000% in some regimes while looking exactly correct in
    others.
    """
    inst = Instance(N=5, K=2, L=3, T=6, seed=5, graph="ring",
                    esc_lo=0.3, esc_hi=0.9, beta=0.7)
    lam = compute_index(inst, np.full(inst.N, inst.K / inst.N))
    rule = make_index_greedyrepair_rule(inst, lam)

    def ref(t, vc, k):
        """Reference: sequential greedy claim, one configuration at a time."""
        acts = inst.actions[inst.vidx[vc]]
        ben = {}
        for v in range(inst.K):
            for z in range(inst.N):
                if inst.adj[vc[v], z]:
                    ben[(v, z)] = (lam[z, k[z], t]
                                   - inst.beta * inst.dist[vc[v], z])
        used_v, used_z, tgt = set(), set(), {}
        while len(used_v) < inst.K:
            cand = [(b, vz) for vz, b in ben.items()
                    if vz[0] not in used_v and vz[1] not in used_z]
            if not cand:
                break
            # Tie-break by lowest vehicle then lowest zone, matching argmax order.
            b, (v, z) = max(cand, key=lambda x: (x[0], -x[1][0], -x[1][1]))
            tgt[v] = z
            used_v.add(v)
            used_z.add(z)
        if len(tgt) < inst.K:
            return acts.index(vc)          # fall back to "everybody stays"
        cand_t = tuple(tgt[v] for v in range(inst.K))
        return acts.index(cand_t) if cand_t in acts else acts.index(vc)

    t = 2
    mismatch = checked = 0
    for vc_i, vc in enumerate(inst.vcfgs):
        got = rule(t, vc_i, len(inst.actions[vc_i]))
        for k in itertools.product(range(inst.L), repeat=inst.N):
            checked += 1
            if int(got[k]) != ref(t, vc, np.array(k)):
                mismatch += 1
    assert mismatch == 0, f"{mismatch}/{checked} mismatches vs reference"


def test_optimal_assignment_beats_repair():
    """Optimal assignment can never be worse than greedy repair.

    Both use the same scores; one searches all legal joint moves, the other grabs
    greedily. A violation would mean the "optimal" search is not actually optimal.
    """
    inst = Instance(N=5, K=2, L=3, T=6, seed=5, graph="ring",
                    esc_lo=0.3, esc_hi=0.9, beta=0.7)
    lam = compute_index(inst, np.full(inst.N, inst.K / inst.N))
    m = action_visited_masks(inst)
    v_rep = initial_value(
        backward(inst, m, rule=make_index_greedyrepair_rule(inst, lam)), inst)
    v_opt = initial_value(
        backward(inst, m, rule=make_index_optassign_rule(inst, lam)), inst)
    assert v_opt <= v_rep + 1e-7


# ======================================================================
def test_index_properties(inst):
    """The index must be non-negative, zero when clean, and monotone in escalation.

    All three follow from its definition as "future harm avoided by resetting now":
    resetting an already-clean zone gains nothing, and a worse zone cannot be less
    valuable to reset.
    """
    lam = compute_index(inst, np.full(inst.N, inst.K / inst.N))
    assert np.all(lam >= -1e-9), "index went negative"
    assert np.allclose(lam[:, 0, :], 0.0), "index at level 0 should be exactly 0"
    assert np.all(np.diff(lam, axis=1) >= -1e-9), \
        "index is not monotone in escalation level"


# ======================================================================
def _monte_carlo(inst, rule, n_roll=20000, seed=7):
    """Independent Monte-Carlo estimate of a rule's expected total cost.

    Shares almost no code with the exact evaluator, which is the point: agreement
    between the two is strong evidence both are right.
    """
    rng = np.random.default_rng(seed)
    cache = {}
    tot = 0.0
    for _ in range(n_roll):
        k = np.zeros(inst.N, dtype=int)
        vc = tuple(range(inst.K))
        c = 0.0
        for t in range(inst.T):
            c += float(inst.r @ inst.level[k])
            vc_i = inst.vidx[vc]
            key = (t, vc_i)
            if key not in cache:
                cache[key] = rule(t, vc_i, len(inst.actions[vc_i]))
            tgt = inst.actions[vc_i][int(cache[key][tuple(k)])]
            nk = k.copy()
            u = rng.random(inst.N)
            for i in range(inst.N):
                if i in tgt:
                    nk[i] = 0
                else:
                    nk[i] = int(np.searchsorted(np.cumsum(inst.Tun[i, k[i]]), u[i]))
            k, vc = nk, tgt
        tot += c
    return tot / n_roll


def test_exact_evaluation_matches_monte_carlo(inst, masks):
    """Exact policy evaluation agrees with simulation to within sampling error.

    Tolerance is 1.5% at 20k rollouts, which is loose enough not to be flaky and
    tight enough to catch a systematic error in either path.
    """
    rule = make_greedy_rule(inst)
    exact = initial_value(backward(inst, masks, rule=rule), inst)
    mc = _monte_carlo(inst, rule)
    assert abs(exact - mc) / exact < 0.015, f"exact={exact:.4f} mc={mc:.4f}"


# ======================================================================
def test_optimum_is_a_lower_bound(inst):
    """Nothing can beat the exact optimum, and greedy must beat random.

    The standing invariant. It ran hundreds of times during the study and is the
    cheapest possible guard against a broken solver.
    """
    res, meta = evaluate(inst)
    opt = res["OPT"]
    for k, v in res.items():
        assert v >= opt - 1e-7, f"{k}={v} beat the optimum {opt}"
    assert res["RANDOM"] >= res["GREEDY"] - 1e-9
    # The greedy-repair fallback ("everybody stays" when the greedy grab fails)
    # should never actually trigger.
    assert meta["repair_fallback_rate"] == 0.0


def test_cost_non_decreasing_in_horizon():
    """A longer horizon cannot cost less, since all costs are non-negative.

    Catches off-by-one errors in the backward recursion.
    """
    prev = -1.0
    for T in (5, 10, 15):
        i = Instance(T=T, **BASE)
        v = initial_value(backward(i, action_visited_masks(i), rule=None), i)
        assert v >= prev - 1e-9
        prev = v


# ======================================================================
def test_lp_bound_is_below_optimum():
    """The LP relaxation must never exceed the exact optimum.

    A relaxation permits things the real problem forbids, so its optimum is at or
    below the true one. If this fails, a constraint is wrong or too tight.
    """
    for (N, K, L, T, g) in [(4, 2, 3, 8, "ring"), (5, 2, 3, 10, "ring2"),
                            (6, 3, 3, 8, "ring2")]:
        i = Instance(N=N, K=K, L=L, T=T, seed=0, graph=g, risk_spread=0.0,
                     esc_lo=0.3, esc_hi=0.95, cost_pow=2.0)
        lb = solve_lp_bound(i)
        opt = initial_value(backward(i, action_visited_masks(i), rule=None), i)
        assert lb <= opt + 1e-6, f"LP {lb} exceeded optimum {opt} at N={N}"


def test_lp_bound_per_zone_is_scale_invariant():
    """Under proportional scaling the LP bound PER ZONE must be constant.

    This is a construction validator, not a result. It is what proved the scaling
    setup was correct after a first attempt where a clustered initial fleet made the
    far side of a large ring unreachable, so per-zone cost grew with N and the whole
    asymptotic trend was an artefact.

    Uses the uniform initial placement; with a clustered start this test fails,
    which is exactly the point.
    """
    from capstone_experimentation.scaling import BigInstance

    per_zone = []
    for h in (2, 3, 4):
        inst = BigInstance(h, L=3, T=8, base_seed=0)
        ip = np.zeros(inst.N)
        ip[inst.init_positions()] = 1.0
        per_zone.append(solve_lp_bound(inst, init_pos=ip) / inst.N)
    assert np.allclose(per_zone, per_zone[0], rtol=1e-6), \
        f"per-zone bound drifted across scales: {per_zone}"


# ======================================================================
def test_staleness_optimum_bounds_policies():
    """In the multi-epoch-travel model too, nothing beats the exact optimum."""
    from capstone_experimentation import staleness as st

    inst = st.StaleInstance(N=5, K=2, L=3, T=6, seed=0)
    opt = st.initial_value(st.backward(inst), inst)
    lam = compute_index(inst, np.full(inst.N, inst.K / inst.N))
    for mode, beta in [("greedy", 0.0), ("now", 1.0), ("arrival", 1.0)]:
        v = st.initial_value(
            st.backward(inst, st.make_rule(inst, lam, mode, beta)), inst)
        assert v >= opt - 1e-7, f"{mode} beat the optimum"
