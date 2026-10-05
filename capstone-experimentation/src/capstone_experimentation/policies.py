"""
Decision rules (policies) and the full comparison ladder.

A "rule" here is callable: rule(t, vc_i, n_actions) -> int aray of shape
cfg_shape, giving the chosen action index for EVERY zone configuration at once.
That vectrorised signature is what lets `model.backward` evaluate a policy exactly
rather than by simulation: there is no sampling anywhere in the avaluation path.

THE LADDER
----------
    OPT                             exact optimum (no rule; backward minimises)
    RANDOM                          uniform over legal joint moves -- the floor
    GREEDY                          myopic: highest current risk-weighted damage
    IDX-naive/OPT-A                 classical index + optimal assignment
    IDX-naive/GREEDY                classical index + greedy sequential repair
    IDX-match/OPT-A                 matching-aware index + optimal assignment
    IDX-match/GREEDY                matching-aware index + greedy repair

That's a 2x2 factorial (index type x assignment method) plus three references,
which is what lets each factor's contribution be isolated.
"""

import numpy as np

from .model import (action_visited_masks, backward, compute_index, initial_value, zone_score_tensors)

# Sentinel for an illegal pairing. Large enoguh to lose every comparison,
# small enough to stay well within the float range so arithmetic on it is safe.
BIG = 1e12

# MYOPIC BASELINE
def make_greedy_rule(inst):
    """
    Send vehicles to the reachable zones with the most risk-weighted damage now.

    Score of zone z is r_z * w(k_z): purely the cost being incured this instant
    with no lookahead.
    """
    # Precompute each zone's myopic score as a full-configuration tensor, once.
    scores = []
    for i in range(inst.N):
        shape = [1] * inst.N
        shape[i] = inst.L
        scores.append(np.broadcast_to(
            (inst.r[i] * inst.level).reshape(shape), inst.cfg_shape))
        
    def rule(t, vc_i, n_acts):
        # For each legal joint move, total score = sum of all visited zones'
        # scores. Stacking gives shape (n_actions,) + cfg_shape, so a single
        # argmax over axis 0 picks the best actions for every configuration.
        tot = np.stack([sum(scores[z] for z in tgt) for tgt in inst.actions[vc_i]])
        return np.argmax(tot, axis = 0).astype(np.intp)
    return rule

# INDEX + OPTIMAL ASSINGMENT
def make_index_optassign_rule(inst, lam):
    """
    Index scoring with optimal vehicle->zone assignment.

    Having a score per zone gives no information about which vehicle goes where.
    Vehicles have different reachable sets and cannot collide. This assignment
    problems can normally be solved using the Hungarian algorithm.

    See Kuhn, H.W. (1955), The Hungarian method for the assignment problem†. 
        Naval Research Logistics, 2: 83-97. https://doi.org/10.1002/nav.3800020109

    Here we get the identical answer by exhaustive argmax over ther enumerated legal
    joint moves. That set is the set of legal assignments (distinct targets, adjacency
    respected), so its maximum is the optimal assignment. Unlike calling the Hungarian
    per configuration, it stays fully vectorized across all L^N configurations.

    Benefit of a move = sum over vehicles of (index of target - beta * hops).
    beta = 0 ignores travel entirely; beta > 0 charges for it.
    """
    cache = {}      # score tensors depend only on t, build each t once.

    def rule(t, vc_i, n_acts):
        if t not in cache:
            cache[t] = zone_score_tensors(inst, lam, t)
        S = cache[t]
        vc = inst.vcfgs[vc_i]
        tot = []
        for tgt in inst.actions[vc_i]:
            # Value gained: the indices of the zones this action covers
            v = sum(S[z] for z in tgt)
            # Price paid: travel summed over vehicles -> a scalar for this action.
            # (it depends on positions, not on zone levels), so it subtracts
            # uniformly across configurations.
            pen = inst.beta * sum(inst.dist[vc[m], tgt[m]] for m in range(inst.K))
            tot.append(v - pen)
        return np.argmax(np.stack(tot), axis = 0).astype(np.intp)
    return rule

# INDEX + GREEDY SEQUENTIAL PAIR
def make_index_greedyrepair_rule(inst, lam, stats = None):
    """
    Index scoring with a GREEDY sequential assignment instead of the optimum.

    Repeatedly take the single best remaining (vehicle, reachable zone) pair, mark
    both as used, repeat K times. Suboptimal by construction. An early greedy grab
    can block a better overall pairing and is structurally similar to the repair
    step used in the reference MAB-ML implementation. It isolates how much the
    assignment method alone is worth.

    Everything is vectorised across all L^N zone configurations at once, because
    doing it one configuration at a time is far too slow. Vectorised combinatorics
    is error-prone. Hence, this function is cross-checked against a plain per-
    configuration reference loop.
    """
    cache = {}
    N, K = inst.N, inst.K

    def rule(t, vc_i, n_acts):
        if t not in cache:
            cache[t] = zone_score_tensors(inst, lam, t)
        S = cache[t]
        vc = inst.vcfgs[vc_i]
        acts = inst.actions[vc_i]

        # B[v, z, <configuration axes>] = benefit of sending vehicle v to zone z
        # Unreachable pairs are -BIG so they can never win an argmax.
        B = np.full((K, N) + inst.cfg_shape, -BIG)
        for v in range(K):
            for z in range(N):
                if inst.adj[vc[v], z]:
                    B[v, z] = S[z] - inst.beta * inst.dist[vc[v], z]
                    
        # Availability bookkeeping, per configuration: which vehicles still need
        # a target, and which zones are still free.
        avail_v = np.ones((K, ) + inst.cfg_shape, dtype = bool)
        avail_z = np.ones((N, ) + inst.cfg_shape, dtype = bool)

        # target[v] = chosen zone for vehicle v; -1 means "not yet assigned"
        target = np.full((K, ) + inst.cfg_shape, -1, dtype = np.intp)

        for _ in range(K):      # K greedy picks
            # Mask out the pairs whose vehicles or zone is already taken
            M = np.where(avail_v[:, None] & avail_z[None, :], B, -BIG)

            # Flatten the (vehicle, zone) pair axes into one so a single argmax
            # finds the globally best remaining pair per configuration
            flat = M.reshape((K * N, ) + inst.cfg_shape)
            best = np.argmax(flat, axis=0)

            # Decode the flat index back into (vehicle, zone).
            bv, bz = best // N, best % N

            # If even the best pair is -BIG, no legal pair remains for this
            # configuration; `ok` marks where the pick is genuine.
            ok = np.take_along_axis(flat, best[None, ...], axis = 0)[0] > -BIG / 2

            # Build a boolean selector: True at (v, config) where v == bv and the
            # pick was genuine. `ar`is arange(K) shaped to broadcast against the
            # configuration axes.
            ar = np.arange(K).reshape((K, ) + (1, ) * inst.N)
            sel_v = (ar == bv[None, ...]) & ok[None, ...]

            # BUG NOTE: this was originally np.putmask(target, sel_v, np.broadcast_to(bz, target.shape)[sel_v])
            # np.putmask assigns target.flat[n] = values[n], indexing by FLAT POSITION in the target rather than
            # by count of selected entries. If fed a compressed array of the right length, it silently writes the
            # wrong values to the right place, producing gaps above 1000% in some regimes and in exact agreement
            # with the optimum in some others.
            target = np.where(sel_v, np.broadcast_to(bz, target.shape), target)

            # Retrieve the chosen vehicle and zone
            avail_v &= ~sel_v
            arz = np.arange(N).reshape((N, ) + (1, ) * inst.N)
            avail_z &= ~((arz == bz[None, ...]) & ok[None, ...])
        
        # Convert the per-vehicle targets into an index into `acts`. Encode this target tuple in base N, then
        # look it up.
        code = np.zeros(inst.cfg_shape, dtype = np.int64)
        valid = np.ones(inst.cfg_shape, dtype = bool)
        for v in range(K):
            valid &= target[v] >= 0                 # every vehicle got a target
            code += np.maximum(target[v], 0) * (N ** v)
        for v in range(K):                          # and the targets are distinct
            for u in range(v + 1, K):
                valid &= target[v] != target[u]

        # Lookup table from encoded target tuple -> action index; -1 for tuples
        # that are not legal actions
        lut = np.full(N ** K, -1, dtype = np.intp)
        for a_i, tgt in enumerate(acts):
            lut[sum(tgt[m] * (N ** m) for m in range(K))] = a_i
        out = lut[code]

        # Wherever greedy produces an incomplete or illegal assignment, fall back to
        # no movement, which is always legal. This is implemented so that the fallback
        # rate doesn't go hidden.
        bad = (~valid) | (out < 0)
        if stats is not None:
            stats["fallback"] += int(bad.sum())
            stats["total"] += int(bad.size)
        out = np.where(bad, acts.index(vc), out)
        return out
    return rule

# THE MATCHING AWARE INDEX - REALIZED SERVICE RATES BY FIXED POINT
def realised_service_rates(inst, masks, rule, n_roll = 4000, seed = 1):
    """
    Estimate how often each zone is actually visited under a given policy.

    Used only to set the reference parameter q_i for the matching-aware index.
    q_i is a policy hyperparamter, not a reported result. The final evaluation
    of the resulting policy is still exact, so sampling noise cannot flatter
    or penalise any policy's reported cost.
    """
    rng = np.random.default_rng(seed)
    visits = np.zeros(inst.N)
    cache = {}          # rule outputs are deterministic per (t, vc_i); to be reused

    for _ in range(n_roll):
        k = np.zeros(inst.N, dtype = int)           # all zones start quiescent
        vc = tuple(range(inst.K))                   # canonical start positions
        for t in range(inst.T):
            vc_i = inst.vidx[vc]
            key = (t, vc_i)
            if key not in cache:
                cache[key] = rule(t, vc_i, len(inst.actions[vc_i]))

            # Index the vectorised decision array at this configuration
            a_i = int(cache[key][tuple(k)])
            tgt = inst.actions[vc_i][a_i]
            visits[list(tgt)] += 1

            # Advance the zones. Sampling from the row of the transition kernel
            # via inverse-CDF: searchsorted on the cumulative row.
            nk = k.copy()
            u = rng.random(inst.N)
            for i in range(inst.N):
                if i in tgt:
                    nk[i] = 0           # visited -> reset
                else:
                    nk[i] = int(np.searchsorted(np.cumsum(inst.Tun[i, k[i]]), u[i]))
            k, vc = nk, tgt
    
    # Visits per zone per step = the empiirical service rate.
    return visits / (n_roll * inst.T)

def matched_index(inst, masks, n_iter = 6, damp = 0.5, verbose = False):
    """
    Iterate index <-> realised service rate to a fixed point.

    The proposed contribution under test: instead of assuming every zone gets
    q = K/N attention, give each zone the rate it actually recieves, and recompute
    the index accordingly. Repeat until the assumption and outcome agree.

    The update is damped (damp < 1) because the raw map can oscillate.

    For the record, this makes the performance WORSE, monotonically and on every
    seed tested. Iterating harder makes it worse still. See experiment 03.
    """
    q = np.full(inst.N, inst.K / inst.N)        # start from the classical assumption
    lam = compute_index(inst, q)
    for it in range(n_iter):
        rule = make_index_optassign_rule(inst, lam)
        q_new = realised_service_rates(inst, masks, rule, n_roll = 1500, seed = 100 + it)

        # Clip away from 0 and 1: q = 0 or q = 1 makes the single zone recursion degenerate
        # and the index uninformative.
        q_new = np.clip(q_new, 1e-3, 0.999)
        shift = np.abs(q_new - q).max()
        q = (1 - damp) * q + damp * q_new
        lam = compute_index(inst, q)
        if verbose:
            print(f"Fixed point iteration {it}: max|dq| = {shift:.4f}")
        if shift < 5e-3:        # converged
            break
    return lam, q

# THE FULL LADDER
ORDER = ["OPT", "RANDOM", "GREEDY", "IDX-naive/GREEDY", "IDX-naive/OPT-A",
         "IDX-match/GREEDY", "IDX-match/OPT-A"]

def full_ladder(inst, verbose = False):
    """
    Evaluate every policy on one instance and return {name: expected cost}

    Every entry is compared by exact backward induction, so the numbers carry no
    sampling error. The ending assertion is a standing invariant. Nothing beats
    the optimum.
    """
    masks = action_visited_masks(inst)
    r = {}

    r["OPT"] = initial_value(backward(inst, masks, rule = None), inst)
    r["RANDOM"] = initial_value(backward(inst, masks, randomize = True), inst)
    r["GREEDY"] = initial_value(backward(inst, masks, rule = make_greedy_rule(inst)), inst)

    # Classical relaxation: uniform share budget
    lam_n = compute_index(inst, np.full(inst.N, inst.K / inst.N))
    r["IDX-naive/OPT-A"] = initial_value(backward(inst, masks, rule = make_index_optassign_rule(inst, lam_n)), inst)
    r["IDX-naive/GREEDY"] = initial_value(backward(inst, masks, rule = make_index_greedyrepair_rule(inst, lam_n)), inst)

    # Matching aware: realized service rates.
    lam_n, q_m = matched_index(inst, masks, verbose = verbose)
    r["IDX-match/OPT-A"] = initial_value(backward(inst, masks, rule = make_index_optassign_rule(inst, lam_n)), inst)
    r["IDX-match/GREEDY"] = initial_value(backward(inst, masks, rule = make_index_greedyrepair_rule(inst, lam_n)), inst)

    for k, v in r.items():
        assert v >= r["OPT"] - 1e-7, f"{k} beat the exact optimum."
    return r, q_m

def gap(res, key):
    """
    Percentage above the exact optimum.
    """
    return 100.0 * (res[key] - res["OPT"]) / res["OPT"]

def gaps(res):
    """
    Percentage above the exact optimum, for every policy at once.
    """
    opt = res["OPT"]
    return {k: 100.0 * (v - opt) / opt for k, v in res.items()}

def evaluate(inst, verbose = False):
    """
    Wrapper returning (costs, metadata) including the repair fallback rate, which
    should be zero.
    """
    masks = action_visited_masks(inst)
    res = {}

    res["OPT"] = initial_value(backward(inst, masks, rule = None), inst)
    res["RANDOM"] = initial_value(backward(inst, masks, randomize = True), inst)
    res["GREEDY"] = initial_value(backward(inst, masks, rule = make_greedy_rule(inst)), inst)

    lam_naive = compute_index(inst, np.full(inst.N, inst.K / inst.N))
    res["IDX-naive/OPT-A"] = initial_value(backward(inst, masks, rule = make_index_optassign_rule(inst, lam_naive)), inst)
    st = {"fallback": 0, "total": 0}
    res["IDX-naive/GREEDY"] = initial_value(backward(inst, masks, rule = make_index_greedyrepair_rule(inst, lam_naive, stats = st)), inst)

    lam_m, q_m = matched_index(inst, masks, verbose = verbose)
    res["IDX-match/OPT-A"] = initial_value(backward(inst, masks, rule = make_index_optassign_rule(inst, lam_m)), inst)
    res["IDX-match/GREEDY"] = initial_value(backward(inst, masks, rule = make_index_greedyrepair_rule(inst, lam_m)), inst)

    meta = {
        "q_naive": np.full(inst.N, inst.K / inst.N),
        "q_matched": q_m,
        "repair_fallback_rate": st["fallback"] / max(st["total"], 1)
    }

    return res, meta
    
