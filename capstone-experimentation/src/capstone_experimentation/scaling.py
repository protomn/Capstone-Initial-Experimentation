"""
Large-N machinery: a lightweight instance and a Monte-Carlo evaluator.

WHY A SEPARATE INSTANCE CLASS
-----------------------------
`model.Instance` precompute `cost_cfg`, an array of shape (L, ) * N. At N = 20 that is
4^20 = 10^12 entries, which becomes difficult to allocate. `BigInstance` provides only
what the LP bound and the simulator need (N, K, L, T, r, p, e , adj, dist, Tun, level)
and skips every joint-state-space object.

THE SCALING CONSTRUCTION
------------------------
To test the asymptotic claim the system must be scaled the way the claim intends:
K/N fixed, local structure fixed, zone types replicated, no redrawing.

    N = n_types * h             zones
    K = k_per_type * h          vehicles (K/N is constant in h)
    graph: each zone adjacent to +/-1 and +/2 -> degree 5 regardless of N
    zone parameters (r, p, e) tile a fixed base set of n_types values.

This is how geography grows while local structure does not. The mean-field regime.

VALIDATION HOOK
---------------
Under this construction the LP bound PER ZONE must be constant in h. It is exactly
3.6267 for the default settings, from N = 10 to N = 320. See init_positions below.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment as lsa

# Cost used for illegal (unreachable) vehicle->zone pairings inside the assignment
# solver. Must be large enough to never be chosen, small enough to keep the LP
# solver's arithmetic well conditioned.
BIG = 1e9

class BigInstance:
    """
    Scalable instance: everything the LP and the simulator need, nothing that scales
    like L^N.
    """

    def __init__(self, h, L = 4, T = 15, base_seed = 0, cost_pow = 2.0, beta = 0.0,
                 risk_spread = 0.0, n_types = 5, k_per_type = 2):
        # The generator is seeded by base_seed only, not by h. The base zones
        # must be identical at every scale, so that growing the system replicates
        # types rather than resampling them.
        rng = np.random.default_rng(base_seed)

        self.N = n_types * h
        self.K = k_per_type * h
        self.L, self.T, self.beta = L, T, beta
        self.level = np.arange(L, dtype = float) ** cost_pow

        # Draw the base types once, then tile them h times around the ring.
        r0 = (1 - risk_spread) * np.ones(n_types) + risk_spread * rng.uniform(0.25, 1.75, n_types)
        p0 = rng.uniform(0.15, 0.55, n_types)
        e0 = rng.uniform(0.30, 0.95, n_types)
        self.r = np.tile(r0, h)
        self.p = np.tile(p0, h)
        self.e = np.tile(e0, h)

        # Ring plus chords at distance 2: degree 5 including the self-loop, at any N.
        # Local structure is therefore scale invariant, a requirement of the mean-field
        # argument
        N = self.N
        A = np.eye(N, dtype = bool)
        for i in range(N):
            for d in (1, 2):
                A[i, (i + d) % N] = A[(i + d) % N, i] = True
        self.adj = A

        # Same per-zone kernels as model.Instance; kept structurally identical so
        # that the two code paths describe the same dynamics.
        self.Tun = np.zeros((N, L, L))
        for i in range(N):
            self.Tun[i, 0, 0] = 1 - self.p[i]
            self.Tun[i, 0, 1] = self.p[i]
            for k in range(1, L - 1):
                self.Tun[i, k, k + 1] = self.e[i]
                self.Tun[i, k, k] = 1 - self.e[i]
            self.Tun[i, L - 1, L - 1] = 1.0

        # Floyd-Warshall shortest paths, for the travel-cost term in policies.
        D = np.where(A, 1.0, np.inf)
        np.fill_diagonal(D, 0.0)
        for m in range(N):
            D = np.minimum(D, D[:, m, None] + D[None, m, :])
        self.dist = D

        # Cumulative transition rows, precomputed for fast inverse-CDF sampling
        # in the simulator: cum[i, k, :] is the CDF over the next levels.
        self.cum = np.cumsum(self.Tun, axis = 2)

    def init_positions(self):
        """
        Spread the fleet uniformly across the ring.

        Starting every vehicle in zones 0...K-1 clusters the fleet, and because
        moves at most 2 zones per step, the far side of a larger ring is
        unreachable within the horizon. Those zones escalate to the cap and stay
        there, so per-zone cost grows with N and scale-invariance is destroyed.
        """

        return np.round(np.linspace(0, self.N, self.K, endpoint = False)).astype(int)
        
def simulate(inst, score_fn, n_roll = 300, seed = 0):
    """
    Monte-Carlo evaluation of a policy at a scale where exact evaluation is impossible.

    `score_fn(k, t) -> (R, N)` gives each zone's priority score for every rollout
    at once, where `k` is the (R, N) array of current escalation levels.

    Assignment uses scipy's Hungarian Algorithm (`linear_sum_assignment`) per rollout,
    because at these sizes the joint action set cannot be enumerated the way it is
    for the exact solver.

    Return (mean total cost, standard error, fraction of illegal picks). The last number
    should be 0. It is a guard against the assignment solver being forced into an
    unreachable pairing.
    """
    rng = np.random.default_rng(seed)
    N, K, T = inst.N, inst.K, inst.T
    R = n_roll

    k = np.zeros((R, N), dtype = int)                           # all zones quiescent
    pos = np.tile(inst.init_positions(), (R, 1))                # fleet spread uniformly
    tot = np.zeros(R)
    bad = 0

    for t in range(T):
        # Pay this step's cost for every rollout: sum over zones of r * w(level)
        tot += (inst.r * inst.level[k]).sum(axis = 1)

        S = score_fn(k, t)                                      # (R, N) priority scores
        tgt = np.empty((R, K), dtype = int)
        for rr in range(R):
            # Which zones each vehicle in this rollout can reach
            reach = inst.adj[pos[rr]]                           # (K, N) bool

            # Hungarian minimizes, negate the benefit
            # The benefit is: score(target) - beta * travel_distance.
            C = -(S[rr][None, :] - inst.beta * inst.dist[pos[rr]])

            # Illegal pairings priced out of contention
            C = np.where(reach, C, BIG)

            ri, ci = lsa(C)
            tgt[rr] = ci

            # Guard: count any chosen pairing that was actually illegal
            bad += int((C[ri, ci] >= BIG / 2).sum())

        # Advance all zones in all rollouts at once by inverse-CDF sampling.
        # C3[r, i, :] is the CDF for zone i at its current level in rollout r;
        # counting how many CDF entries a uniform draw exceeds gives the sample.

        u = rng.random((R, N))
        C3 = inst.cum[np.arange(N)[None, :], k]                 # (R, N, L)
        nk = (u[:, :, None] > C3).sum(axis = 2)

        # Visited zones reset to 0, overwriting whatever was sampled.
        np.put_along_axis(nk, tgt, 0, axis = 1)

        k, pos = nk, tgt

    return tot.mean(), tot.std(ddof = 1) / np.sqrt(R), bad / (R * K * T)

def make_scores(inst, kind, lam = None):
    """
    Build a score function for `simulate`.

    'greedy' -> myopic risk-weighted damage, r_i * w(k_i)
    'index'  -> precomputed index lam[zone, level, t]

    `lam[np.arange(N)[None, :], k, t]` pairs zone index j with
    the level k[r, j] for every rollout r, producing an (R, N)
    array in one shot.
    """
    if kind == "greedy":
        return lambda k, t: inst.r * inst.level[k]
    if kind == "index":
        return lambda k, t: lam[np.arange(inst.N)[None, :], k, t]       #type: ignore
    raise ValueError(kind)
