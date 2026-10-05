"""
Multi-epoch travel: the staleness problem

WHAT CHANGES
------------
Everywhere else, a vehicle arrives within one step. Here a vehicle may commit to
ANY zone and the trip takes dist(from, to) steps, during which it covers nothing.
That is the mechanism by which the score which justified a dispatch can be out of
date on arrival.

Vehicle state becomes a pair (destination d, eta e):
    e == 0      present at d and available; may commit to an unreserved zone
    e > 0       in transit, committed, counting down
Destinations are reserved, which extends one-vehicle-per-zone to bookings. A vehicle
cannot be sent to z zone another vehicle is already travelling to.

This is still small enough for exact value iteration, so the optimum is ground truth.

TWO CANDIDATE FIXES
-------------------
    charge for travel       subtract beta * travel time -> treats delay as a cost
    project to arrival      score the zone at its expected state on arrival ->
                            treats delay as an estimation problem

RESULT
------
Charging for travel is worth about ~28.5 pp, projecting is worth a further +2.8pp
but only once beta is returned. At a shared beta, the projection looks actively
harmful, because forward projection inflates distant zones' scores (they escalate
during travel) and therefore rewards long trips. The stale-aware variant needs
roughly a third more beta.
"""

import itertools
import numpy as np
from .model import compute_index, expect

class StaleInstance:
    """
    Instance with multi-epoch travel. Ring graph, so travel times are hop
    distances around the cycle and the longest trip is floor(N / 2).
    """

    def __init__(self, N = 5, K = 2, L = 3, T = 10, seed = 0, cost_pow = 2.0,
                 p_lo = 0.15, p_hi = 0.55, esc_lo = 0.3, esc_hi = 0.95,
                 risk_spread = 0.0, beta = 0.0):
        rng = np.random.default_rng(seed)
        self.N, self.K, self.L, self.T, self.beta = N, K, L, T, beta
        self.level = np.arange(L, dtype = float) ** cost_pow
        self.r = (1 - risk_spread) * np.ones(N) + risk_spread * rng.uniform(0.25, 1.75, N)
        self.p = rng.uniform(p_lo, p_hi, N)
        self.e = rng.uniform(esc_lo, esc_hi, N)

        # Ring distances: min of going clockwise or anti-clockwise. Integer.
        # Used directly as a number of time steps.
        self.dist = np.zeros((N, N), dtype = int)
        for i in range(N):
            for j in range(N):
                d = abs(i - j)
                self.dist[i, j] = min(d, N - d)
        self.D = int(self.dist.max())           # longest possible trip = eta ceiling

        # Same zone dynamics as everywhere else
        self.Tun = np.zeros((N, L, L))
        for i in range(N):
            self.Tun[i, 0, 0] = 1 - self.p[i]
            self.Tun[i, 0, 1] = self.p[i]
            for k in range(1, L - 1):
                self.Tun[i, k, k + 1] = self.e[i]
                self.Tun[i, k, k] = 1 - self.e[i]
            self.Tun[i, L - 1, L - 1] = 1.0
        self.Tvis = np.zeros((L, L))
        self.Tvis[:, 0] = 1.0

        self.cfg_shape = (L, ) * N
        cost = np.zeros(self.cfg_shape)
        for i in range(N):
            sh = [1] * N
            sh[i] = L
            cost = cost + self.r[i] * self.level.reshape(sh)
        self.cost_cfg = cost

        # VEHICLE CONFIGS
        # Each vehicle is a (destination, eta) pair. The state space is all
        # K-tuples of such pairs with distinct destinations (the reservation
        # constraint).
        singles = [(d, e) for d in range(N) for e in range(self.D + 1)]
        self.vcfgs = [vc for vc in itertools.permutations(singles, K) if len({s[0] for s in vc}) == K]
        self.vidx = {vc: m for m, vc in enumerate(self.vcfgs)}
        self.n_v = len(self.vcfgs)

        # Legal joint actions and their visited sets
        self.actions, self.visited = [], []
        for vc in self.vcfgs:
            opts = []
            for (d, e) in vc:
                if e == 0:
                    # Available: may pick any zone. Committing to dp sets the eta
                    # to the travel time; picking dp == d means stay put, eta = 0.
                    opts.append([(dp, int(self.dist[d, dp])) for dp in range(N)])
                else:
                    # In transit: just count down
                    opts.append([(d, e - 1)])
            acts, vis = [], []
            for cand in itertools.product(*opts):
                # Reservation constraint -> destinations must stay distinct.
                if len({s[0] for s in cand}) != K:
                    continue
                acts.append(cand)

                # A zone is only covered by a vehical that has arrived (eta 0).
                # Vehicles mid-trip cover nothing -- this single line is
                # the entire staleness mechanism.
                m = np.zeros(N, dtype = bool)
                for (dp, ep) in cand:
                    if ep == 0:
                        m[dp] = True
                vis.append(m)
            assert acts, "no feasible action"
            self.actions.append(acts)
            self.visited.append(vis)

    def travel_of(self, vc_i, a_i):
        """
        Total NEWLY COMMITTED travel time for an action (in-transit vehicles
        already paid, so they are excluded).
        """
        vc, a = self.vcfgs[vc_i], self.actions[vc_i][a_i]
        tot = 0
        for (d, e), (dp, ep) in zip(vc, a):
            if e == 0:
                tot += self.dist[d, dp]
        return tot
    
def backward(inst, rule=None):
    """
    Exact backward induction, same structure as model.backward.

    The only differences are that the successor vehicle configuration is the action
    itself (an action already IS a full (dest, eta) tuple), and the visited mask
    comes from `inst.visited` rather than being derived from target zones.
    """
    V = np.zeros(inst.cfg_shape + (inst.n_v,))
    for t in range(inst.T - 1, -1, -1):
        Vn = np.empty_like(V)

        for vc_i in range(inst.n_v):
            acts = inst.actions[vc_i]
            Q = np.empty((len(acts),) + inst.cfg_shape)
            for a_i, a in enumerate(acts):
                nxt = inst.vidx[a]      # the action is the next vehicle config
                Q[a_i] = expect(V[..., nxt], inst, inst.visited[vc_i][a_i])

            if rule is None:
                sel = Q.min(axis=0)     # exact optimum
            else:
                ch = rule(t, vc_i)
                sel = np.take_along_axis(Q, ch[None, ...], axis=0)[0]

            Vn[..., vc_i] = inst.cost_cfg + sel
        V = Vn
    return V


def initial_value(V, inst):
    """
    Expected total cost from the canonical start: all zones quiescent, vehicles
    present (eta 0) and spread evenly around the ring.

    Spreading rather than clustering matters for the same reason as in the scaling
    study -- a clustered start on a ring leaves distant zones effectively
    unreachable early on.
    """
    start = tuple((int(round(i * inst.N / inst.K)), 0) for i in range(inst.K))
    return float(V[(0,) * inst.N + (inst.vidx[start],)])


def projected_index(inst, lam):
    """
    Lam[z, k, t, tau] = E[ lambda_z(level at t+tau, t+tau) | level now = k ].

    Computed by raising the zone's unvisited kernel to the power tau (the travel
    time) and pushing the index through it. P is built iteratively: P[tau] is
    Tun^tau, so P[tau] @ lam[z, :, t+tau] gives the expected index on arrival.

    NOTE the direction of the bias this introduces: a zone escalates while you
    travel, so the projected value of a DISTANT zone is HIGHER than its current
    value. Forward projection therefore makes long trips look more attractive, and
    needs a larger travel charge to offset. That interaction is the main finding of
    experiment 07.
    """
    N, L, T, D = inst.N, inst.L, inst.T, inst.D
    out = np.zeros((N, L, T + 1, D + 1))

    for z in range(N):
        P = [np.eye(L)]                     # Tun^0 = identity
        for _ in range(D):
            P.append(P[-1] @ inst.Tun[z])   # successive matrix powers
        for tau in range(D + 1):
            for t in range(T):
                tt = min(t + tau, T)        # clamp at the horizon
                out[z, :, t, tau] = P[tau] @ lam[z, :, tt]

    return out


def make_rule(inst, lam, mode, beta=0.0):
    """
    Build a decision rule.

    mode:
      'greedy'   myopic risk-weighted damage, travel ignored
      'now'      index at the zone's CURRENT state (stale-blind)
      'arrival'  index at the zone's PROJECTED state on arrival (stale-aware)

    beta charges for newly committed travel time. Sweep it SEPARATELY for each mode
    -- at a shared beta the comparison between 'now' and 'arrival' is not
    meaningful, for the reason documented in projected_index.
    """
    # Only the 'arrival' mode needs the projected table; building it is not free.
    Lam = projected_index(inst, lam) if mode == "arrival" else None

    myopic = None
    if mode == "greedy":
        myopic = []
        for z in range(inst.N):
            sh = [1] * inst.N
            sh[z] = inst.L
            myopic.append(np.broadcast_to(
                (inst.r[z] * inst.level).reshape(sh), inst.cfg_shape))

    def score_tensor(z, t, tau):
        """
        Zone z's score as a full-configuration tensor, given a travel time tau.
        """
        sh = [1] * inst.N
        sh[z] = inst.L
        if mode == "greedy":
            return myopic[z]    #type: ignore
        if mode == "now":
            return np.broadcast_to(lam[z, :, t].reshape(sh), inst.cfg_shape)

        return np.broadcast_to(Lam[z, :, t, tau].reshape(sh), inst.cfg_shape)   #type: ignore

    def rule(t, vc_i):
        vc = inst.vcfgs[vc_i]
        tots = []
        for a_i, a in enumerate(inst.actions[vc_i]):
            s = np.zeros(inst.cfg_shape)
            for (d, e), (dp, ep) in zip(vc, a):
                if e == 0:
                    # Only vehicles that are actually choosing contribute to the
                    # score. In-transit vehicles are committed, so their
                    # (unavoidable) continuation must not bias the comparison.
                    tau = int(inst.dist[d, dp])
                    s = s + score_tensor(dp, t, tau) - beta * tau
            tots.append(s)
        return np.argmax(np.stack(tots), axis=0).astype(np.intp)
    return rule
