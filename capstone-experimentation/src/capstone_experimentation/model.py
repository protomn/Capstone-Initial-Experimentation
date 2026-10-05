"""
The environment and exact solver.

A miniature version of the Multi-Agent patrol problem, small enough that the mathematically
perfect answer can be calculated by dynamic programming. Every policy is then scored against
that perfect answer instead of against another heuristic, which is the entire point of the
spike. A number like "12% above optimal" is a fact, while "4% better than greedy" is only a
comparison.

THE MODEL
-----------
N zones sit on a graph. K vehicles sit on zones, at most one per zone. Each step a vehicle
stays put or leaves to an adjacent zone (self-loops make "stay" always legal), and all 
destinations must be distinct.

Each zone i carries an escalation level k_i in {0, ..., L - 1}:
    k = 0                       nothing needs attention
    0 -> 1 w.p. p_i             a need arises
    k -> k+1 w.p. e_i           it worses (for k >= 1)
    k = L - 1                   absorbing, cannot worsen further, only to be cleared
    visited                     resets to 0

Cost paid at each step is sum_i r_i * w(k_i) where r_i is a per-zone risk weight and w is
set by `cost_pow` (1 -> linear in neglect, 2 -> accelerating harm).

Objective: minimize the expected total cost over a T-step horizon.
"""

import itertools
import numpy as np
from typing import Type

class Instance:
    """
    A single concrete problem instance: graph, zone dynamics, costs, and the enumerated state/
    action spaces the exact solver needs.
    """

    def __init__(self, N=6, K=2, L=4, T=20, seed=0, graph="ring2",
                 risk_spread=1.0, p_lo=0.15, p_hi=0.55, beta=0.0,
                 esc_lo=1.0, esc_hi=1.0, cost_pow=1.0):
        # A seeded generator, so every instance is reproducible from its seed
        # alone. All randomness in the instance is drawn from this one stream.
        rng = np.random.default_rng(seed)
        self.N, self.K, self.L, self.T = N, K, L, T
        
        # beta is the weight on travel cost inside a policy's scoring rule. It
        # lives on the instance purely so policies can read it without extra
        # plumbing. It does not change the environment or the optimum.
        self.beta = beta

        # risk weights
        # risk_spread interpolates between homogeneous and heterogeneous zones
        #   0.0 -> every r_i == 1               (zones differ only in dynamics)
        #   1.0 -> r_i ~ Uniform(0.25, 1.75)    (a 7x spread in importance)
        # when risky weights vary widely, they dominate every scoring rule's
        # ranking, which makes an index policy and myopic polivy pick the same
        # zones (See EXP 01).
        base = np.ones(N)
        het = rng.uniform(0.25, 1.75, size=N)
        self.r = (1.0 - risk_spread) * base + risk_spread * het

        # zone dynamics
        # p_i is the probability a quiescent zone develops a need this step
        self.p = rng.uniform(p_lo, p_hi, size=N)

        # e_i is the probability an already escalating zone worsens by one level
        # esc_lo == esc_hi == 1.0 gives deterministic escalation, in which case
        # the index degenerates to the myopic score (the marginal value of serving
        # becomes nearly proportional to the current level), so the index and 
        # greedy policies coincide. Setting a range can break this degeneracy.
        self.e = rng.uniform(esc_lo, esc_hi, size=N)

        # graph
        self.adj = self._build_graph(graph, rng)
        # staying put must always be an option, otherwise the action set can be
        # empty and the whole formulation can break. Self-loops guarantee it.
        assert np.all(self.adj.diagonal()), "self-loops (stay) must be allowed."

        # cost structure
        # level[k] is the harm weight of being at escalation level k
        self.level = np.arange(L, dtype=float) ** cost_pow

        # A zone config is an N-tuple of levels, so the space of
        # the config is an N-dim array of shape (L, L, ..., L).
        # Representing it as a dense tensor (instead of a flat vector)
        # makes the kroenecker trick in `expect` possible.
        self.cfg_shape = (L, ) * N
        self.n_cfg = L ** N

        # precompute the per-step cost for every zone config at once.
        # cost is additive across zones and can be built by adding one zones
        # contribution at a time, broadcasting along the given zone's axis
        #   shape = [1, 1, ..., L, ..., 1] puts the L values on the ith axis only
        #   and numpy broadcasts them across all other axes.
        cost = np.zeros(self.cfg_shape)
        for i in range(N):
            shape = [1] * N
            shape[i] = L
            cost += self.r[i] * self.level.reshape(shape)
        self.cost_cfg = cost

        # vehicle config
        # a vehicle config is an ordered tuple of distinct zone indices.
        # one vehicle per zone is enforced by construction. 
        # ordered -> carries order of K! symmetry redundancy which costs 
        # speed but keep indexing simple.
        self.vcfgs = [t for t in itertools.permutations(range(N), K)]

        # reverse lookup: tuple -> integer index
        # finds successor vehicle config after an action
        self.vidx = {t: j for j, t in enumerate(self.vcfgs)}
        self.n_v = len(self.vcfgs)

        # legal join actions
        # for each vehicle config, we enumerate every legal joint move.
        # joint move: tuple of target zones, one per vehicle, each target
        # is adjacent or equal to the vehicle's current zone and all 
        # targets are distinct
        self.actions = []
        for vc in self.vcfgs:
            # the neighbors (incl self) of its current position
            opts = [np.flatnonzero(self.adj[pos]) for pos in vc]
            #cartesian prod of per-vehicle options, filtered for distinction
            acts = [t for t in itertools.product(*opts) if len(set(t)) == K]
            # stay put must survive the filter, current posns are distinct by
            # construction
            assert vc in acts, "staying put is always feasible"
            self.actions.append(acts)

        # per-zone transition kernels
        # Tun[i] is LxL transition matrix for zone i when it is NOT visited.
        # row k -> gives the distribution over next levels given curr level k
        self.Tun = np.zeros((N, L, L))
        for i in range(N):
            # from quiescent, stay quiescent or develop a need
            self.Tun[i, 0, 0] = 1.0 - self.p[i]
            self.Tun[i, 0, 1] = self.p[i]
            #from an active level below the cap: worsen or hold
            for k in range(1, L - 1):
                self.Tun[i, k, k + 1] = self.e[i]
                self.Tun[i, k, k] = 1.0 - self.e[i]
            # the top level is absorbing when unvisited
            self.Tun[i, L - 1, L - 1] = 1.0

        # Tvis is the kernel for a visited zone, whatever the level was, it
        # becomes 0, column 0 is all ones, same for every zone, hence no index
        self.Tvis = np.zeros((L, L))
        self.Tvis[:, 0] = 1.0

        # all pairs shortest-path hop distances. only used by policies that charge
        # for travel; the environment itself is single hop per step
        self.dist = self._hops()

    def _build_graph(self, kind, rng):
        """
        Adjacency matrix, boolean, always including the diagonal (self-loops).

        "ring2" connects each node to distance 1 and 2, so on N = 5, it reaches
        every other node and is the complete graph. it is only genuinely sparse
        for N >= 8. mistaking it for a constrained grpah at N = 5 was a error
        encountered during development.
        """
        N = self.N
        A = np.eye(N, dtype=bool)   # starts with self-loops only
        if kind == "ring":
            # cycle: each node joined to its two immediate neighbors. degree 3
            # counting the self-loop. restrictive movement
            for i in range(N):
                A[i, (i + 1) % N] = A[(i + 1) % N, i] = True
        elif kind == "ring2":
            # cycle plus chords at distance 2, degree 5 counting the self loop
            for i in range(N):
                A[i, (i + 1) % N] = A[(i + 1) % N, i] = True
                A[i, (i + 2) % N] = A[(i + 2) % N, i] = True
        elif kind == "complete":
            # no movement constraints at all
            A[:] = True
        elif kind == "random":
            # a hamiltonian cycle forced on top so the graph is guaranteed connected
            for i in range(N):
                for j in range(i + 1, N):
                    if rng.random() < 0.45:
                        A[i, j] = A[j, i] = True
            for i in range(N):
                A[i, (i + 1) % N] = A[(i + 1) % N, i] = True
        else:
            raise ValueError(kind)
        
        return A
    
    def _hops(self):
        """
        All-pairs shortest path lengths by Floyd-Warshall
        """
        N = self.N
        # Directed edges cost 1 hop; non-edges start at inf
        D = np.where(self.adj, 1.0, np.inf)
        np.fill_diagonal(D, 0.0) # zero cost to stay where you are
        # standard floyd warshall, vectorized over the outer 2 loops:
        # for each intermediate node m, check if routing i->m->j beats 
        # i->j
        for m in range(N):
            D = np.minimum(D, D[:, m, None] + D[None, m, :])
        return D
    
# THE EXPECTATION OPERATOR
def expect(V, inst, visited_mask):
    """
    Return R[V(next config)] as a function of the current config

    Given a fixed action, the zones evolve independently. The action fixes which zones
    are visited, after which zone 1's transition tells you nothing about zone 5's. So
    the joint kernel factorizes as follows:

        P(all next levels | all current levels) = prod_i P_i(next_i | current_i)
    
    which turns out to be the Kronecker product of N tiny L x L matrices. Applying
    kronecker product to a vector does not require forming the big matrix. If V is 
    viewed as an N-dim tensor of shape (L, ) * N, each small matrix is applied along
    its own axis, one at a time.

    This allows the cost to fall from O(L^N * L^N) to O(N * L^(N+1)), approximately
    170 times fewer operations.
    The function is cross-verified against brute-force enumeration of all successors in 
    test/test_correctness.py.
    """
    out = V
    for i in range(inst.N):
        # pick the right kernel for this zone under this action
        Ti = inst.Tvis if visited_mask[i] else inst.Tun[i]
        # tensordot(Ti, out, axes=([1], [i])) contracts Ti's second index against
        # out's axis i. The result's axis 0 is Ti's first index, and the remaining
        # axes are out's axes with axis i removed.

        # The contraction leaves the new axis at position 0, and moveaxis puts it
        # back where zone i belongs. the wrong ordering here permutes the zones.
        out = np.moveaxis(np.tensordot(Ti, out, axes=([1], [i])), 0, i)
    return out

def action_visited_masks(inst):
    """
    Precompute, for every pair, which zones end up visited.

    Depends on action's target zones, not on the zone levels, so it can be
    hoisted out of the time loop entirely. Returned as a list-of-lists 
    indexed [vehicle_config_index][action_index].
    """
    masks = []
    for vc_i in range(inst.n_v):
        per = []
        for tgt in inst.actions[vc_i]:
            m = np.zeros(inst.N, dtype=bool)
            m[list(tgt)] = True     # the targets of the move are the visited zones
            per.append(m)
        masks.append(per)
    return masks

# BACKWARD INDUCTION: exactl= optimum or exact eval of a fixed policy
def backward(inst, masks, rule=None, randomize=False):
    """
    Solve or eval by dynamic programming ocer the full joint state space.

    Three modes, selected by arguments:
    rule=None, randomize=False --> minimize over actions, the exact optimum
    randomize=True             --> average over actions: a uniform random policy
    rule=<callable>            --> follow that decision rule: exact evaluation
                                   of the policy (no Monte Carlo, no noise)
    
    `rule(t, vc_i, n_actions)` must return an int array of shape cfg_shape giving, 
    for every zone config, which action index to take
    Returns V with shape cfg_shape + (n_v, ): the expected cost-to-go at t=0 for
    every (zone config, vehicle config) pair.

    The recursion is finite-horizon Bellman equation:
        V_t(s) = cost(s) + [min or E or chosen] over a of E[V_{t+1}(s')]
    with V_t = 0. working backwards from the horizon gives the exact answer.
    """
    # Terminal condition, nothing left to pay after the horizon
    V = np.zeros(inst.cfg_shape + (inst.n_v, ))

    for t in range(inst.T - 1, -1, -1):     # walk backwards in time
        Vn = np.empty_like(V)
        for vc_i, vc in enumerate(inst.vcfgs):
            acts = inst.actions[vc_i]
            # Q[a] holds, for action a, the expected continuation value as a 
            # function of the current zone config.
            Q = np.empty((len(acts), ) + inst.cfg_shape)
            for a_i, tgt in enumerate(acts):
                # where the vehicles end up is deterministic given the action
                nxt = inst.vidx[tgt]
                # marginalize the stochastic zone transitions
                Q[a_i] = expect(V[..., nxt], inst, masks[vc_i][a_i])

            if randomize:
                # uniform random policy
                sel = Q.mean(axis=0)
            elif rule is None:
                # take the cheapest action per config
                sel = Q.min(axis=0)
            else:
                # fixed policy -> gather the chosen action's value per config
                choice = rule(t, vc_i, len(acts))
                sel = np.take_along_axis(Q, choice[None, ...], axis=0)[0]

            # Bellman
            Vn[..., vc_i] = inst.cost_cfg + sel
        V = Vn
    return V

def initial_value(V, inst):
    """
    Read the expected total cost from the canonical start state:
    every zone quiescent, vehicles parked on zones 0...K-1
    """
    return float(V[(0, ) * inst.N + (inst.vidx[tuple(range(inst.K))], )])

# THE INDEX (per zone priority score)

def compute_index(inst, q):
    """
    Marginal-cost-of-neglect index, lambda_i(k, t).

    WHAT AN INDEX IS
    ----------------
    The classical Whittle index asks: what price would make me indifferent between
    serving this zone and not serving it? Higher indifference price means more urgent.
    Computing it exactly requires a bisection search on that price.

    We use a closely related quantity that is directly computable and is what the
    reference work's movement index is built from. Consider zone i ALONE, and let

        Z_i(k, t) = the expected future cost of zone i from level k with t steps left,
                    assuming it gets served with probability q_i each step.

    the define

        lambda_i(k, t) = Z_i(k, t) - Z_i(0, t)

    which reads: how much future harm you avoid by resetting this zone to clean
    RIGHT NOW. Zero for an already clean zone, increasing as the zone worsens.

    WHY THIS BEATS A MYOPIC SCORE
    -----------------------------
    Z_i depends on p_i (how fast trouble returns) and e_i (how fast it worsens).
    Two zones at the same level with the same risk weight get different indices if
    one deteriorates faster. A myopic score r_i * w(k) cannot express that.

    q IS THE EXPERIMENTAL VARIABLE
    ------------------------------
    q_i is the zone's assumption about how much attention it will get.
        q_i = K/N               the classical relaxation (everyone gets their fair share).
        q_i = realised rate     matiching aware (what actually happens under the policy)
    Substituting the second for the first is the contribution tested in experiment 03,
    and it makes things worse. See FINDINGS

    Returns an array of shape (N, L, T + 1)
    """
    N, L, T = inst.N, inst.L, inst.T

    # Z[i, k, t] -> the extra terminal T + 1 slot is 0
    Z = np.zeros((N, L, T + 1))
    for i in range(N):
        for t in range(T - 1, -1, -1):      # backwards, same as the joint solver
            # Cost of sitting at each level this step, as a length-L vector
            c = inst.r[i] * inst.level

            # If served, the zone is reset, so the continuation is Z at level 0
            served = Z[i, 0, t + 1]

            # If not served, apply the unvisited kernel. The matrix vector product
            # Tun[i] @ Z[i, :, t + 1] gives, each current level, the expected
            # continuation. Reusing inst.Tun here (rather than re-deriving the
            # dynamics) means the single-zone and joint models cannot drift apart.
            # A single source of truth for the transition structure.
            unserved = inst.Tun[i] @ Z[i, :, t + 1]
            Z[i, :, t] = c + q[i] * served + (1 - q[i]) * unserved

    # Subtract the level-0 value at each time to get the benefit of resetting.
    # Z[:, 0:1, :] keeps the middle axis so broadcasting lines up.
    lam = Z - Z[:, 0:1, :]

    # Serving can never cost more future harm than not serving, so the index is
    # non-negotiable. A violation means the recursion above is wrong.
    assert np.all(lam >= -1e-9), "index must be non-negative"
    return lam

def zone_score_tensors(inst, lam, t):
    """
    Broadcast lam_i(k_i, t) into full configuration tensors, one per zone.

    A policy needs the score of zone "z" as a function of the whole configuration,
    so it can be added up across the zones an action visits and compared across
    all configuration at once. Reshaping the length-L vector onto axis z and
    broadcasting achieves that with no copying.
    """ 
    out = []
    for i in range(inst.N):
        shape = [1] * inst.N
        shape[i] = inst.L
        out.append(np.broadcast_to(lam[i, :, t].reshape(shape), inst.cfg_shape))
    return out
