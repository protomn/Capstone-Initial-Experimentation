"""
LP (fluid) relaxation lower bound.

Exact value iteration requires L^N * N!/(N-K) states, which dies at about N=7. So it
cannot say anything about a claim concerning large systems.
We establish a lower bound, a number provably below the optimum. Then:
    (policy_cost - bound)/bound
is an UPPER bound on how suboptimal the policy is. If it is small, the policy is
definitely good.

We solve an easier problem that permits things the real on forbids. More freedom
cannot cost more, so the relaxed optimum sits at or below the true optimum.

WHAT IS BEING RELAXED?
------------------------
The problem is written as a LP over EXPECTED quantities rather than
over actual integer states. The variables are probabilities and expected flows.
This means LP can do physically impossible things. 
It is movement-aware: vehicle flow is confined to graph edges, so this
is not the naive budget-only Whittle bound that ignore geography.
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog as lp

def solve_lp_bound(inst, init_pos=None, verbose=False):
    """
    Return the LP relaxation's optimal value: a valid lower bound on the true
    optimum for this instance.

    `inst` only needs N, K, L, T, adj, Tun, r, level; so both the small
    `Instance` and the large `BigInstance` work here unchanged.

    `init_pos` is the expected vehicle count per zone before the first move. It
    matters because clustering the fleet makes the far side of a large graph
    unreachable within the horizon, which destroys scale invariance.
    See scaling.BigInstance.init_positions.
    """

    N, L, T, K = inst.N, inst.L, inst.T, inst.K

    if init_pos is None:
        # Default: vehicles on zones 0...K-1, works for small N
        # falls apart for large N
        init_pos = np.zeros(N)
        init_pos[:K] = 1.0

    # Flow variables exist only for the edges (including self-loops), which helps
    # keep the LP sparse
    edges = [(i, j) for i in range(N) for j in range(N) if inst.adj[i, j]]
    eidx = {e: m for m, e in enumerate(edges)}
    nE = len(edges)

    # We use a single flat vector to hold all four blocks contiguously. The lambdas
    # convert semantic indices into positions in the vector
    ny = N * L * T      # y
    nv = N * L * T      # v
    npos = N * T        # pos
    ng = nE * T         # g
    total = ny + nv + npos + ng

    off_y, off_v, off_pos, off_g = 0, ny, ny + nv, ny + nv + npos
    Y = lambda i, k, t: off_y + (i * L + k) * T + t
    V = lambda i, k, t: off_v + (i * L + k) * T + t
    P = lambda i, t: off_pos + i * T + t
    G = lambda i, j, t: off_g + eidx[(i, j)] * T + t

    # equality constraints
    rows, cols, vals, beq = [], [], [], []
    rc = 0 # row counter

    def add(entries, rhs):
        """
        append one equality row: sum(coeff * var) == rhs
        """
        nonlocal rc
        for c, a in entries:
            rows.append(rc)
            cols.append(c)
            vals.append(a)
        beq.append(rhs)     #type: ignore
        rc += 1

    # modelling each zone's level distribution as a probability distribution
    for i in range(N):
        for t in range(T):
            add([(Y(i, k, t), 1.0) for k in range(L)], 1.0)

    # a zone is visited exactly when a vehicle is there
    # so the visited area mass summed over levels is the expected occupancy
    for i in range(N):
        for t in range(T):
            add([(V(i, k, t), 1.0) for k in range(L)] + [(P(i, t), -1.0)], 0.0)

    # the fleet neither grows nor shrinks
    for t in range(T):
        add([(P(i, t), 1.0) for i in range(N)], float(K))

    # flow conservation, bi-directional - this makes the bound movement aware
    # mass can move along edges
    for t in range(T):
        for i in range(N):
            out = [(G(i, j, t), 1.0) for j in range(N) if inst.adj[i, j]]
            if t == 0:
                # at the first step, out flow comes from a given initial layout
                add(out, float(init_pos[i]))
            else:
                # from wherever the vehicles were last step
                add(out + [(P(i, t - 1), -1.0)], 0.0)
        
        for j in range(N):
            inn = [(G(i, j, t), 1.0) for i in range(N) if inst.adj[i, j]]
            add(inn + [(P(j, t), -1.0)], 0.0)

    # Zone dynamics: for each zone and the next level k', the mass arriving at
    # k' is the visited mass (all of which lands at level 0) plus the unvisited
    # mass pushed through the unvisited kernel
    for i in range(N):
        for t in range(T - 1):
            for kp in range(L):
                ent = [(Y(i, kp, t + 1), -1.0)]
                for k in range(L):
                    # Unvisited mass at level k contrubutes Tun[k, k']
                    coef_y = inst.Tun[i, k, kp]
                    # Visited mass is (y - v) removed and v added at level 0, so
                    # v's coefficient is (indicator of k'==0) minus Tun[k, k'].
                    coef_v = (1.0 if kp == 0 else 0.0) - inst.Tun[i, k, kp]
                    if coef_y:
                        ent.append((Y(i, k, t), coef_y))
                    if coef_v:
                        ent.append((V(i, k, t), coef_v))
                add(ent, 0.0)

    #Initial condition: every zone starts quiescent
    for i in range(N):
        add([(Y(i, 0, 0), 1.0)], 1.0)

    Aeq = sp.coo_matrix((vals, (rows, cols)), shape=(rc, total)).tocsr()
    beq = np.array(beq)

    # inquality constraint v <= y
    # you cannot visit more mass than is present at that level
    irows, icols, ivals = [], [], []
    r2 = 0
    for i in range(N):
        for k in range(L):
            for t in range(T):
                irows += [r2, r2]
                icols += [V(i, k, t), Y(i, k, t)]
                ivals += [1.0, -1.0]    # v - y = 0
                r2 += 1

    Aub = sp.coo_matrix((ivals, (irows, icols)), shape=(r2, total)).tocsr()
    bub = np.zeros(r2)

    # objective
    # only y carries the cost: expected harm is the level weight times the
    # probability of being at that level.
    c = np.zeros(total)
    for i in range(N):
        for k in range(L):
            for t in range(T):
                c[Y(i, k, t)] = inst.r[i] * inst.level[k]

    # bounds
    lb = np.zeros(total)            # all variables non-negative
    ub = np.full(total, np.inf)     # flows unbounded above (fleet cap binds)
    for i in range(N):
        for t in range(T):
            # The relaxation: occupancy is a real number inside [0, 1], not an integer
            ub[P(i, t)] = 1.0
        for k in range(L):
            for t in range(T):
                ub[Y(i, k, t)] = 1.0        # probabilities
                ub[V(i, k, t)] = 1.0

    # HIGHS is Scipy's default and handles large sizes well (about 45k variables at
    # N = 160). np.column_stack gives linprog the (lb, ub) pairs it expects.
    res = lp(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
             bounds=np.column_stack([lb, ub]), method='highs')
    
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    if verbose:
        print(f"LP: {total} vars, {rc} eq, {r2} ineq -> {res.fun:.6f}")

    return float(res.fun)