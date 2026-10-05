"""
MODULE MAP:
    model.py -> the environment and dp solver
    policies.py -> decision rules and comparison ladder
    lp.py -> LP relaxation LB, for scales that cannot be reached by the exact solver
    scaling.py -> large-N instance and Monte-Carlo evaluator
    staleness.py -> multi-epoch travel, separate model and solver

"""

from . import lp, model, policies, scaling, staleness
from .lp import solve_lp_bound
from .model import (Instance, action_visited_masks, backward, compute_index,
                    expect, initial_value, zone_score_tensors)
from .policies import (ORDER, evaluate, full_ladder, gap, gaps,
                       make_greedy_rule, make_index_greedyrepair_rule,
                       make_index_optassign_rule, matched_index, realised_service_rates)

__version__ = "1.0.0"

__all__ = [
    # submodules
    "model", "policies", "lp", "scaling", "staleness",
    # environment + exact solver
    "Instance", "expect", "action_visited_masks", "backward", "initial_value",
    "compute_index", "zone_score_tensors",
    # policies
    "make_greedy_rule", "make_index_optassign_rule", "make_index_greedyrepair_rule",
    "realised_service_rates", "matched_index", "full_ladder", "evaluate", "gap", 
    "gaps", "ORDER",
    # bound
    "solve_lp_bound"
]