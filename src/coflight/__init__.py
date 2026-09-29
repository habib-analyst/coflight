"""Co-failure ceiling estimation for multi-agent LLM systems.

The central quantity is beta: the rate at which every model in a pool is
wrong on the same query. For any selection policy whose output is one member
model's answer -- a router, a majority vote, a cascade -- accuracy cannot
exceed 1 - beta. Pairwise error correlation rho cannot identify beta, so
reporting rho alone gives no information about the ceiling.
"""

from coflight.decision import Verdict, decide
from coflight.economics import (
    ModelEconomics,
    cheapest_meeting,
    delegation_savings,
    model_economics,
    pareto_front,
)
from coflight.estimate import (
    BetaCertificate,
    clopper_pearson,
    co_failure_rate,
    common_shock_fit,
    mcnemar_exact,
    oracle_gain,
    paired_gap,
    pairwise_correlation,
    pool_report,
    rank_models,
)
from coflight.matrix import Matrix, load_matrix

__version__ = "0.2.0"

__all__ = [
    "BetaCertificate",
    "Matrix",
    "ModelEconomics",
    "Verdict",
    "cheapest_meeting",
    "clopper_pearson",
    "co_failure_rate",
    "common_shock_fit",
    "decide",
    "delegation_savings",
    "load_matrix",
    "mcnemar_exact",
    "model_economics",
    "oracle_gain",
    "paired_gap",
    "pairwise_correlation",
    "pareto_front",
    "pool_report",
    "rank_models",
    "__version__",
]
