"""Beta estimation, the Clopper-Pearson certificate, and pool reporting.

The certificate matters more than the point estimate. A pool of 17 models
that happened to co-fail on 2 of 200 queries is not distinguishable from one
that co-failed on 8; only the interval separates them, and the interval is
what tells an operator whether the ceiling is worth caring about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Sequence, TypedDict

import numpy as np

from coflight.matrix import Matrix
from coflight.numerics import betaincinv


# Bounds for the common-shock fit. pi is a moment estimate that becomes
# ill-conditioned outside these bounds; see common_shock_fit.
_SHOCK_MIN_POOL = 8
_SHOCK_MAX_PI = 0.75


class PairedGap(TypedDict):
    """Result of the exact McNemar test on two models' discordant queries."""

    best_only: int
    second_only: int
    p_value: float
    significant: bool


class OracleGain(TypedDict):
    """Decomposition of the best model's error mass into rescuable vs shared."""

    best_accuracy: float
    best_model: str
    best_wrong: float
    beta: float
    oracle_gain: float


class ShockFit(TypedDict):
    """Common-shock decomposition, plus an identifiability flag and reason."""

    pi: float
    identifiable: bool
    note: str
    alpha0: float
    alpha_min: float
    alpha_max: float
    beta_independent: float
    beta_fit: float
    beta_observed: float


@dataclass(frozen=True, slots=True)
class BetaCertificate:
    """Co-failure rate for a pool, with an exact binomial confidence interval."""

    pool_size: int
    n_queries: int
    n_all_wrong: int
    beta: float
    ci_low: float
    ci_high: float
    confidence: float = 0.90

    @property
    def ceiling(self) -> float:
        """Accuracy no selection policy can exceed on this workload."""
        return 1.0 - self.beta

    @property
    def ceiling_low(self) -> float:
        """Optimistic end: uses the smallest credible beta."""
        return 1.0 - self.ci_low

    @property
    def all_wrong_queries(self) -> int:
        return self.n_all_wrong

    def summary(self) -> str:
        return (
            f"beta={self.beta:.4f} "
            f"[{self.ci_low:.4f}, {self.ci_high:.4f}] "
            f"({self.n_all_wrong}/{self.n_queries}, k={self.pool_size})"
        )


def clopper_pearson(
    successes: int, n: int, confidence: float = 0.90
) -> tuple[float, float]:
    """Exact (conservative) binomial confidence interval.

    Deliberately exact rather than Wilson or normal-approximation: the
    all-wrong events are rare by construction, which is exactly the regime
    where the normal approximation is worst and where an optimistic interval
    would understate risk.
    """
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} out of range for n={n}")

    alpha = 1.0 - confidence
    low = 0.0 if successes == 0 else betaincinv(successes, n - successes + 1, alpha / 2.0)
    high = (
        1.0
        if successes == n
        else betaincinv(successes + 1, n - successes, 1.0 - alpha / 2.0)
    )
    return (low, high)


def co_failure_rate(correctness: dict[str, dict[str, int]]) -> tuple[int, int]:
    """Count queries where every model in the pool is wrong."""
    total = 0
    for row in correctness.values():
        if row and not any(row.values()):
            total += 1
    return total, len(correctness)


def pairwise_correlation(correctness: dict[str, dict[str, int]]) -> float:
    """Mean pairwise error correlation phi.

    Reported because the field uses it, and to show it is not the binding
    constraint: rho cannot identify beta, so a high rho here does not
    necessarily mean a tight ceiling.
    """
    if len(correctness) < 2:
        return 0.0

    models = sorted(next(iter(correctness.values())))
    n = len(correctness)
    phis: list[float] = []

    for a, b in combinations(models, 2):
        both = neither = only_a = only_b = 0
        for row in correctness.values():
            ea, eb = 1 - row[a], 1 - row[b]
            if ea and eb:
                both += 1
            elif not ea and not eb:
                neither += 1
            elif ea:
                only_a += 1
            else:
                only_b += 1
        # A model that always errs on the same queries as another gives a
        # 2x2 table with an empty margin, so the denominator can be zero.
        denom = (both + only_a) * (only_b + neither) * (both + only_b) * (only_a + neither)
        if denom <= 0:
            continue
        num = both * neither - only_a * only_b
        phi = num / np.sqrt(denom)
        if np.isfinite(phi):
            phis.append(float(phi))
    return float(np.mean(phis)) if phis else 0.0


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value for paired correctness outcomes.

    Comparing two models on the same queries is a paired design, so
    unpaired confidence intervals on each accuracy are needlessly
    conservative: they treat the two models as independent samples when they
    were measured on identical queries. The relevant comparison uses only
    the discordant pairs, where one model is right and the other is wrong.

    Null hypothesis: each discordant query goes to the winner by coin flip,
    so b ~ Binomial(a + b, 0.5).
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2.0**n
    return float(min(1.0, 2.0 * tail))


def paired_gap(
    correctness: dict[str, dict[str, int]], best: str, second: str
) -> PairedGap:
    """Significance of the best-vs-runner-up accuracy gap, paired by query.

    Returns the discordant counts, the exact two-sided p-value, and whether
    the gap survives a 0.05 test.
    """
    best_only = sum(
        1 for row in correctness.values() if row[best] == 1 and row[second] == 0
    )
    second_only = sum(
        1 for row in correctness.values() if row[best] == 0 and row[second] == 1
    )
    p = mcnemar_exact(best_only, second_only)
    return {
        "best_only": best_only,
        "second_only": second_only,
        "p_value": p,
        "significant": bool(p < 0.05),
    }


def common_shock_fit(correctness: dict[str, dict[str, int]]) -> ShockFit:
    """Marshall-Olkin common-shock decomposition of the error law.

    Homogeneous version (all models share one independent error rate) assumes
    homogeneous model quality, which is false for a real frontier pool where
    accuracy spans 0.88 down to 0.30. Under that assumption the fit degenerates
    to alpha0 = 0, which would predict every query is co-hard and contradict the
    observed beta.

    So this uses the heterogeneous form: a query is co-hard with probability pi,
    and otherwise each model errs independently at *its own* rate alpha_i.

        P(query co-hard)               = pi
        P(model i errs | not co-hard)  = alpha_i

    The co-failure rate is then a product of marginals, which is what makes it
    pool-size dependent in the observed way.
    """
    models = sorted(next(iter(correctness.values())))
    k = len(models)
    if k < 2 or not correctness:
        return {
            "pi": 0.0,
            "alpha0": 0.0,
            "alpha_min": 0.0,
            "alpha_max": 0.0,
            "beta_independent": 0.0,
            "beta_fit": 0.0,
            "beta_observed": 0.0,
            "identifiable": False,
            "note": "pool too small to identify a common-shock component",
        }

    rows = np.array(
        [[1 - r[m] for m in models] for r in correctness.values()], dtype=float
    )
    n = rows.shape[0]

    marg = rows.mean(axis=0)  # alpha_i, per model
    alpha = float(marg.mean())

    n_all_wrong = int((rows.sum(axis=1) == k).sum())
    beta_observed = n_all_wrong / n

    # Independent-error counterfactual in log space: the product of (1-alpha_i)
    # underflows to zero once the pool is large (52 models gives ~1e-40), so
    # the naive product is unusable and the log-sum is required.
    log_terms = np.log1p(-np.clip(marg, 0.0, 1.0 - 1e-12))
    log_p_all_correct = float(log_terms.sum())
    beta_indep = float(-np.expm1(log_p_all_correct))  # 1 - exp(log_p)

    # Common-shock mass by moments:
    #   P(all correct) = (1 - pi) * prod_i (1 - alpha_i)
    #   => pi = 1 - P(all correct) / prod_i (1 - alpha_i)
    # Only meaningful when the denominator is not vanishingly small relative to
    # the numerator, i.e. a large enough pool. Flagged rather than guessed.
    p_all_correct = float(np.exp(log_p_all_correct))
    observed_all_correct = 1.0 - float((rows.sum(axis=1) == 0).sum()) / n

    pi = 0.0
    identifiable = False
    note = ""
    if p_all_correct <= 0 or observed_all_correct >= p_all_correct:
        note = "independent counterfactual is degenerate; cannot separate pi"
    elif k >= _SHOCK_MIN_POOL:
        pi_estimate = float(np.clip(1.0 - observed_all_correct / p_all_correct, 0.0, 1.0))
        if pi_estimate > _SHOCK_MAX_PI:
            pi = 0.0
            note = (
                f"moment estimate pi={pi_estimate:.3f} exceeds {_SHOCK_MAX_PI}; "
                f"treated as unidentifiable"
            )
        else:
            pi = pi_estimate
            identifiable = True
    else:
        note = f"pool of {k} below _SHOCK_MIN_POOL={_SHOCK_MIN_POOL}; pi not identifiable"

    beta_fit = pi + (1.0 - pi) * beta_indep if identifiable else beta_observed

    return {
        "pi": pi,
        "identifiable": identifiable,
        "note": note,
        "alpha0": alpha,
        "alpha_min": float(marg.min()),
        "alpha_max": float(marg.max()),
        "beta_independent": beta_indep,
        "beta_fit": beta_fit,
        "beta_observed": beta_observed,
    }


def rank_models(correctness: dict[str, dict[str, int]]) -> list[tuple[float, str]]:
    """Accuracy per model, best first, with a deterministic tie-break.

    Every consumer must rank through this function. Ranking on accuracy alone
    leaves the winner up to dict iteration order, and two different tie-breaks
    silently produced two different "best models" for the same pool.
    """
    n = len(correctness)
    if n == 0:
        return []
    first = next(iter(correctness.values()))
    accs = {m: sum(row[m] for row in correctness.values()) / n for m in first}
    return sorted(((acc, name) for name, acc in accs.items()), reverse=True)


def oracle_gain(correctness: dict[str, dict[str, int]]) -> OracleGain:
    """Best achievable gain over the single best model.

    gain = P(best model wrong) - beta, the part of the best model error
    mass that is *resolvable* by picking a different model, as opposed to the
    co-failure mass no policy can touch.
    """
    if not correctness:
        return {
            "best_accuracy": 0.0,
            "best_model": "",
            "best_wrong": 0.0,
            "beta": 0.0,
            "oracle_gain": 0.0,
        }

    ranked = rank_models(correctness)
    best_acc, best = ranked[0]
    n = len(correctness)
    best_wrong = 1.0 - best_acc
    n_all_wrong, _ = co_failure_rate(correctness)
    beta = n_all_wrong / n
    return {
        "best_accuracy": best_acc,
        "best_model": best,
        "best_wrong": best_wrong,
        "beta": beta,
        "oracle_gain": max(0.0, best_wrong - beta),
    }


@dataclass(slots=True)
class PoolReport:
    """Everything an operator needs before deciding to build an ensemble."""

    models: list[str]
    certificate: BetaCertificate
    rho: float
    best_accuracy: float
    best_model: str
    oracle_gain: float
    shock: dict[str, float] = field(default_factory=dict)
    datasets: list[str] = field(default_factory=list)

    @property
    def ceiling(self) -> float:
        return self.certificate.ceiling

    @property
    def resolved_fraction(self) -> float:
        """Share of the best model error mass that is actually fixable."""
        if self.best_accuracy >= 1.0:
            return 0.0
        return self.oracle_gain / (1.0 - self.best_accuracy)


def pool_report(
    matrix: Matrix,
    models: Sequence[str] | None = None,
    confidence: float = 0.90,
) -> PoolReport:
    # `models is None` rather than `if models`: an explicitly empty pool must
    # raise, not silently fall back to every model in the matrix.
    pool = list(matrix.models_answering_all()) if models is None else list(models)
    if not pool:
        raise ValueError("no models available; matrix may have no fully-answered queries")

    correctness = matrix.correctness(pool)
    if not correctness:
        raise ValueError("no queries answered by every model in the pool")

    n_all_wrong, n = co_failure_rate(correctness)
    beta = n_all_wrong / n
    lo, hi = clopper_pearson(n_all_wrong, n, confidence)

    gains = oracle_gain(correctness)
    return PoolReport(
        models=pool,
        certificate=BetaCertificate(
            pool_size=len(pool),
            n_queries=n,
            n_all_wrong=n_all_wrong,
            beta=beta,
            ci_low=lo,
            ci_high=hi,
            confidence=confidence,
        ),
        rho=pairwise_correlation(correctness),
        best_accuracy=gains["best_accuracy"],
        best_model=gains["best_model"],
        oracle_gain=gains["oracle_gain"],
        shock=common_shock_fit(correctness),
        datasets=matrix.datasets,
    )