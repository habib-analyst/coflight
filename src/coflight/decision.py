"""The actual decision: is a co-failure ceiling the binding constraint?

Measuring beta tells an operator the ceiling. It does not tell them whether
that ceiling is what is stopping them, and on real pools it usually is not.
The binding constraint is the accuracy gap between the best model and the
second best, because a selection policy must pay that gap to capture any of
the oracle gain at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coflight.estimate import (
    BetaCertificate,
    clopper_pearson,
    co_failure_rate,
    common_shock_fit,
    oracle_gain,
    paired_gap,
    pairwise_correlation,
    rank_models,
)
from coflight.matrix import Matrix

# An oracle gain below this is not worth a routing system's complexity, no
# matter how open the ceiling is. At a 0.98 baseline the remaining 2% is not
# recoverable headroom, it is measurement noise territory.
_MIN_WORTHWHILE_GAIN = 0.01


@dataclass(slots=True)
class Verdict:
    """Whether orchestration is worth building, and what actually limits it."""

    label: str
    reason: str
    beta: BetaCertificate
    best_accuracy: float
    second_accuracy: float
    best_model: str
    accuracy_gap: float
    oracle_gain: float
    ceiling_gain: float
    rescuable_errors: int
    best_wrong: int
    n_queries: int
    rho: float = 0.0
    shock: dict[str, float] = field(default_factory=dict)
    gap_test: dict[str, float] = field(default_factory=dict)

    @property
    def binding_constraint(self) -> str:
        if self.ceiling_gain <= 0:
            return "co-failure ceiling (unresolvable)"
        if self.accuracy_gap <= 0:
            return "no runner-up (nothing to route to)"
        return "accuracy gap (routing headroom)"

    def render(self) -> str:
        c = self.beta
        if self.gap_test:
            g = self.gap_test
            gap_line = (
                f"  runner-up           {self.second_accuracy:.4f}   "
                f"(gap {self.accuracy_gap:+.4f}, McNemar p={g['p_value']:.4f}"
                f"{'  SIGNIFICANT' if g['significant'] else '  not significant'})"
            )
        else:
            gap_line = (
                f"  runner-up           {self.second_accuracy:.4f}   "
                f"(gap {self.accuracy_gap:+.4f})"
            )
        return "\n".join(
            [
                f"  beta (co-failure)   {c.beta:.4f}  [{c.ci_low:.4f}, {c.ci_high:.4f}]"
                f"   {c.n_all_wrong}/{c.n_queries} queries, k={c.pool_size}",
                f"  rho (pairwise)      {self.rho:.4f}   (field standard; cannot see beta)",
                f"  best model          {self.best_accuracy:.4f}  {self.best_model}",
                gap_line,
                f"  accuracy ceiling    {c.ceiling:.4f}   (1 - beta)",
                f"  ceiling gain        {self.ceiling_gain:+.4f}",
                f"  oracle gain         {self.oracle_gain:+.4f}"
                f"   (best wrong {self.best_wrong}/{self.n_queries},"
                f" rescuable {self.rescuable_errors})",
                f"  binding constraint  {self.binding_constraint}",
                "",
                f"  VERDICT: {self.label}",
                f"  {self.reason}",
            ]
        )


def decide(
    matrix: Matrix,
    models: list[str] | None = None,
    confidence: float = 0.90,
) -> Verdict:
    # `models is None` rather than `if models`: an explicitly empty pool must
    # raise, not silently fall back to analysing every model in the matrix.
    pool = list(matrix.models_answering_all()) if models is None else list(models)
    if not pool:
        raise ValueError("no models available")

    correctness = matrix.correctness(pool)
    if not correctness:
        raise ValueError("no queries answered by every model in the pool")

    n = len(correctness)
    # Rank through the shared helper so decide() and oracle_gain() can never
    # name different best models for the same pool.
    ranked = rank_models(correctness)
    best_acc, best_name = ranked[0]
    second_acc = ranked[1][0] if len(ranked) > 1 else 0.0

    n_all_wrong, _ = co_failure_rate(correctness)
    lo, hi = clopper_pearson(n_all_wrong, n, confidence)
    cert = BetaCertificate(
        pool_size=len(pool),
        n_queries=n,
        n_all_wrong=n_all_wrong,
        beta=n_all_wrong / n,
        ci_low=lo,
        ci_high=hi,
        confidence=confidence,
    )

    best_wrong = sum(1 for row in correctness.values() if row[best_name] == 0)
    rescuable = sum(
        1 for row in correctness.values() if row[best_name] == 0 and any(row.values())
    )
    gains = oracle_gain(correctness)
    ceiling_gain = best_wrong / n - cert.beta
    gap = best_acc - second_acc
    best_wrong_rate = best_wrong / n

    runner_up = ranked[1][1] if len(ranked) > 1 else None
    gap_test = paired_gap(correctness, best_name, runner_up) if runner_up else {}

    # Precedence runs from the most fundamental constraint to the most
    # discretionary, so a verdict is never masked by a softer warning:
    #   1. the co-failure ceiling caps every policy equally
    #   2. an exact tie leaves nothing to route on
    #   3. a real but tight ceiling, reported even if the gap is fine
    #   4. the gap is not statistically real, so routing on it is a coin flip
    #   5. real headroom exists but is too small to be worth the system
    if cert.beta >= best_wrong_rate:
        label = "DO NOT BUILD"
        reason = (
            f"Every query the best model misses is also missed by the whole pool "
            f"({best_wrong}/{n}). Orchestration cannot help; the ceiling binds."
        )
    elif gap <= 0.005:
        label = "DO NOT BUILD"
        reason = (
            f"Best and runner-up are within {gap:.3f}. There is no accuracy gap to "
            f"route on, so the {gains['oracle_gain']:+.3f} oracle gain is unreachable "
            f"by any selection policy."
        )
    elif cert.beta > 0.02:
        label = "CAUTION"
        reason = (
            f"{cert.n_all_wrong}/{n} queries defeat the entire pool, capping accuracy at "
            f"{cert.ceiling:.4f}. Route only where the gap is real."
        )
    elif gap_test and not gap_test["significant"]:
        label = "CAUTION"
        reason = (
            f"The {gap:+.4f} gap to {runner_up} is not statistically significant "
            f"(exact McNemar p={gap_test['p_value']:.4f}, "
            f"{gap_test['best_only']} vs {gap_test['second_only']} discordant queries). "
            f"Model ordering on this benchmark is noise; do not route on it."
        )
    elif gains["oracle_gain"] <= _MIN_WORTHWHILE_GAIN:
        label = "CAUTION"
        reason = (
            f"Even a perfect router recovers only {gains['oracle_gain']:+.4f} "
            f"({rescuable}/{n} queries). A {gap:.3f} gap exists but the absolute "
            f"headroom does not justify the routing system."
        )
    else:
        label = "WORTH BUILDING"
        reason = (
            f"Ceiling is open (beta={cert.beta:.4f}) and a {gap:.4f} accuracy gap to "
            f"{runner_up} is statistically significant "
            f"(exact McNemar p={gap_test['p_value']:.4f}). "
            f"The ceiling is not the binding constraint."
        )

    return Verdict(
        label=label,
        reason=reason,
        beta=cert,
        best_accuracy=best_acc,
        second_accuracy=second_acc,
        best_model=best_name,
        accuracy_gap=gap,
        oracle_gain=gains["oracle_gain"],
        ceiling_gain=ceiling_gain,
        rescuable_errors=rescuable,
        best_wrong=best_wrong,
        n_queries=n,
        rho=pairwise_correlation(correctness),
        shock=common_shock_fit(correctness),
        gap_test=gap_test,
    )