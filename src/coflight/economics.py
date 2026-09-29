"""Cost-normalised routing economics.

Accuracy and price are separate questions, and an ensemble can be accurate
and unaffordable at the same time. A router that only calls the best model is
not free either: it pays the premium on every query to save itself on the
small fraction it delegates.

The comparison here is spend per correct answer, not accuracy at equal spend.
That reframes the decision: a 3x more expensive model that is 2x more
accurate is not 3x more expensive to operate, it is 1.5x more expensive per
correct answer, and it may still be the cheaper way to buy an accuracy target.
"""

from __future__ import annotations

from dataclasses import dataclass

from coflight.matrix import Matrix


@dataclass(frozen=True, slots=True)
class ModelEconomics:
    """Accuracy and unit price for one model over a fixed query set."""

    name: str
    accuracy: float
    mean_cost: float

    @property
    def cost_per_correct(self) -> float:
        """Dollars per correct answer.

        The number a budget holder actually needs. Accuracy and price
        separately are not comparable; spend divided by what it buys is.
        """
        if self.accuracy <= 0:
            return float("inf")
        return self.mean_cost / self.accuracy


def model_economics(matrix: Matrix, models: list[str]) -> dict[str, ModelEconomics]:
    """Accuracy and mean cost per model, keyed and sorted by cost per correct."""
    stats = matrix.per_model(models)
    out = {
        s.name: ModelEconomics(name=s.name, accuracy=s.accuracy, mean_cost=s.mean_cost)
        for s in stats.values()
    }
    return dict(sorted(out.items(), key=lambda kv: kv[1].cost_per_correct))


def pareto_front(economics: dict[str, ModelEconomics]) -> list[str]:
    """Models that are not beaten on both accuracy and cost per correct.

    A model is dominated when another is at least as accurate and no more
    expensive per correct answer, and strictly better on one of the two. Only
    these are worth a decision; the rest should not appear in a shortlist.
    """
    front = []
    for name, candidate in economics.items():
        dominated = any(
            other.cost_per_correct <= candidate.cost_per_correct
            and other.accuracy >= candidate.accuracy
            and (
                other.cost_per_correct < candidate.cost_per_correct
                or other.accuracy > candidate.accuracy
            )
            for other_name, other in economics.items()
            if other_name != name
        )
        if not dominated:
            front.append(name)
    return front


def cheapest_meeting(
    economics: dict[str, ModelEconomics], target_accuracy: float
) -> tuple[str | None, float]:
    """Cheapest-per-correct model that clears an accuracy target.

    Returns (None, inf) when nothing in the pool reaches the target. That is
    itself a result: the target is unreachable from this roster, so adding
    cheaper models cannot fix it.
    """
    eligible = {n: e for n, e in economics.items() if e.accuracy >= target_accuracy}
    if not eligible:
        return None, float("inf")
    name = min(eligible, key=lambda n: eligible[n].cost_per_correct)
    return name, eligible[name].cost_per_correct


def delegation_savings(
    best: ModelEconomics, delegate: ModelEconomics, delegable_share: float
) -> float:
    """Spend per query under best-only vs delegating a share of the traffic.

    `delegable_share` is the fraction of queries the router is willing to
    send to the cheaper model. Returns best_only_cost - routed_cost, so a
    negative value means delegation costs more than it saves.

    This is a plain budget calculation and deliberately makes no claim about
    whether the delegation is *accurate*: which queries are safe to delegate
    is exactly the routing problem the rest of this package leaves open.
    """
    if delegable_share <= 0:
        return 0.0
    share = min(1.0, delegable_share)
    best_only = best.mean_cost
    routed = (1.0 - share) * best.mean_cost + share * delegate.mean_cost
    return best_only - routed
