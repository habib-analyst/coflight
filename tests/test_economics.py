"""Cost-normalised economics."""

from __future__ import annotations

import pytest

from coflight.economics import (
    ModelEconomics,
    cheapest_meeting,
    delegation_savings,
    model_economics,
    pareto_front,
)
from coflight.matrix import Matrix


def costed(entries: dict[str, tuple[int, int, float]]) -> Matrix:
    """Build a matrix from (n_correct, n_queries, cost_per_query) per model."""
    n_queries = max(q for _, q, _ in entries.values())
    cells = {}
    for i in range(n_queries):
        cells[f"q:{i}"] = {
            "dataset": "test",
            "kind": "mc",
            "gold": None,
            "models": {
                name: {
                    "correct": int(i < n_correct),
                    "cost": cost,
                }
                for name, (n_correct, _, cost) in entries.items()
            },
        }
    return Matrix(cells)


def test_cost_per_correct_divides_price_by_accuracy():
    e = ModelEconomics(name="m", accuracy=0.5, mean_cost=0.01)
    assert e.cost_per_correct == pytest.approx(0.02)


def test_zero_accuracy_is_infinite_cost():
    e = ModelEconomics(name="m", accuracy=0.0, mean_cost=0.01)
    assert e.cost_per_correct == float("inf")


def test_economics_sorted_by_cost_per_correct():
    matrix = costed(
        {
            # 0.10 / 0.90 = 0.111 per correct
            "pricey_good": (90, 100, 0.10),
            # 0.001 / 0.10 = 0.010 per correct: cheaper per answer
            "cheap_bad": (10, 100, 0.001),
        }
    )
    econ = model_economics(matrix, ["pricey_good", "cheap_bad"])
    assert list(econ) == ["cheap_bad", "pricey_good"]


def test_pareto_front_keeps_only_non_dominated():
    """Ties on cost per correct are broken by accuracy.

    'good' and 'cheap_bad' both cost 0.010 per correct answer, so the more
    accurate one dominates outright. 'cheap_bad' survives only when it is
    genuinely cheaper per correct answer, which is what the second test
    covers.
    """
    econ = {
        "cheap_bad": ModelEconomics("cheap_bad", 0.10, 0.001),
        "good": ModelEconomics("good", 1.00, 0.010),
        "dominated": ModelEconomics("dominated", 0.90, 0.020),
    }
    front = set(pareto_front(econ))
    assert "good" in front
    assert "cheap_bad" not in front  # same cost per correct, far worse accuracy
    assert "dominated" not in front


def test_a_cheap_model_survives_when_it_is_cheaper_per_correct():
    econ = {
        # 0.001 / 0.70 = 0.00143 per correct
        "cheap_bad": ModelEconomics("cheap_bad", 0.70, 0.001),
        # 0.010 / 0.90 = 0.0111 per correct
        "good": ModelEconomics("good", 0.90, 0.010),
    }
    assert set(pareto_front(econ)) == {"cheap_bad", "good"}


def test_dominated_model_is_removed():
    econ = {
        "strong_cheap": ModelEconomics("strong_cheap", 0.80, 0.001),
        "weak_pricey": ModelEconomics("weak_pricey", 0.20, 0.010),
    }
    assert pareto_front(econ) == ["strong_cheap"]


def test_cheapest_meeting_picks_lowest_cost_among_eligible():
    econ = {
        "cheap": ModelEconomics("cheap", 0.70, 0.001),
        "pricy": ModelEconomics("pricy", 0.90, 0.010),
    }
    # both clear 0.65, but "cheap" wins on cost per correct
    name, cpc = cheapest_meeting(econ, 0.65)
    assert name == "cheap"
    assert cpc == pytest.approx(0.001 / 0.70)


def test_cheapest_meeting_reports_unreachable():
    econ = {"weak": ModelEconomics("weak", 0.10, 0.001)}
    name, cpc = cheapest_meeting(econ, 0.90)
    assert name is None
    assert cpc == float("inf")


def test_delegation_savings_is_linear_in_share():
    best = ModelEconomics("best", 0.90, 0.010)
    cheap = ModelEconomics("cheap", 0.70, 0.001)
    per_query_gap = best.mean_cost - cheap.mean_cost  # 0.009
    ten = delegation_savings(best, cheap, 0.10)
    half = delegation_savings(best, cheap, 0.50)
    assert ten == pytest.approx(per_query_gap * 0.10)
    assert half == pytest.approx(per_query_gap * 0.50)
    assert half == pytest.approx(ten * 5)


def test_delegation_never_negative_when_delegate_is_cheaper():
    best = ModelEconomics("best", 0.90, 0.010)
    cheap = ModelEconomics("cheap", 0.70, 0.001)
    assert delegation_savings(best, cheap, 1.0) > 0


def test_delegating_to_a_costlier_model_loses_money():
    best = ModelEconomics("best", 0.90, 0.001)
    dear = ModelEconomics("dear", 0.95, 0.010)
    assert delegation_savings(best, dear, 0.5) < 0


def test_zero_share_saves_nothing():
    best = ModelEconomics("best", 0.90, 0.010)
    cheap = ModelEconomics("cheap", 0.70, 0.001)
    assert delegation_savings(best, cheap, 0.0) == 0.0


def test_economics_reads_real_prices():
    matrix = costed({"a": (50, 100, 0.002), "b": (90, 100, 0.010)})
    econ = model_economics(matrix, ["a", "b"])
    assert econ["a"].accuracy == pytest.approx(0.5)
    assert econ["a"].mean_cost == pytest.approx(0.002)
    assert econ["b"].accuracy == pytest.approx(0.9)
