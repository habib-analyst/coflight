"""Regression tests for defects found in the audit.

Each test here corresponds to a bug that produced a wrong or misleading
result without raising. They are kept together so the failure modes stay
readable; see FINDINGS.md section 4.
"""

from __future__ import annotations

import pytest

from conftest import build, from_error_sets
from coflight.decision import decide
from coflight.estimate import common_shock_fit, oracle_gain, pool_report, rank_models


def test_ranking_is_shared_so_best_model_cannot_diverge():
    """oracle_gain and decide must name the same best model.

    Regression: oracle_gain took max() over a dict, decide sorted by
    (accuracy, name) descending. On a tied pool they picked different models
    from the same matrix, so two answers to "which model is best" disagreed.
    Both now route through rank_models.
    """
    # a and b are exactly tied; c is worse. Whichever wins, both must agree.
    matrix = build({f"q:{i}": {"a": i % 2, "b": i % 2, "c": 0} for i in range(50)})
    pool = ["a", "b", "c"]
    corr = matrix.correctness(pool)
    verdict = decide(matrix, pool)
    gains = oracle_gain(corr)
    assert verdict.best_model == gains["best_model"]
    assert rank_models(corr)[0][1] == verdict.best_model


def test_tie_break_is_deterministic_across_calls():
    """Dict insertion order must not change the winner.

    Regression: max() over a dict depends on the order keys were inserted, so
    two callers with the same models in a different order got different
    "best" models from the same data.
    """
    forward = build({f"q:{i}": {"a": 1, "b": 1} for i in range(20)})
    reordered = build({f"q:{i}": {"b": 1, "a": 1} for i in range(20)})
    assert rank_models(forward.correctness(["a", "b"]))[0][1] == (
        rank_models(reordered.correctness(["b", "a"]))[0][1]
    )


def test_pool_report_raises_on_explicitly_empty_pool():
    """Regression: `if models` treated [] as "unspecified".

    An explicitly empty pool silently analysed every model in the matrix
    instead of raising, so a caller who filtered a pool down to nothing got a
    confident report for the whole roster.
    """
    matrix = build({f"q:{i}": {"a": 1, "b": 0} for i in range(10)})
    with pytest.raises(ValueError, match="no models available"):
        pool_report(matrix, [])


def test_shock_note_reports_the_estimate_not_the_clamped_value():
    """Regression: the diagnostic reported pi after it had been zeroed.

    The message read "moment estimate pi=0.000 exceeds 0.75", which is
    self-contradictory and hid the actual estimate. It now reports the value
    that overflowed the threshold.
    """
    # 8 models, 3 queries where all are wrong, 97 where all are right:
    # a large common-shock mass, so the moment estimate saturates pi.
    models = list("abcdefgh")
    matrix = build(
        {
            **{f"q:{i}": {m: 0 for m in models} for i in range(3)},
            **{f"q:{3 + i}": {m: 1 for m in models} for i in range(97)},
        }
    )
    shock = common_shock_fit(matrix.correctness(models))
    if shock["note"]:
        assert "0.000" not in shock["note"], shock["note"]


def test_common_shock_returns_no_surprising_types():
    """The public result must be floats plus the two documented extras."""
    matrix = from_error_sets(50, {"a": {1, 2}, "b": {1, 2}, "c": {3}})
    shock = common_shock_fit(matrix.correctness(["a", "b", "c"]))
    for key in (
        "pi",
        "alpha0",
        "alpha_min",
        "alpha_max",
        "beta_independent",
        "beta_fit",
        "beta_observed",
    ):
        assert isinstance(shock[key], float), key
    assert isinstance(shock["identifiable"], bool)
    assert isinstance(shock["note"], str)
