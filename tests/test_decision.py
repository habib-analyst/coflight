"""The build/don't-build decision logic."""

from __future__ import annotations

import pytest

from conftest import from_error_sets
from coflight.decision import decide


def test_ceiling_binding_gives_do_not_build():
    """Every best-model error is shared, so nothing is rescuable."""
    errors = {name: set(range(20)) for name in ("m_strong", "m_mid", "m_weak")}
    matrix = from_error_sets(100, errors)
    verdict = decide(matrix, sorted(errors))
    assert verdict.label == "DO NOT BUILD"
    assert verdict.binding_constraint == "co-failure ceiling (unresolvable)"


def test_near_ceiling_gains_are_not_worth_building():
    """Guard: a 0.98 baseline is not a routing opportunity.

    A real gap exists (0.02) and the ceiling is open, but the total headroom
    is 2% of queries. Reporting "WORTH BUILDING" here would push an operator
    to build a system for noise.
    """
    # best model errs on 2 of 100, both rescuable; runner-up is 0.02 behind.
    errors = {
        "m_best": {10, 11},
        "m_second": {10, 11, 40, 41},
        "m_weak": {40, 41, 42, 43},
    }
    matrix = from_error_sets(100, errors)
    verdict = decide(matrix, sorted(errors))
    assert verdict.oracle_gain == pytest.approx(0.02)
    assert verdict.accuracy_gap == pytest.approx(0.02)
    assert verdict.label == "CAUTION"


def test_real_headroom_is_worth_building():
    """A wide, clearly significant gap with real rescuable mass.

    The runner-up repeats the best model's 5 errors, so those stay shared
    (unrescuable), and adds 25 more, so the runner-up is decisively behind:
    25 discordant wins to 5 gives p=0.0003.
    """
    errors = {
        "m_best": set(range(5)),
        "m_weak": set(range(5, 30)),
        "m_second": set(range(5, 30)) | set(range(5)),
    }
    matrix = from_error_sets(100, errors)
    verdict = decide(matrix, sorted(errors))
    assert verdict.gap_test["significant"] is True
    assert verdict.label == "WORTH BUILDING"
    assert verdict.binding_constraint == "accuracy gap (routing headroom)"


def test_insignificant_gap_blocks_the_build():
    """The guard that matters: a visible gap that is pure noise.

    The best model leads by 0.02 on accuracy, but it wins only 2 discordant
    queries and loses 2, so exact McNemar gives p=1.0. Recommending a routing
    system here would be recommending a coin flip.
    """
    errors = {
        "m_best": {1, 2},
        "m_second": {1, 2, 3, 4},
        "m_weak": {3, 4, 5, 6},
    }
    matrix = from_error_sets(100, errors)
    verdict = decide(matrix, sorted(errors))
    assert verdict.accuracy_gap == pytest.approx(0.02)
    assert verdict.gap_test["significant"] is False
    assert verdict.label == "CAUTION"
    assert "not statistically significant" in verdict.reason


def test_high_beta_warns_even_with_headroom():
    """beta above 2% caps accuracy regardless of headroom.

    Branch precedence: the real ceiling outranks an insignificant gap, so
    this reports the ceiling rather than the noise.
    """
    errors = {
        "m_best": set(range(8)) | set(range(8, 30)),
        "m_second": set(range(8)) | set(range(30, 45)),
        "m_weak": set(range(8)) | set(range(45, 52)),
    }
    matrix = from_error_sets(200, errors)
    verdict = decide(matrix, sorted(errors))
    assert verdict.beta.beta == 0.04
    assert verdict.oracle_gain > 0.01
    assert verdict.label == "CAUTION"
    assert "capping accuracy" in verdict.reason


def test_empty_pool_raises():
    matrix = from_error_sets(10, {"a": set()})
    with pytest.raises(ValueError):
        decide(matrix, [])


def test_no_shared_queries_raises():
    """Restricting to a query set no model answered leaves nothing to compare."""
    matrix = from_error_sets(10, {"a": set(), "b": set()})
    unanswered = matrix.restrict_queries(["q:does-not-exist"])
    with pytest.raises(ValueError, match="no queries"):
        decide(unanswered, ["a", "b"])


def test_render_includes_key_numbers():
    errors = {"m_best": set(range(10)), "m_second": set(range(20))}
    matrix = from_error_sets(100, errors)
    text = decide(matrix, sorted(errors)).render()
    assert "beta (co-failure)" in text
    assert "VERDICT:" in text
    assert "binding constraint" in text
