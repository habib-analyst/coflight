"""Beta estimation and the oracle gain decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import from_error_sets
from coflight.estimate import (
    clopper_pearson,
    co_failure_rate,
    common_shock_fit,
    oracle_gain,
    pairwise_correlation,
    pool_report,
)
from coflight.numerics import betainc, betaincinv


# ------------------------------------------------------------------ numerics


def test_betainc_is_symmetric_about_half():
    for a in (0.5, 1.0, 2.5, 10.0):
        assert betainc(a, a, 0.5) == pytest.approx(0.5, abs=1e-10)


def test_betainc_boundaries():
    assert betainc(2.0, 3.0, 0.0) == 0.0
    assert betainc(2.0, 3.0, 1.0) == 1.0


def test_betaincinv_round_trips():
    for a, b in ((2, 5), (3, 10), (7, 7)):
        for x in (0.1, 0.35, 0.5, 0.72, 0.9):
            assert betaincinv(a, b, betainc(a, b, x)) == pytest.approx(x, abs=1e-9)


# ------------------------------------------------------- clopper-pearson


def test_cp_with_zero_events_has_zero_lower_bound():
    low, high = clopper_pearson(0, 100, 0.90)
    assert low == 0.0
    # textbook: upper bound for 0/100 at 90% confidence
    assert high == pytest.approx(0.0295, abs=1e-3)


def test_cp_with_all_events_has_unit_upper_bound():
    low, high = clopper_pearson(20, 20, 0.90)
    assert high == 1.0
    # textbook two-sided 90% lower bound for 20/20
    assert low == pytest.approx(0.8609, abs=1e-3)


def test_cp_brackets_the_point_estimate():
    low, high = clopper_pearson(13, 500)
    assert low < 13 / 500 < high


def test_cp_is_conservative_wider_than_normal_approximation():
    # exact interval must not be narrower than the naive +-1.96*se
    low, high = clopper_pearson(2, 100)
    se = (0.02 * 0.98 / 100) ** 0.5
    assert (high - low) / 2 > 1.96 * se


def test_cp_rejects_impossible_counts():
    with pytest.raises(ValueError):
        clopper_pearson(5, 3)


# ------------------------------------------------------------- co-failure


def test_co_failure_counts_all_wrong_queries(independent_pool):
    correct = independent_pool.correctness(independent_pool.models_answering_all())
    assert co_failure_rate(correct) == (0, 100)


def test_co_failure_detects_shared_failure(co_failing_pool):
    correct = co_failing_pool.correctness(co_failing_pool.models_answering_all())
    assert co_failure_rate(correct) == (8, 100)


def test_beta_is_zero_when_all_errors_are_rescuable(independent_pool):
    """The bug this guards: tuple order in co_failure_rate.

    With no co-failure, beta must be 0, not 1.
    """
    correct = independent_pool.correctness(independent_pool.models_answering_all())
    gains = oracle_gain(correct)
    assert gains["beta"] == 0.0
    # best model errs on 5, all rescuable
    assert gains["oracle_gain"] == pytest.approx(0.05)


def test_common_shock_survives_large_pool_underflow():
    """prod(1 - alpha_i) underflows to 0.0 for big pools.

    The pre-fix naive product made beta_fit come out as exactly 1.0 on the
    52-model pool. Log-space arithmetic must keep it finite, and the
    independent baseline must saturate rather than go NaN.
    """
    models = [f"m{i}" for i in range(40)]
    errors = {m: set(range(10 + i, 40 + i)) for i, m in enumerate(models)}
    m = from_error_sets(200, errors)
    correct = m.correctness(models)
    shock = common_shock_fit(correct)
    assert 0.0 <= shock["beta_independent"] <= 1.0
    assert np.isfinite(shock["beta_fit"])
    assert shock["beta_fit"] <= 1.0


def test_independence_baseline_saturates_on_wide_pool():
    """Independence is a degenerate baseline on a large heterogeneous pool.

    P(all correct) = prod(1 - alpha_i) over 52 models whose error rates reach
    0.96 is ~5e-24, so beta_independent pins to 1.0. The real 52-model pool
    exhibits exactly this, which means the independence ratio carries no
    information at that pool size. Pinned here so the limitation is explicit
    rather than discovered later.
    """
    models = [f"m{i}" for i in range(52)]
    # One strong model at 0.85, the rest much weaker, as in the real pool.
    errors = {models[0]: set(range(30, 200))}
    for i, m in enumerate(models[1:], start=1):
        errors[m] = set(range(20 + i, 200))
    matrix = from_error_sets(200, errors)
    shock = common_shock_fit(matrix.correctness(models))
    assert shock["beta_independent"] == 1.0
    assert shock["beta_observed"] < shock["beta_independent"]


def test_oracle_gain_excludes_unresolvable_mass(co_failing_pool):
    correct = co_failing_pool.correctness(co_failing_pool.models_answering_all())
    gains = oracle_gain(correct)
    # best model errs on 5, all inside the shared set of 8
    assert gains["beta"] == pytest.approx(0.08)
    assert gains["oracle_gain"] == pytest.approx(0.0)


def test_oracle_gain_counts_mixed_rescuable_mass(mixed_pool):
    """Best model errs on 5; 3 shared, 2 private -> gain 0.02."""
    correct = mixed_pool.correctness(mixed_pool.models_answering_all())
    gains = oracle_gain(correct)
    assert gains["beta"] == pytest.approx(0.03)
    assert gains["oracle_gain"] == pytest.approx(0.02)


def test_oracle_gain_never_negative():
    # every query co-hard: nothing is rescuable
    cells = {f"q:{i}": {"a": 0, "b": 0, "c": 0} for i in range(50)}
    cells.update({f"q:{i}": {"a": 1, "b": 1, "c": 1} for i in range(50, 100)})
    from conftest import build

    m = build(cells)
    gains = oracle_gain(m.correctness(["a", "b", "c"]))
    assert gains["oracle_gain"] == pytest.approx(0.0)
    assert gains["beta"] == pytest.approx(0.5)


# ------------------------------------------------------- correlation / shock


def test_pairwise_correlation_is_positive_for_shared_errors(nested_pool):
    """Errors nested across models must show positive phi.

    Named for what it asserts: independence cannot produce nesting.
    """
    correct = nested_pool.correctness(nested_pool.models_answering_all())
    assert pairwise_correlation(correct) > 0


def test_pairwise_correlation_undefined_for_single_model():
    cells = {f"q:{i}": {"only": i % 2} for i in range(20)}
    from conftest import build

    m = build(cells)
    assert pairwise_correlation(m.correctness(["only"])) == 0.0


def test_common_shock_does_not_overpredict_beta(nested_pool):
    """The fitted co-failure rate must never exceed the observed one.

    Regression guard: an earlier version set pi from the *observed* co-failure
    count and added the product on top, double-counting those queries.
    """
    correct = nested_pool.correctness(nested_pool.models_answering_all())
    shock = common_shock_fit(correct)
    n_all_wrong, n = co_failure_rate(correct)
    observed = n_all_wrong / n
    assert shock["beta_observed"] == pytest.approx(observed, abs=1e-12)
    assert shock["beta_fit"] <= observed + 1e-9


def test_common_shock_flags_small_pools_as_unidentifiable(independent_pool):
    """A 3-model pool cannot separate pi from independent error.

    Asserts the estimator refuses to guess instead of returning a confident
    number derived from an ill-conditioned ratio.
    """
    correct = independent_pool.correctness(independent_pool.models_answering_all())
    shock = common_shock_fit(correct)
    assert shock["identifiable"] is False
    assert shock["note"]


def test_common_shock_reports_independent_counterfactual(independent_pool):
    correct = independent_pool.correctness(independent_pool.models_answering_all())
    shock = common_shock_fit(correct)
    # with no co-hardness, independent errors still co-fail sometimes
    assert 0.0 <= shock["beta_independent"] <= 1.0


# ------------------------------------------------------------------ report


def test_pool_report_certificate(independent_pool):
    report = pool_report(independent_pool, ["m_strong", "m_mid", "m_weak"])
    cert = report.certificate
    assert cert.pool_size == 3
    assert cert.n_queries == 100
    assert cert.beta == 0.0
    assert cert.ceiling == 1.0
    # m_weak errs on 5 of 100, so it is the best model
    assert report.best_model == "m_weak"
    assert report.best_accuracy == pytest.approx(0.95)


def test_pool_report_requires_shared_queries():
    cells = {
        "q:0": {"a": 1, "b": 0},
        "q:1": {"a": 1},  # b did not answer this one
    }
    from conftest import build

    m = build(cells)
    # q:1 is not fully answered, so it is dropped rather than raising;
    # the pool report must then be based only on q:0.
    report = pool_report(m, ["a", "b"])
    assert report.certificate.n_queries == 1
