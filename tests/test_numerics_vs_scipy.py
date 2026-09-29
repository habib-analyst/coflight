"""Check the pure-Python numerics against scipy.

Makes the claim in coflight.numerics a tested one rather than an assertion.
Skips if scipy is missing; it is a test-only dependency, not a runtime one.
"""

from __future__ import annotations

import numpy as np
import pytest

from coflight.estimate import clopper_pearson
from coflight.numerics import betainc, betaincinv

sp_betainc = pytest.importorskip("scipy.special").betainc
sp_beta = pytest.importorskip("scipy.stats").beta


def test_betainc_matches_scipy():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5000):
        a = float(rng.uniform(0.05, 200.0))
        b = float(rng.uniform(0.05, 200.0))
        x = float(rng.uniform(0.0, 1.0))
        worst = max(worst, abs(betainc(a, b, x) - float(sp_betainc(a, b, x))))
    assert worst < 1e-11, worst


def test_betainc_matches_scipy_on_extreme_shapes():
    cases = [
        (0.001, 0.001, 0.5),
        (0.01, 500.0, 0.5),
        (500.0, 0.01, 0.5),
        (1e-6, 1e-6, 0.5),
        (1000.0, 1.0, 0.5),
        (1.0, 1000.0, 0.5),
    ]
    for a, b, x in cases:
        assert betainc(a, b, x) == pytest.approx(
            float(sp_betainc(a, b, x)), abs=1e-9
        ), (a, b, x)


def test_clopper_pearson_matches_scipy_quantiles():
    for n in (10, 30, 100, 200, 530, 1000):
        for k in range(0, n + 1, max(1, n // 20)):
            low, high = clopper_pearson(k, n, 0.90)
            ref_low = 0.0 if k == 0 else float(sp_beta.ppf(0.05, k, n - k + 1))
            ref_high = 1.0 if k == n else float(sp_beta.ppf(0.95, k + 1, n - k))
            assert low == pytest.approx(ref_low, abs=1e-12), (k, n)
            assert high == pytest.approx(ref_high, abs=1e-12), (k, n)


def test_clopper_pearson_at_several_confidence_levels():
    for confidence in (0.80, 0.90, 0.95, 0.99):
        alpha = 1.0 - confidence
        for n in (50, 200, 530):
            for k in (0, 1, n // 2, n - 1, n):
                low, high = clopper_pearson(k, n, confidence)
                ref_low = (
                    0.0 if k == 0 else float(sp_beta.ppf(alpha / 2, k, n - k + 1))
                )
                ref_high = (
                    1.0
                    if k == n
                    else float(sp_beta.ppf(1 - alpha / 2, k + 1, n - k))
                )
                assert low == pytest.approx(ref_low, abs=1e-12), (k, n, confidence)
                assert high == pytest.approx(ref_high, abs=1e-12), (k, n, confidence)


def test_betaincinv_inverts_betainc():
    rng = np.random.default_rng(1)
    for _ in range(500):
        a = float(rng.uniform(0.1, 50.0))
        b = float(rng.uniform(0.1, 50.0))
        p = float(rng.uniform(0.01, 0.99))
        assert betainc(a, b, betaincinv(a, b, p)) == pytest.approx(p, abs=1e-6)


def test_betaincinv_matches_scipy_on_bimodal_shapes():
    for a, b, p in [(0.5, 50.0, 0.5), (50.0, 0.5, 0.5), (0.1, 20.0, 0.9)]:
        assert betaincinv(a, b, p) == pytest.approx(
            float(sp_beta.ppf(p, a, b)), abs=1e-6
        ), (a, b, p)
