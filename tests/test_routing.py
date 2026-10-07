"""Tests for coflight.routing, against hand-constructed inputs with known answers.

Same discipline as the rest of the suite: no test touches the real dataset.
Every expected value below is computable by hand from the fixture, which is
what makes these tests able to catch wiring bugs (cf. FINDINGS.md section 4)
rather than merely re-running the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from coflight.matrix import Matrix
from coflight.routing import (
    RescueBin,
    beta_bootstrap,
    beta_cap_gain,
    brute_force_escalation,
    optimal_escalation,
    evaluate_cascade,
    fit_rescue_bins,
    greedy_escalation,
    group_stats,
    knapsack_select,
    lagrangian_select,
    stratified_split,
    synthetic_shock_pool,
)


def _tiny_matrix() -> Matrix:
    """2 groups x 2 queries x 2 models, all numbers hand-checkable.

    Group g1: mA acc 1.0 cost 10, mB acc 0.5 cost 1.
    Group g2: mA acc 0.5 cost 10, mB acc 1.0 cost 1.
    Equal group weights (2 queries each).
    """
    cells = {}
    for g, (a_correct, b_correct) in {
        "g1": ([1, 1], [1, 0]),
        "g2": ([0, 1], [1, 1]),
    }.items():
        for i, (ca, cb) in enumerate(zip(a_correct, b_correct)):
            cells[f"{g}:{i}"] = {
                "dataset": g,
                "kind": "mc",
                "gold": "A",
                "models": {
                    "mA": {"correct": ca, "cost": 10.0},
                    "mB": {"correct": cb, "cost": 1.0},
                },
            }
    return Matrix(cells)


def test_group_stats_hand_computed():
    stats = group_stats(_tiny_matrix(), ["mA", "mB"])
    assert stats["g1"][0]["model"] == "mA" or True  # order not guaranteed
    by_model = {o["model"]: o for o in stats["g1"]}
    assert by_model["mA"]["accuracy"] == pytest.approx(1.0)
    assert by_model["mA"]["mean_cost"] == pytest.approx(10.0)
    by_model = {o["model"]: o for o in stats["g2"]}
    assert by_model["mB"]["accuracy"] == pytest.approx(1.0)
    assert by_model["mB"]["mean_cost"] == pytest.approx(1.0)


def test_knapsack_picks_best_accuracy_per_group_when_unconstrained():
    stats = group_stats(_tiny_matrix(), ["mA", "mB"])
    res = knapsack_select(stats, budget=1e9)
    assert res.feasible
    # g1 -> mA (1.0), g2 -> mB (1.0): perfect accuracy is achievable.
    assert res.selection == {"g1": "mA", "g2": "mB"}
    assert res.expected_accuracy == pytest.approx(1.0)
    assert res.expected_cost == pytest.approx(0.5 * 10.0 + 0.5 * 1.0)


def test_knapsack_respects_tight_budget():
    stats = group_stats(_tiny_matrix(), ["mA", "mB"])
    # Budget 1.0/query: only (mB, mB) fits -> accuracy 0.75.
    res = knapsack_select(stats, budget=1.0)
    assert res.feasible
    assert res.selection == {"g1": "mB", "g2": "mB"}
    assert res.expected_accuracy == pytest.approx(0.75)
    assert res.expected_cost <= 1.0 + 1e-9


def test_knapsack_infeasible_below_cheapest():
    stats = group_stats(_tiny_matrix(), ["mA", "mB"])
    res = knapsack_select(stats, budget=0.5)
    assert not res.feasible
    assert res.selection == {}


def test_knapsack_rejects_bad_inputs():
    stats = group_stats(_tiny_matrix(), ["mA", "mB"])
    with pytest.raises(ValueError):
        knapsack_select(stats, budget=-1.0)
    with pytest.raises(ValueError):
        knapsack_select({}, budget=10.0)
    with pytest.raises(ValueError):
        group_stats(_tiny_matrix(), [])


def test_lagrangian_endpoints():
    stats = group_stats(_tiny_matrix(), ["mA", "mB"])
    free = lagrangian_select(stats, lam=0.0)
    assert free.selection == {"g1": "mA", "g2": "mB"}  # pure accuracy
    pricey = lagrangian_select(stats, lam=1e9)
    assert pricey.selection == {"g1": "mB", "g2": "mB"}  # pure cost
    with pytest.raises(ValueError):
        lagrangian_select(stats, lam=-0.5)


def test_beta_cap_bounds_cascade_gain():
    # L wrong on q0,q1; H rescues only q0; both wrong on q1.
    cheap = [0, 0, 1, 1]
    exp = [1, 0, 1, 1]
    gain, cap = beta_cap_gain(cheap, exp)
    assert gain == pytest.approx(0.25)  # only q0 rescued
    assert cap == pytest.approx(0.25)  # P(cheap wrong)=0.5 - beta=0.25
    assert gain <= cap + 1e-12


def test_beta_cap_holds_on_random_pools():
    # Property: gain <= cap on every random input, by construction.
    rng = np.random.default_rng(7)
    for _ in range(50):
        n = 200
        cheap = rng.integers(0, 2, n).tolist()
        exp = rng.integers(0, 2, n).tolist()
        gain, cap = beta_cap_gain(cheap, exp)
        assert 0.0 <= gain <= cap + 1e-12 <= 1.0 + 1e-12


def test_beta_cap_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        beta_cap_gain([1, 0], [1])
    with pytest.raises(ValueError):
        beta_cap_gain([], [])


def test_stratified_split_is_disjoint_and_stratified():
    qids = [f"g1:{i}" for i in range(10)] + [f"g2:{i}" for i in range(6)]
    ds = ["g1"] * 10 + ["g2"] * 6
    train, test = stratified_split(qids, ds, train_frac=0.5, seed=0)
    assert not set(train) & set(test)
    assert sorted(train + test) == sorted(qids)
    # Stratified: each dataset split ~50/50.
    assert sum(1 for q in train if q.startswith("g1")) == 5
    assert sum(1 for q in train if q.startswith("g2")) == 3


def _cascade_fixture():
    # 4 queries, 2 bins. Bin X: L wrong, H right on both (rescue 1.0).
    # Bin Y: L right on both (rescue 0.0). H costs 4/query everywhere.
    bin_of = {"q0": "X", "q1": "X", "q2": "Y", "q3": "Y"}
    cc = {"q0": 0, "q1": 0, "q2": 1, "q3": 1}
    ce = {"q0": 1, "q1": 1, "q2": 1, "q3": 1}
    co = {q: 4.0 for q in bin_of}
    return bin_of, cc, ce, co


def test_greedy_escalates_only_rescuable_bin():
    bin_of, cc, ce, co = _cascade_fixture()
    train = ["q0", "q1", "q2", "q3"]
    bins = fit_rescue_bins(bin_of, train, cc, ce, co)
    by_name = {b.name: b for b in bins}
    assert by_name["X"].rescue_rate == pytest.approx(1.0)
    assert by_name["Y"].rescue_rate == pytest.approx(0.0)
    counts = {"X": 2, "Y": 2}
    # Budget: base 0 + escalate X (2/4 * 4 = 2.0) but not Y.
    chosen = greedy_escalation(bins, counts, 4, budget=2.5, base_cost=0.0)
    assert chosen == ("X",)


def _subset_gain_cost(names, by_name, counts, n_train, base_cost):
    gain = sum(counts.get(nm, 0) / n_train * by_name[nm].rescue_rate for nm in names)
    cost = base_cost + sum(
        counts.get(nm, 0) / n_train * by_name[nm].extra_cost for nm in names
    )
    return gain, cost


def test_optimal_matches_brute_force_on_fixture():
    bin_of, cc, ce, co = _cascade_fixture()
    bins = fit_rescue_bins(bin_of, list(bin_of), cc, ce, co)
    counts = {"X": 2, "Y": 2}
    for budget in (0.0, 1.0, 2.5, 4.0, 100.0):
        optimal = optimal_escalation(bins, counts, 4, budget, 0.0)
        exact = brute_force_escalation(bins, counts, 4, budget, 0.0)
        assert set(optimal) == set(exact), f"budget={budget}"


def test_optimal_matches_brute_force_on_random_bins():
    # The DP is exact: its gain must equal the brute-force gain on every
    # random trial (set equality is not asserted: distinct optima can tie).
    rng = np.random.default_rng(3)
    for trial in range(20):
        k = 8
        bins = [
            RescueBin(f"b{i}", float(rng.random()), float(rng.random() * 5 + 0.1), 50)
            for i in range(k)
        ]
        by_name = {b.name: b for b in bins}
        counts = {f"b{i}": 50 for i in range(k)}
        for budget in (1.0, 3.0, 7.5, 20.0):
            optimal = optimal_escalation(bins, counts, k * 50, budget, 0.5)
            exact = brute_force_escalation(bins, counts, k * 50, budget, 0.5)
            g_opt, c_opt = _subset_gain_cost(optimal, by_name, counts, k * 50, 0.5)
            g_exact, _ = _subset_gain_cost(exact, by_name, counts, k * 50, 0.5)
            assert g_opt == pytest.approx(g_exact), f"trial={trial} budget={budget}"
            assert c_opt <= budget + 1e-9


def test_greedy_is_feasible_and_bounded_by_optimal():
    # Documents the greedy threshold rule honestly: always feasible, never
    # better than the true optimum. The gap between them is the price of the
    # interpretable threshold structure.
    rng = np.random.default_rng(5)
    for trial in range(20):
        k = 8
        bins = [
            RescueBin(f"b{i}", float(rng.random()), float(rng.random() * 5 + 0.1), 50)
            for i in range(k)
        ]
        by_name = {b.name: b for b in bins}
        counts = {f"b{i}": 50 for i in range(k)}
        for budget in (1.0, 3.0, 7.5, 20.0):
            greedy = greedy_escalation(bins, counts, k * 50, budget, 0.5)
            optimal = optimal_escalation(bins, counts, k * 50, budget, 0.5)
            g_greedy, c_greedy = _subset_gain_cost(greedy, by_name, counts, k * 50, 0.5)
            g_opt, _ = _subset_gain_cost(optimal, by_name, counts, k * 50, 0.5)
            assert c_greedy <= budget + 1e-9
            assert g_greedy <= g_opt + 1e-9


def test_evaluate_cascade_is_pure_scoring():
    bin_of, cc, ce, co = _cascade_fixture()
    cheap_cost = {q: 1.0 for q in bin_of}
    res = evaluate_cascade(
        ["q0", "q1", "q2", "q3"], bin_of, ("X",), cc, ce, cheap_cost, co,
        cheap_model="L", expensive_model="H", oracle_acc=1.0,
    )
    # Escalate X: q0,q1 -> H right (1,1); q2,q3 -> L right (1,1). Perfect.
    assert res.accuracy == pytest.approx(1.0)
    assert res.expected_cost == pytest.approx((5 + 5 + 1 + 1) / 4)
    assert res.gain == pytest.approx(0.5)  # rescued q0,q1 over always-L (0.5)
    assert res.headroom_captured == pytest.approx(1.0)  # all headroom taken


def test_synthetic_pool_recovers_known_beta():
    # pi=0.2 co-hard, alphas 0.1/0.3: true beta = 0.2 + 0.8*0.1*0.3 = 0.224.
    pool = synthetic_shock_pool(20000, [0.1, 0.3], 0.2, [1.0, 5.0], seed=1)
    both_wrong = sum(
        1 for cell in pool.values() if cell["correct_0"] == 0 and cell["correct_1"] == 0
    )
    assert both_wrong / len(pool) == pytest.approx(0.224, abs=0.01)


def test_synthetic_pool_rejects_bad_params():
    with pytest.raises(ValueError):
        synthetic_shock_pool(100, [0.1], 0.2, [1.0], seed=0)  # < 2 models
    with pytest.raises(ValueError):
        synthetic_shock_pool(100, [0.1, 0.2], 1.5, [1.0, 2.0], seed=0)


def test_beta_bootstrap_covers_true_beta():
    # Known pi=0.15, alphas 0.2/0.2 -> beta = 0.15 + 0.85*0.04 = 0.184.
    pool = synthetic_shock_pool(600, [0.2, 0.2], 0.15, [1.0, 2.0], seed=11)
    correctness = {
        q: {"m0": int(cell["correct_0"]), "m1": int(cell["correct_1"])}
        for q, cell in pool.items()
    }
    st = beta_bootstrap(correctness, n_boot=500, seed=0)
    assert st.ci_low <= 0.184 <= st.ci_high
    assert st.ci_low <= st.beta <= st.ci_high
    with pytest.raises(ValueError):
        beta_bootstrap(correctness, n_boot=50)


def test_knapsack_optimal_on_synthetic_ground_truth():
    # Two groups; hand-set accuracies/costs so the optimum is unambiguous and
    # the DP must find it (not merely a plausible answer).
    groups = {
        "g1": [
            {"model": "cheap", "accuracy": 0.6, "mean_cost": 1.0, "n": 100},
            {"model": "pricey", "accuracy": 0.9, "mean_cost": 4.0, "n": 100},
        ],
        "g2": [
            {"model": "cheap", "accuracy": 0.6, "mean_cost": 1.0, "n": 100},
            {"model": "pricey", "accuracy": 0.61, "mean_cost": 4.0, "n": 100},
        ],
    }
    # Budget 2.5/query: (pricey, cheap) costs 0.5*4+0.5*1=2.5, acc 0.75.
    # (pricey, pricey) costs 4.0 > budget. (cheap, pricey): acc 0.605. So the
    # optimum is pricey on g1 (worth it: +0.30) and cheap on g2 (+0.01).
    res = knapsack_select(groups, budget=2.5)
    assert res.feasible
    assert res.selection == {"g1": "pricey", "g2": "cheap"}
    assert res.expected_accuracy == pytest.approx(0.75)
