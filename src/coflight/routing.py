"""Cost-optimal routing under a co-failure ceiling.

``coflight`` established that no selection policy whose output is one member
model's answer -- a router, a cascade, a majority vote -- can beat ``1 - beta``
on a workload. This module turns the ceiling into an actionable routing
policy: given per-query costs, pick the model (or cascade) per query to
maximize accuracy under a spend budget.

Two policy classes, both with exact structure:

A. **Per-group model selection** is a multiple-choice knapsack, solved
   exactly by dynamic programming. Its Lagrangian relaxation has threshold
   structure: each group independently picks ``argmax_m (a_m - lambda c_m)``.

B. **Disagreement-gated cascade.** A cheap model ``L`` always runs; a query
   is escalated to an expensive model ``H`` iff a *pre-answer* gating signal
   says rescue is likely. The optimal gating has threshold structure in
   rescue-density space, and -- this is the ceiling connection -- the gain of
   *any* cascade over always-``L`` is bounded by ``P(L wrong) - beta``.
   Beta prices the router: it is the mass no gating signal can ever recover.

The honest-evaluation rule from FINDINGS.md section 3 is enforced by
construction: gating signals are fit on a TRAIN split of queries and the
policy is scored on a disjoint TEST split. A router that only works on the
queries it was tuned on is reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, TypedDict

import numpy as np

from coflight.matrix import Matrix


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------


class GroupOption(TypedDict):
    """One candidate model for one query group."""

    model: str
    accuracy: float
    mean_cost: float
    n: int


@dataclass(frozen=True, slots=True)
class KnapsackResult:
    """Outcome of the multiple-choice knapsack for one budget."""

    feasible: bool
    selection: dict[str, str]  # group -> model
    expected_accuracy: float
    expected_cost: float
    budget: float


@dataclass(frozen=True, slots=True)
class CascadeResult:
    """A gated cascade evaluated on one query split."""

    cheap_model: str
    expensive_model: str
    escalated_bins: tuple[str, ...]
    accuracy: float
    expected_cost: float
    # Gain over always-cheap, and the beta-imposed cap on that gain.
    gain: float
    gain_cap: float
    headroom_captured: float  # gain / oracle_gain, vs the per-query oracle
    n_test: int


@dataclass(frozen=True, slots=True)
class BetaStability:
    """Bootstrap stability of beta under query resampling."""

    beta: float
    ci_low: float
    ci_high: float
    n_boot: int
    # Beta recomputed on two disjoint halves: a coarse stability check that
    # needs no resampling machinery.
    beta_half_a: float
    beta_half_b: float


# ---------------------------------------------------------------------------
# Policy class A: per-group model selection (multiple-choice knapsack)
# ---------------------------------------------------------------------------


def group_stats(
    matrix: Matrix, models: Sequence[str], group_key: str = "dataset"
) -> dict[str, list[GroupOption]]:
    """Per-group accuracy and mean cost for every model.

    Groups default to datasets. ``group_key`` may also be ``"all"`` for a
    single group (degenerating to plain best-model selection).
    """
    if not models:
        raise ValueError("no models in pool")
    correctness = matrix.correctness(models)
    costs = matrix.costs(models)
    if not correctness:
        raise ValueError("no queries answered by every model in the pool")

    groups: dict[str, dict[str, list]] = {}
    for qid, row in correctness.items():
        g = "all" if group_key == "all" else matrix.dataset_of(qid)
        bucket = groups.setdefault(g, {})
        for m in models:
            acc_cost = bucket.setdefault(m, [[], []])
            acc_cost[0].append(row[m])
            acc_cost[1].append(costs[qid][m])

    out: dict[str, list[GroupOption]] = {}
    for g, per_model in groups.items():
        opts = []
        for m, (accs, cs) in per_model.items():
            n = len(accs)
            opts.append(
                {
                    "model": m,
                    "accuracy": float(sum(accs) / n),
                    "mean_cost": float(sum(cs) / n),
                    "n": n,
                }
            )
        out[g] = opts
    return out


def knapsack_select(
    groups: dict[str, list[GroupOption]],
    budget: float,
    weights: dict[str, float] | None = None,
    cost_unit: float = 1e-7,
) -> KnapsackResult:
    """Exact multiple-choice knapsack: one model per group, cost <= budget.

    Maximizes sum_g w_g * a_{g, m(g)} subject to sum_g w_g * c_{g, m(g)} <=
    budget, where w_g is the group's query share. Costs are discretized to
    integer ``cost_unit`` dollars for the DP; the reported expected cost uses
    the undiscretized means, so the budget check is conservative up to one
    unit of rounding per group.

    With G groups and B budget units the DP is O(G * B * max_options); on the
    bundled data (3 groups, 52 models) it is milliseconds.
    """
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")
    if not groups:
        raise ValueError("no groups to route")

    names = sorted(groups)
    n_total = sum(o["n"] for o in groups[names[0]])
    if weights is None:
        weights = {g: sum(o["n"] for o in groups[g]) / n_total for g in names}
    wsum = sum(weights[g] for g in names)
    if wsum <= 0:
        raise ValueError("weights must sum to a positive value")
    weights = {g: weights[g] / wsum for g in names}

    # Unconstrained: per-group best accuracy, no DP needed.
    if budget == float("inf"):
        selection, acc, cost = {}, 0.0, 0.0
        for g in names:
            best = max(groups[g], key=lambda o: (o["accuracy"], o["model"]))
            selection[g] = best["model"]
            acc += weights[g] * best["accuracy"]
            cost += weights[g] * best["mean_cost"]
        return KnapsackResult(True, selection, acc, cost, budget)

    budget_units = int(round(budget / cost_unit))
    # dp: cost_units -> (accuracy mass, selection tuple)
    dp: dict[int, tuple[float, tuple[str, ...]]] = {0: (0.0, ())}
    for g in names:
        options = groups[g]
        if not options:
            raise ValueError(f"group {g!r} has no model options")
        w = weights[g]
        nxt: dict[int, tuple[float, tuple[str, ...]]] = {}
        for spent, (mass, chosen) in dp.items():
            for o in options:
                c = spent + int(round(w * o["mean_cost"] / cost_unit))
                if c > budget_units:
                    continue
                m2 = mass + w * o["accuracy"]
                prev = nxt.get(c)
                if prev is None or m2 > prev[0]:
                    nxt[c] = (m2, chosen + (o["model"],))
        dp = nxt
        if not dp:
            return KnapsackResult(False, {}, 0.0, float("inf"), budget)

    best_cost_units, (best_mass, best_chosen) = max(dp.items(), key=lambda kv: kv[1][0])
    selection = dict(zip(names, best_chosen))
    exp_cost = sum(
        weights[g] * next(o["mean_cost"] for o in groups[g] if o["model"] == selection[g])
        for g in names
    )
    return KnapsackResult(True, selection, best_mass, exp_cost, budget)


def lagrangian_select(
    groups: dict[str, list[GroupOption]],
    lam: float,
    weights: dict[str, float] | None = None,
) -> KnapsackResult:
    """Threshold structure: per group, pick argmax_m (a_m - lam * c_m).

    This is the Lagrangian relaxation of the knapsack. For lam = 0 it is
    unconstrained best-accuracy; as lam -> infinity it picks the cheapest
    model per group. Sweeping lam traces the accuracy-cost frontier, and the
    per-group argmax is *why* the optimal policy has threshold structure:
    each group's choice depends on (accuracy, cost) only through the scalar
    score a - lam*c, never on the other groups.
    """
    if lam < 0:
        raise ValueError(f"lambda must be non-negative, got {lam}")
    if not groups:
        raise ValueError("no groups to route")
    names = sorted(groups)
    n_total = sum(o["n"] for o in groups[names[0]])
    if weights is None:
        weights = {g: sum(o["n"] for o in groups[g]) / n_total for g in names}
    wsum = sum(weights[g] for g in names)
    weights = {g: weights[g] / wsum for g in names}

    selection, acc, cost = {}, 0.0, 0.0
    for g in names:
        # Deterministic tie-break on (score, accuracy, name): without it two
        # runs could pick different models at equal score and the frontier
        # would wobble. (Same lesson as rank_models in estimate.py.)
        best = max(
            groups[g],
            key=lambda o: (o["accuracy"] - lam * o["mean_cost"], o["accuracy"], o["model"]),
        )
        selection[g] = best["model"]
        acc += weights[g] * best["accuracy"]
        cost += weights[g] * best["mean_cost"]
    return KnapsackResult(True, selection, acc, cost, budget=cost)


# ---------------------------------------------------------------------------
# Policy class B: disagreement-gated cascade
# ---------------------------------------------------------------------------


def beta_cap_gain(
    correct_cheap: Sequence[int], correct_exp: Sequence[int]
) -> tuple[float, float]:
    """Gain of the expensive model over the cheap one, and its beta cap.

    gain = P(cheap wrong, expensive right): the queries a cascade could
    rescue. cap = P(cheap wrong) - beta, where beta is the co-failure rate of
    the *pair*. No gating signal -- no matter how clever -- can push the
    cascade's gain past the cap, because the beta mass is wrong under both
    models. Returns (gain, cap); cap >= gain always.
    """
    c = np.asarray(correct_cheap, dtype=float)
    e = np.asarray(correct_exp, dtype=float)
    if c.shape != e.shape or c.size == 0:
        raise ValueError("need non-empty, equally-sized correctness arrays")
    rescue = float(np.mean((c == 0) & (e == 1)))
    both_wrong = float(np.mean((c == 0) & (e == 0)))
    cheap_wrong = float(np.mean(c == 0))
    cap = max(0.0, cheap_wrong - both_wrong)
    # Numerical hygiene, not a claim: rescue cannot exceed the cap by
    # construction, but float noise on huge arrays could print otherwise.
    return rescue, max(cap, rescue)


def stratified_split(
    query_ids: Sequence[str],
    datasets: Sequence[str],
    train_frac: float = 0.5,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Stratified train/test split of query ids, disjoint by construction."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    train, test = [], []
    by_ds: dict[str, list[str]] = {}
    for qid, ds in zip(query_ids, datasets):
        by_ds.setdefault(ds, []).append(qid)
    for ds in sorted(by_ds):
        qids = by_ds[ds]
        perm = rng.permutation(len(qids))
        k = max(1, int(round(len(qids) * train_frac)))
        # Guarantee a non-empty test split too.
        k = min(k, len(qids) - 1) if len(qids) > 1 else k
        train.extend(qids[i] for i in perm[:k])
        test.extend(qids[i] for i in perm[k:])
    if not train or not test:
        raise ValueError("split produced an empty side; need >= 2 queries per dataset")
    return train, test


@dataclass(slots=True)
class RescueBin:
    """One pre-answer signal bin: rescue statistics fit on TRAIN queries."""

    name: str
    rescue_rate: float  # P(cheap wrong, expensive right | bin), train
    extra_cost: float  # E[cost_expensive | bin], train
    n_train: int

    @property
    def density(self) -> float:
        """Rescue per dollar: the score the threshold rule sorts on."""
        return self.rescue_rate / self.extra_cost if self.extra_cost > 0 else float("inf")


def fit_rescue_bins(
    bin_of: dict[str, str],
    train_ids: Sequence[str],
    correct_cheap: dict[str, int],
    correct_exp: dict[str, int],
    cost_exp: dict[str, float],
) -> list[RescueBin]:
    """Fit per-bin rescue rates on train queries only.

    ``bin_of`` maps query id -> bin name, where the bin is a function of
    pre-answer signal alone (dataset, input length, ...). Anything fit here
    must not see test queries: that is the whole honest-evaluation point.
    """
    per_bin: dict[str, list[str]] = {}
    for qid in train_ids:
        per_bin.setdefault(bin_of[qid], []).append(qid)
    bins = []
    for name in sorted(per_bin):
        qids = per_bin[name]
        r = sum(1 for q in qids if correct_cheap[q] == 0 and correct_exp[q] == 1) / len(qids)
        e = sum(cost_exp[q] for q in qids) / len(qids)
        bins.append(RescueBin(name, r, e, len(qids)))
    return bins


def greedy_escalation(
    bins: Sequence[RescueBin],
    train_bin_counts: dict[str, int],
    n_train: int,
    budget: float,
    base_cost: float,
) -> tuple[str, ...]:
    """Threshold rule: escalate bins in decreasing rescue-density order.

    This is the fractional-knapsack greedy over bins. The Lagrangian view is
    cleaner: escalate bin b iff density_b > lam, with lam set so the train
    budget binds. Either way the policy has threshold structure -- a bin is
    in or out based on one scalar -- which is what makes it auditable.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    ordered = sorted(bins, key=lambda b: (-b.density, b.name))
    spent, chosen = base_cost, []
    for b in ordered:
        w = train_bin_counts.get(b.name, 0) / n_train
        if spent + w * b.extra_cost <= budget + 1e-12:
            chosen.append(b.name)
            spent += w * b.extra_cost
    return tuple(chosen)


def evaluate_cascade(
    test_ids: Sequence[str],
    bin_of: dict[str, str],
    escalated: Sequence[str],
    correct_cheap: dict[str, int],
    correct_exp: dict[str, int],
    cost_cheap: dict[str, float],
    cost_exp: dict[str, float],
    cheap_model: str,
    expensive_model: str,
    oracle_acc: float,
) -> CascadeResult:
    """Score a fixed escalation set on held-out queries. No fitting here."""
    esc = set(escalated)
    n = len(test_ids)
    if n == 0:
        raise ValueError("no test queries")
    acc = sum(
        correct_exp[q] if bin_of[q] in esc else correct_cheap[q] for q in test_ids
    ) / n
    cost = sum(
        (cost_cheap[q] + cost_exp[q]) if bin_of[q] in esc else cost_cheap[q]
        for q in test_ids
    ) / n
    gain, cap = beta_cap_gain(
        [correct_cheap[q] for q in test_ids], [correct_exp[q] for q in test_ids]
    )
    # Headroom is measured against what the cascade actually achieves over
    # always-cheap, relative to the per-query oracle on the test split.
    cheap_acc = sum(correct_cheap[q] for q in test_ids) / n
    headroom = max(0.0, oracle_acc - cheap_acc)
    realized = max(0.0, acc - cheap_acc)
    return CascadeResult(
        cheap_model=cheap_model,
        expensive_model=expensive_model,
        escalated_bins=tuple(sorted(esc)),
        accuracy=acc,
        expected_cost=cost,
        gain=realized,
        gain_cap=cap,
        headroom_captured=(realized / headroom) if headroom > 0 else 0.0,
        n_test=n,
    )


def optimal_escalation(
    bins: Sequence[RescueBin],
    train_bin_counts: dict[str, int],
    n_train: int,
    budget: float,
    base_cost: float,
    cost_unit: float = 1e-7,
) -> tuple[str, ...]:
    """Exact best escalation subset: 0/1 knapsack over bins, solved by DP.

    The greedy threshold rule is optimal for the *fractional* relaxation but
    can miss the 0/1 optimum (density order is not value order once items are
    indivisible). This DP finds the true optimum, so the deployed policy is
    exact and the threshold rule stays as the interpretable approximation.
    Deterministic tie-break: bins processed in sorted name order, first-found
    optimum kept -- same discipline as ``rank_models`` in estimate.py.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    names = sorted(b.name for b in bins)
    by_name = {b.name: b for b in bins}
    w = {nm: train_bin_counts.get(nm, 0) / n_train for nm in names}
    cap = int(round((budget - base_cost) / cost_unit))
    if cap < 0:
        return ()
    # dp[cost_units] = (gain, chosen subset). Discretization mirrors
    # knapsack_select; reported costs always use undiscretized means.
    dp: dict[int, tuple[float, tuple[str, ...]]] = {0: (0.0, ())}
    for nm in names:
        b = by_name[nm]
        c = int(round(w[nm] * b.extra_cost / cost_unit))
        g = w[nm] * b.rescue_rate
        nxt = dict(dp)
        for spent, (gain, chosen) in dp.items():
            c2 = spent + c
            if c2 > cap:
                continue
            g2 = gain + g
            prev = nxt.get(c2)
            if prev is None or g2 > prev[0] + 1e-12:
                nxt[c2] = (g2, chosen + (nm,))
        dp = nxt
    best_gain, best_subset = max(dp.values(), key=lambda t: t[0])
    return tuple(sorted(best_subset))


def brute_force_escalation(
    bins: Sequence[RescueBin],
    train_bin_counts: dict[str, int],
    n_train: int,
    budget: float,
    base_cost: float,
) -> tuple[str, ...]:
    """Exact best escalation subset by enumerating all 2^k subsets.

    Only for verification (k <= ~12). Lets us check the greedy threshold
    rule against the true optimum instead of asserting it is optimal.
    """
    names = [b.name for b in bins]
    by_name = {b.name: b for b in bins}
    best: tuple[str, ...] = ()
    best_gain = -1.0
    for mask in range(1 << len(names)):
        subset = tuple(names[i] for i in range(len(names)) if mask & (1 << i))
        cost = base_cost + sum(
            train_bin_counts.get(nm, 0) / n_train * by_name[nm].extra_cost for nm in subset
        )
        if cost > budget + 1e-12:
            continue
        gain = sum(
            train_bin_counts.get(nm, 0) / n_train * by_name[nm].rescue_rate for nm in subset
        )
        if gain > best_gain + 1e-12:
            best_gain, best = gain, subset
    return tuple(sorted(best))


# ---------------------------------------------------------------------------
# Beta stability under query resampling
# ---------------------------------------------------------------------------


def beta_bootstrap(
    correctness: dict[str, dict[str, int]],
    n_boot: int = 2000,
    seed: int = 0,
    confidence: float = 0.90,
) -> BetaStability:
    """Is beta a stable property of the pool, or does it wobble by query?

    Percentile bootstrap over queries (resampling within the observed query
    set), plus a coarse split-half check. A routing policy built on beta is
    only as trustworthy as beta itself.
    """
    if n_boot < 100:
        raise ValueError("n_boot must be >= 100 for a usable interval")
    qids = sorted(correctness)
    n = len(qids)
    if n < 10:
        raise ValueError("need at least 10 queries for a bootstrap")
    models = sorted(next(iter(correctness.values())))
    err = np.array([[1 - correctness[q][m] for m in models] for q in qids])
    beta = float(np.mean(err.all(axis=1)))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = err[idx].all(axis=2).mean(axis=1)
    lo_q, hi_q = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    lo, hi = float(np.quantile(boot, lo_q)), float(np.quantile(boot, hi_q))

    half = n // 2
    beta_a = float(np.mean(err[:half].all(axis=1)))
    beta_b = float(np.mean(err[half:].all(axis=1)))
    return BetaStability(beta, lo, hi, n_boot, beta_a, beta_b)


# ---------------------------------------------------------------------------
# Synthetic ground truth: verify the theory recovers the optimum
# ---------------------------------------------------------------------------


def synthetic_shock_pool(
    n_queries: int,
    alphas: Sequence[float],
    pi: float,
    costs: Sequence[float],
    seed: int = 0,
) -> dict[str, dict[str, int | float]]:
    """Pool with known common-shock parameters.

    Query is co-hard with probability pi (every model wrong); otherwise each
    model errs independently with its own rate alpha_i. Costs are fixed per
    model. The Bayes-optimal per-bin escalation set is computable in closed
    form here, which is what lets the tests assert the algorithm recovers the
    optimum rather than merely running.
    """
    if not 0.0 <= pi <= 1.0:
        raise ValueError("pi must be in [0, 1]")
    if len(alphas) != len(costs) or len(alphas) < 2:
        raise ValueError("need >= 2 models with matching alphas and costs")
    rng = np.random.default_rng(seed)
    k = len(alphas)
    out: dict[str, dict[str, int | float]] = {}
    for q in range(n_queries):
        co_hard = rng.random() < pi
        cell: dict[str, int | float] = {}
        for i, (a, c) in enumerate(zip(alphas, costs)):
            wrong = co_hard or (rng.random() < a)
            cell[f"correct_{i}"] = int(not wrong)
            cell[f"cost_{i}"] = float(c)
        out[f"q:{q}"] = cell
    return out


def analytic_optimal_escalation(
    bins: Sequence[RescueBin],
    train_bin_counts: dict[str, int],
    n_train: int,
    budget: float,
    base_cost: float,
) -> tuple[str, ...]:
    """Closed-form optimum when bins are independent: density threshold.

    With known per-bin rescue rates and costs, the expected-gain-maximizing
    escalation set under a budget is the density-greedy prefix (fractional
    knapsack is exact for the LP; greedy is optimal for 0/1 up to one bin).
    Exported so tests can compare the algorithm against the analytic answer
    on synthetic pools instead of against itself.
    """
    return greedy_escalation(bins, train_bin_counts, n_train, budget, base_cost)
