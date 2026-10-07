"""Project #2 experiments: cost-optimal routing on the 67-model coflight data.

Reads data/matrix_marketE2.json (52 models x 530 queries, 3 datasets), runs:

  1. Per-dataset multiple-choice knapsack routing (exact DP) swept over
     budgets -> accuracy-cost Pareto frontier, vs baselines.
  2. Disagreement-gated cascade: cheap model always runs, escalate to the
     expensive model on pre-answer signal bins (dataset x input-length).
     Bins fit on TRAIN queries, policy scored on disjoint TEST queries.
  3. Beta bootstrap stability per dataset.

Outputs: results/*.csv and results/*.png. No API calls, no GPU. All numbers
are measured from the bundled matrix.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coflight.estimate import co_failure_rate, oracle_gain, rank_models
from coflight.matrix import load_matrix
from coflight.routing import (
    beta_bootstrap,
    evaluate_cascade,
    fit_rescue_bins,
    greedy_escalation,
    group_stats,
    knapsack_select,
    optimal_escalation,
    stratified_split,
)

HERE = Path(__file__).resolve().parent
HERE.mkdir(parents=True, exist_ok=True)
REPO = HERE.parent
CHEAP = "inclusionai/ling-2.6-flash"  # cheapest $/correct in FINDINGS.md
EXPENSIVE = "anthropic/claude-sonnet-4.6"  # most accurate in FINDINGS.md


def main() -> None:
    matrix = load_matrix(REPO / "data" / "matrix_marketE2.json")
    pool = matrix.models_answering_all()
    assert len(pool) == 52, f"expected 52-model pool, got {len(pool)}"
    correctness = matrix.correctness(pool)
    costs = matrix.costs(pool)
    qids = sorted(correctness)
    n = len(qids)
    print(f"pool={len(pool)} queries={n}")

    # ------------------------------------------------------------------
    # Experiment 1: knapsack routing Pareto frontier
    # ------------------------------------------------------------------
    groups = group_stats(matrix, pool)
    stats = matrix.per_model(pool)
    best_single = max(pool, key=lambda m: stats[m].accuracy)
    best_acc, best_cost = stats[best_single].accuracy, stats[best_single].mean_cost
    print(f"best single: {best_single} acc={best_acc:.4f} ${best_cost:.6f}/q")

    # Oracle per-query accuracy = 1 - beta over the 52-model pool.
    n_aw, _ = co_failure_rate(correctness)
    beta = n_aw / n
    oracle_acc = 1.0 - beta
    headroom = oracle_acc - best_acc
    print(f"beta={beta:.4f} oracle={oracle_acc:.4f} headroom={headroom:.4f}")

    lo = min(o["mean_cost"] for g in groups.values() for o in g)
    hi = best_cost
    budgets = sorted(set([*[lo * (1.5**i) for i in range(24)], hi]))
    budgets = [b for b in budgets if b <= hi * 1.001]

    rows = []
    for b in budgets:
        r = knapsack_select(groups, b)
        if not r.feasible:
            continue
        captured = (r.expected_accuracy - best_acc) / headroom if headroom > 0 else 0.0
        rows.append(
            {
                "budget": b,
                "accuracy": r.expected_accuracy,
                "cost": r.expected_cost,
                "headroom_captured": captured,
                "selection": ";".join(f"{g}={m}" for g, m in sorted(r.selection.items())),
            }
        )
    # Dedup to the frontier (strictly increasing accuracy).
    frontier, best_a = [], -1.0
    for r in sorted(rows, key=lambda r: r["cost"]):
        if r["accuracy"] > best_a + 1e-9:
            frontier.append(r)
            best_a = r["accuracy"]
    # The unconstrained optimum (per-group best accuracy) caps the frontier.
    unc = knapsack_select(groups, float("inf"))
    frontier.append(
        {
            "budget": float("inf"),
            "accuracy": unc.expected_accuracy,
            "cost": unc.expected_cost,
            "headroom_captured": (unc.expected_accuracy - best_acc) / headroom
            if headroom > 0
            else 0.0,
            "selection": ";".join(f"{g}={m}" for g, m in sorted(unc.selection.items())),
        }
    )
    print(
        f"[knapsack] unconstrained per-dataset best: acc={unc.expected_accuracy:.4f} "
        f"cost=${unc.expected_cost:.6f}/q "
        f"headroom={(unc.expected_accuracy - best_acc) / headroom:.3f}"
    )

    with open(HERE / "knapsack_frontier.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frontier[0]))
        w.writeheader()
        w.writerows(frontier)

    # Baselines.
    cheap_model = min(pool, key=lambda m: stats[m].mean_cost / max(stats[m].accuracy, 1e-9))
    print(f"cheapest-per-correct: {cheap_model}")
    base_rows = [
        ("always-best-single", best_acc, best_cost),
        ("cheapest-per-correct", stats[cheap_model].accuracy, stats[cheap_model].mean_cost),
        ("oracle-per-query", oracle_acc, float("nan")),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [r["cost"] * 1000 for r in frontier]
    ys = [r["accuracy"] for r in frontier]
    ax.plot(xs, ys, "o-", ms=4, label="knapsack routing (exact DP)")
    for name, acc, cost in base_rows:
        if np.isnan(cost):
            ax.axhline(acc, ls="--", c="gray", label=f"{name} ({acc:.3f})")
        else:
            ax.plot([cost * 1000], [acc], "s", ms=7, label=f"{name}")
    ax.set_xlabel("expected $ per 1000 queries")
    ax.set_ylabel("accuracy")
    ax.set_title("Routing Pareto frontier: 52-model pool, per-dataset knapsack")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "knapsack_pareto.png", dpi=120)

    # Headroom capture at 60% of always-best cost (agenda success criterion b).
    target_cost = 0.6 * best_cost
    feas = [r for r in frontier if r["cost"] <= target_cost + 1e-12]
    if feas:
        top = max(feas, key=lambda r: r["accuracy"])
        print(
            f"[knapsack] at <=60% of best-single cost (${target_cost:.6f}/q): "
            f"acc={top['accuracy']:.4f} headroom_captured={top['headroom_captured']:.3f}"
        )
    else:
        print("[knapsack] no frontier point within 60% of best-single cost")

    # ------------------------------------------------------------------
    # Experiment 2: disagreement-gated cascade, honest train/test split
    # ------------------------------------------------------------------
    # Pre-answer signal: dataset x input-length tercile (within dataset).
    tok_in = {}
    for q in qids:
        vals = [
            matrix._cells[q]["models"][m].get("tok_in", 0) for m in pool
        ]
        tok_in[q] = float(np.mean(vals))
    ds_of = {q: matrix.dataset_of(q) for q in qids}
    terc = {}
    for d in sorted(set(ds_of.values())):
        qs = [q for q in qids if ds_of[q] == d]
        ts = np.array([tok_in[q] for q in qs])
        lo_t, hi_t = np.quantile(ts, [1 / 3, 2 / 3])
        for q in qs:
            t = tok_in[q]
            b = "short" if t <= lo_t else ("long" if t > hi_t else "mid")
            terc[q] = f"{d}:{b}"
    bin_of = terc

    cc = {q: correctness[q][CHEAP] for q in qids}
    ce = {q: correctness[q][EXPENSIVE] for q in qids}
    co_c = {q: costs[q][CHEAP] for q in qids}
    co_e = {q: costs[q][EXPENSIVE] for q in qids}

    train_ids, test_ids = stratified_split(qids, [ds_of[q] for q in qids], 0.5, seed=0)
    train_set, test_set = set(train_ids), set(test_ids)
    bins = fit_rescue_bins(bin_of, train_ids, cc, ce, co_e)
    counts = {b.name: b.n_train for b in bins}
    n_train = len(train_ids)
    base_cost = float(np.mean([co_c[q] for q in train_ids]))
    oracle_test = 1.0 - sum(
        1 for q in test_ids if all(correctness[q][m] == 0 for m in pool)
    ) / len(test_ids)

    print(f"[cascade] cheap={CHEAP} expensive={EXPENSIVE}")
    print(f"[cascade] train={len(train_ids)} test={len(test_ids)} base_cost=${base_cost:.6f}/q")
    for b in sorted(bins, key=lambda b: -b.density):
        print(
            f"  bin {b.name:18s} rescue={b.rescue_rate:.3f} "
            f"extra_cost=${b.extra_cost:.6f} density={b.density:.1f}/$ n={b.n_train}"
        )

    budgets_c = [base_cost * mult for mult in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0)]
    casc_rows = []
    for b in budgets_c:
        esc_opt = optimal_escalation(bins, counts, n_train, b, base_cost)
        esc_greedy = greedy_escalation(bins, counts, n_train, b, base_cost)
        for tag, esc in (("optimal-dp", esc_opt), ("greedy-threshold", esc_greedy)):
            r = evaluate_cascade(
                test_ids, bin_of, esc, cc, ce, co_c, co_e, CHEAP, EXPENSIVE, oracle_test
            )
            casc_rows.append(
                {
                    "budget": b,
                    "policy": tag,
                    "escalated": "+".join(r.escalated_bins) or "(none)",
                    "test_accuracy": r.accuracy,
                    "test_cost": r.expected_cost,
                    "gain": r.gain,
                    "gain_cap": r.gain_cap,
                    "headroom_captured": r.headroom_captured,
                    "n_test": r.n_test,
                }
            )
            print(
                f"[cascade] budget~{b/base_cost:.1f}x base | {tag:15s} "
                f"acc={r.accuracy:.4f} cost=${r.expected_cost:.6f} "
                f"headroom={r.headroom_captured:.3f} esc={r.escalated_bins}"
            )

    with open(HERE / "cascade_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(casc_rows[0]))
        w.writeheader()
        w.writerows(casc_rows)

    # Random-gating sanity baseline at MATCHED cost: escalate a random
    # fraction f of test queries with the same expected spend as the learned
    # policy's cheapest non-trivial point. Fair comparison, same dollars.
    learned_cost = next(
        r["test_cost"] for r in casc_rows
        if r["policy"] == "optimal-dp" and r["escalated"] != "(none)"
    )
    mean_ch = float(np.mean([co_c[q] for q in test_ids]))
    mean_eh = float(np.mean([co_e[q] for q in test_ids]))
    f_match = max(0.0, (learned_cost - mean_ch) / mean_eh)
    rng = np.random.default_rng(0)
    rand_accs = []
    for _ in range(500):
        mask = rng.random(len(test_ids)) < f_match
        acc = float(np.mean([ce[q] if m else cc[q] for q, m in zip(test_ids, mask)]))
        rand_accs.append(acc)
    learned_acc = next(
        r["test_accuracy"] for r in casc_rows
        if r["policy"] == "optimal-dp" and r["escalated"] != "(none)"
    )
    print(
        f"[cascade] matched-cost random gating (f={f_match:.3f}, "
        f"${learned_cost:.6f}/q): mean acc={np.mean(rand_accs):.4f} "
        f"vs learned {learned_acc:.4f}"
    )

    # ------------------------------------------------------------------
    # Experiment 3: beta stability
    # ------------------------------------------------------------------
    stab_rows = []
    for d in sorted(set(ds_of.values())):
        sub = {q: correctness[q] for q in qids if ds_of[q] == d}
        st = beta_bootstrap(sub, n_boot=2000, seed=0)
        stab_rows.append(
            {
                "dataset": d,
                "n": len(sub),
                "beta": st.beta,
                "ci_low": st.ci_low,
                "ci_high": st.ci_high,
                "beta_half_a": st.beta_half_a,
                "beta_half_b": st.beta_half_b,
            }
        )
        print(
            f"[beta] {d:10s} beta={st.beta:.4f} 90% CI [{st.ci_low:.4f}, {st.ci_high:.4f}] "
            f"split-half {st.beta_half_a:.4f}/{st.beta_half_b:.4f}"
        )
    with open(HERE / "beta_stability.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stab_rows[0]))
        w.writeheader()
        w.writerows(stab_rows)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ds_names = [r["dataset"] for r in stab_rows]
    betas = [r["beta"] for r in stab_rows]
    lo = [r["beta"] - r["ci_low"] for r in stab_rows]
    hi = [r["ci_high"] - r["beta"] for r in stab_rows]
    ax2.errorbar(ds_names, betas, yerr=[lo, hi], fmt="o", capsize=5)
    ax2.set_ylabel("beta (co-failure rate)")
    ax2.set_title("Beta stability: 2000-query bootstrap, 90% CI")
    fig2.tight_layout()
    fig2.savefig(HERE / "beta_stability.png", dpi=120)

    print("done: results/*.csv + results/*.png")


if __name__ == "__main__":
    main()
