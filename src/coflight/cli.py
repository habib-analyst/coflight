"""Command line entry points.

    coflight certify MATRIX.json          # is orchestration worth building?
    coflight compare MATRIX.json          # per-dataset breakdown
    coflight sweep MATRIX.json            # beta vs pool size
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from coflight.decision import decide
from coflight.economics import cheapest_meeting, model_economics, pareto_front
from coflight.estimate import clopper_pearson, co_failure_rate
from coflight.matrix import load_matrix


def _load_pool(matrix, model_names: str | None) -> list[str]:
    if model_names:
        pool = [m.strip() for m in model_names.split(",") if m.strip()]
        missing = [m for m in pool if m not in matrix.all_models()]
        if missing:
            raise SystemExit(f"unknown model(s): {', '.join(missing)}")
    else:
        pool = matrix.models_answering_all()
    if len(pool) < 2:
        raise SystemExit("need at least 2 models in the pool")
    return pool


def cmd_certify(args: argparse.Namespace) -> int:
    matrix = load_matrix(args.matrix)
    pool = _load_pool(matrix, args.models)
    sub = matrix.filter_datasets(args.dataset) if args.dataset else matrix
    try:
        verdict = decide(sub.restrict_models(pool), pool, confidence=args.confidence)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"coflight certify  ({sub.source}, {len(pool)} models)")
    print()
    print(verdict.render())
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    matrix = load_matrix(args.matrix)
    pool = _load_pool(matrix, args.models)
    print(f"coflight compare  ({matrix.source}, {len(pool)} models)\n")
    header = (
        f"{'dataset':<10} {'n':>5} {'aw':>4} {'beta':>8} {'90% CI':>17} "
        f"{'rho':>6} {'best':>7} {'gap':>7} {'McNemar p':>10} {'gain':>7}  verdict"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for ds in matrix.datasets:
        sub = matrix.filter_datasets([ds]).restrict_models(pool)
        try:
            v = decide(sub, pool, confidence=args.confidence)
        except ValueError:
            continue
        c = v.beta
        rows.append((ds, v))
        p = v.gap_test.get("p_value")
        p_str = f"{p:.4f}" if p is not None else "n/a"
        print(
            f"{ds:<10} {c.n_queries:>5} {c.n_all_wrong:>4} {c.beta:>8.4f} "
            f"[{c.ci_low:.4f},{c.ci_high:.4f}] {v.rho:>6.3f} "
            f"{v.best_accuracy:>7.4f} {v.accuracy_gap:>7.4f} {p_str:>10} "
            f"{v.oracle_gain:>7.4f}  {v.label}"
        )

    if len(rows) > 1:
        betas = {ds: v.beta.beta for ds, v in rows}
        lo = min(betas, key=betas.get)
        hi = max(betas, key=betas.get)
        print()
        if betas[lo] > 0:
            print(
                f"beta ranges {betas[lo]:.4f} ({lo}) to {betas[hi]:.4f} ({hi}) "
                f"on the same pool: {betas[hi] / betas[lo]:.1f}x"
            )
        else:
            print(
                f"beta is 0.0000 on {lo} but {betas[hi]:.4f} on {hi} "
                f"-- same models, different workload"
            )
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """beta and oracle gain as a function of pool size, with a fixed seed."""
    matrix = load_matrix(args.matrix)
    pool = _load_pool(matrix, args.models)
    sub = matrix.filter_datasets(args.dataset) if args.dataset else matrix
    stats = sub.per_model(pool)
    ranked = [name for _, name in sorted((s.accuracy, s.name) for s in stats.values())[::-1]]

    rng = np.random.default_rng(args.seed)
    sizes = [k for k in args.sizes if 2 <= k <= len(ranked)]
    if not sizes:
        raise SystemExit("no valid pool sizes requested")

    print(f"coflight sweep  ({sub.source}, seed={args.seed})")
    print(
        f"subsets drawn from the {len(ranked)} models ranked by accuracy; "
        f"{args.trials} trials per size\n"
    )
    header = f"{'k':>4} {'beta':>9} {'pooled 90% CI':>17} {'best':>8} {'gap':>8} {'gain':>8}"
    print(header)
    print("-" * len(header))

    for k in sizes:
        betas, gains, accs, gaps = [], [], [], []
        total_aw, total_n = 0, 0
        for _ in range(args.trials):
            subset = [str(m) for m in rng.choice(ranked, size=k, replace=False)]
            corr = sub.correctness(subset)
            aw, n = co_failure_rate(corr)
            total_aw += aw
            total_n += n
            betas.append(aw / n)

            st = sub.per_model(subset)
            r = sorted((s.accuracy, s.name) for s in st.values())[::-1]
            best_acc = r[0][0]
            second = r[1][0] if len(r) > 1 else 0.0
            accs.append(best_acc)
            gaps.append(best_acc - second)
            gains.append(max(0.0, (1.0 - best_acc) - aw / n))

        lo, hi = clopper_pearson(total_aw, total_n, 0.90)
        print(
            f"{k:>4} {np.mean(betas):>9.4f} [{lo:.4f},{hi:.4f}] "
            f"{np.mean(accs):>8.4f} {np.mean(gaps):>8.4f} {np.mean(gains):>8.4f}"
        )
    return 0


def cmd_cost(args: argparse.Namespace) -> int:
    """Rank a pool by spend per correct answer and show what is worth buying."""
    matrix = load_matrix(args.matrix)
    pool = _load_pool(matrix, args.models)
    sub = matrix.filter_datasets(args.dataset) if args.dataset else matrix
    econ = model_economics(sub, pool)
    if not econ:
        raise SystemExit("no cost data available")

    front = set(pareto_front(econ))
    best = max(econ.values(), key=lambda e: e.accuracy)
    cheapest = next(iter(econ.values()))

    print(f"coflight cost  ({sub.source}, {len(econ)} models)")
    print(
        f"accuracy is not price. Ranked by dollars per correct answer.\n"
    )
    header = (
        f"{'':<2} {'accuracy':>8} {'$/query':>10} {'$/correct':>11}  model"
    )
    print(header)
    print("-" * (len(header) + 4))

    show = list(econ)
    if args.pareto_only:
        show = [n for n in econ if n in front]
    for name in show:
        e = econ[name]
        mark = "*" if name in front else " "
        print(
            f"{mark:<2} {e.accuracy:>8.4f} {e.mean_cost:>10.6f} "
            f"{e.cost_per_correct:>11.6f}  {name}"
        )

    print(f"\n  * on the Pareto front: not beaten on both accuracy and cost "
          f"({len(front)} of {len(econ)} models)")

    ratio = best.cost_per_correct / cheapest.cost_per_correct
    print(
        f"\n  most accurate: {best.name} at {best.accuracy:.4f} "
        f"(${best.cost_per_correct:.6f}/correct)"
    )
    print(
        f"  best value:    {cheapest.name} at {cheapest.accuracy:.4f} "
        f"(${cheapest.cost_per_correct:.6f}/correct)"
    )
    print(
        f"  -> {best.name} buys {best.accuracy / cheapest.accuracy:.2f}x the "
        f"accuracy for {ratio:.0f}x the price per correct answer"
    )

    print("\n  cheapest way to reach an accuracy target:")
    for target in (0.50, 0.60, 0.70, 0.80, 0.85):
        name, cpc = cheapest_meeting(econ, target)
        if name is None:
            print(f"    >= {target:.2f}  unreachable from this pool")
        else:
            print(f"    >= {target:.2f}  {name}  ${cpc:.6f}/correct")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coflight", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("matrix", help="outcome matrix JSON")
        p.add_argument("--models", help="comma-separated pool (default: all common)")
        p.add_argument("--dataset", action="append", help="restrict to dataset(s)")
        p.add_argument("--confidence", type=float, default=0.90)

    p1 = sub.add_parser("certify", help="is orchestration worth building?")
    common(p1)
    p1.set_defaults(func=cmd_certify)

    p2 = sub.add_parser("compare", help="per-dataset breakdown")
    common(p2)
    p2.set_defaults(func=cmd_compare)

    p3 = sub.add_parser("sweep", help="beta vs pool size")
    common(p3)
    p3.add_argument("--sizes", type=int, nargs="+", default=[3, 5, 8, 12, 20, 32, 52])
    p3.add_argument("--trials", type=int, default=25)
    p3.add_argument("--seed", type=int, default=0)
    p3.set_defaults(func=cmd_sweep)

    p4 = sub.add_parser("cost", help="rank by spend per correct answer")
    common(p4)
    p4.add_argument("--pareto-only", action="store_true", help="non-dominated only")
    p4.set_defaults(func=cmd_cost)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())