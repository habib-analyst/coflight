[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)] [![GitHub stars](https://img.shields.io/github/stars/habib-analyst/coflight.svg)]

# coflight

Estimate the co-failure ceiling of an LLM pool, and decide whether a router is worth building before you build it.

Define the **co-failure rate** `beta` as the fraction of queries that *every* model in the pool gets wrong. Then any policy that picks one model's answer per query — a router, a majority vote, a cascade — is capped at accuracy `1 - beta`, no matter how good the models are. The field's standard diagnostic, pairwise error correlation `rho`, is mathematically incapable of identifying `beta`: many different joint error distributions produce the same pairwise correlations. coflight computes `beta` with exact binomial confidence bounds and returns a verdict.

## Quickstart

```bash
git clone https://github.com/habib-analyst/coflight.git && cd coflight
pip install -e .          # runtime: numpy only
coflight certify data/matrix_marketE2.json
```

Real output on the bundled 52-model dataset:

```
coflight certify  (matrix_marketE2.json, 52 models)

  beta (co-failure)   0.0000  [0.0000, 0.0228]   0/130 queries, k=52
  rho (pairwise)      0.2512   (field standard; cannot see beta)
  best model          0.8462  anthropic/claude-opus-4.8
  runner-up           0.7308  (gap +0.1154, McNemar p=0.0026  SIGNIFICANT)
  accuracy ceiling    1.0000  (1 - beta)
  ceiling gain        +0.1538
  oracle gain         +0.1538   (best wrong 20/130, rescuable 20)
  binding constraint  accuracy gap (routing headroom)

  VERDICT: WORTH BUILDING
```

Also: `coflight compare` (per-dataset table), `coflight sweep` (beta vs pool size), `coflight cost` (Pareto frontier), `coflight route` (cost-optimal per-dataset routing — see `ROUTING.md`). Feed your own outcome matrix: `query_id -> model -> 0|1` with optional per-cell cost.

## Measured results

Routing under the ceiling, measured on the bundled data (gating signals fit on train queries, scored on held-out test queries):

| policy | accuracy | note |
|---|---|---|
| best single model | 0.8849 | — |
| per-dataset knapsack routing | **0.9283** | +4.3pp (+2.8pp at identical spend) |
| learned cascade gating | — | **+2.3pp** over cost-matched random gating |

Honest negative result: dataset-level routing captures only 39% of the oracle headroom — the rest needs query-level rescue prediction, which is still unsolved. `WORTH BUILDING` is necessary, not sufficient: the oracle gain is what a perfect router would capture with hindsight, and reaching it means finding the same queries *before* seeing any answer.

## How it works

- **`beta`, exact.** Clopper-Pearson binomial bounds via a from-scratch implementation of the regularized incomplete beta function (checked against scipy to ~1e-14 in the test suite) — no normal approximation, no extra dependencies.
- **Gap test.** McNemar's exact paired test on the top-two models, so you don't route on leaderboard noise.
- **Routing.** Per-dataset model selection as an exact multiple-choice knapsack (threshold structure via its Lagrangian relaxation); disagreement-gated cascades whose gain is provably capped by `P(cheap wrong) − beta`.
- **Verdicts.** `DO NOT BUILD` (every best-model error is pool-wide co-failure, or the top two are tied) · `CAUTION` (ceiling open but the gap isn't statistically real or is too small to matter) · `WORTH BUILDING` (ceiling open *and* the gap survives the paired test).

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
