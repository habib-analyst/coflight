# coflight

Before you build a router, a cascade, or a multi-agent ensemble: find out
whether one can work at all on your benchmark.

The usual justification is a leaderboard. The best model scores highest, a
runner-up is close behind, so route between them. That argument is missing a
term. Define the **co-failure rate** `beta` as the fraction of queries that
*every* model in the pool gets wrong. Then any policy that picks one model's
answer per query -- a router, a majority vote, a cascade -- is capped at
accuracy `1 - beta`, no matter how good the models are.

The field's standard diagnostic is the pairwise error correlation `rho`. It is
a reasonable number to report, but it is mathematically incapable of
identifying `beta`: many different joint error distributions produce the same
pairwise correlations. A pool can look nicely diverse by `rho` and still have
a ceiling that makes the ensemble pointless.

coflight computes `beta` with exact binomial confidence bounds, measures how
much of the headroom is actually rescuable, tests whether the leaderboard gap
is statistically real, weighs the winner against its price, and returns a
verdict.

## What it found on the public 67-model dataset

52 models, 530 fully-answered queries, three benchmarks. Every number below
reproduces with one command; the full writeup is in [FINDINGS.md](FINDINGS.md).

| dataset | n | beta | 90% CI | gap | McNemar p | verdict |
|---|---|---|---|---|---|---|
| gpqa | 130 | 0.0000 | [0.0000, 0.0228] | 0.1154 | 0.0026 | WORTH BUILDING |
| math500 | 200 | 0.0050 | [0.0003, 0.0235] | 0.0200 | 0.2891 | CAUTION |
| mmlu_pro | 200 | 0.0050 | [0.0003, 0.0235] | 0.0200 | 0.4240 | CAUTION |
| codegen | 63 | 0.0000 | [0.0000, 0.0464] | 0.0159 | 1.0000 | CAUTION |

Three results worth the read:

**`beta` is a property of the task, not of the roster.** The same 52 models
produce `beta` of 0.000, 0.005, and 0.005 on three benchmarks, and 0.000 on a
fourth drawn from a completely different pool. Model quality moves the
ceiling far less than answer format does.

**Leaderboards are mostly noise.** On three of the four benchmarks the top two
models are statistically indistinguishable. MMLU-Pro looks routable at
n=200 -- a 0.02 gap -- until you run the paired test and get p=0.42. A router
built on that ordering is routing on noise, and coflight says so rather than
returning a green light.

**The most accurate model is rarely the right one to buy.** On the same pool,
`claude-sonnet-4.6` buys 1.34x the accuracy of `ling-2.6-flash` for 233x the
price per correct answer. Only 6 of 52 models are on the Pareto front of
accuracy against cost. If you are choosing a model for a budget rather than
for a leaderboard, the cost command is the relevant one, and the accuracy
table is close to the wrong tool.

## Install

```bash
pip install -e .            # runtime: numpy only
pip install -e ".[dev]"     # adds pytest and scipy for the reference tests
```

The exact binomial bounds need the regularized incomplete beta function, so
`numerics.py` implements `I_x(a, b)` and its inverse by continued fraction and
bisection. That is deliberate: the reference implementation is a few dozen
lines, and it keeps a compiled dependency out of a package whose entire value
is that you can reproduce the numbers. The implementation is checked against
scipy in the test suite, where the confidence bounds agree to about `1e-14`.

## Use

```bash
# is orchestration on this pool worth building?
coflight certify data/matrix_marketE2.json

# restrict to one benchmark
coflight certify data/matrix_marketE2.json --dataset gpqa

# per-dataset table with verdicts
coflight compare data/matrix_marketE2.json

# how beta moves with pool size
coflight sweep data/matrix_marketE2.json --sizes 3 5 8 12 20 --trials 40

# which models are actually worth buying
coflight cost data/matrix_marketE2.json --pareto-only
```

Example:

```text
coflight certify  (matrix_marketE2.json, 52 models)

  beta (co-failure)   0.0000  [0.0000, 0.0228]   0/130 queries, k=52
  rho (pairwise)      0.2512   (field standard; cannot see beta)
  best model          0.8462  anthropic/claude-opus-4.8
  runner-up           0.7308   (gap +0.1154, McNemar p=0.0026  SIGNIFICANT)
  accuracy ceiling    1.0000   (1 - beta)
  ceiling gain        +0.1538
  oracle gain         +0.1538   (best wrong 20/130, rescuable 20)
  binding constraint  accuracy gap (routing headroom)

  VERDICT: WORTH BUILDING
```

## Your own data

Feed a JSON outcome matrix: `query_id -> model -> 0|1`, where `1` is correct.
Optional `cost` per cell buys you the cost analysis.

```json
{
  "q1": {
    "dataset": "my-bench",
    "models": {
      "gpt-5": { "correct": 1, "cost": 0.012 },
      "claude-opus-4.8": { "correct": 0, "cost": 0.009 }
    }
  }
}
```

```bash
coflight certify my_matrix.json --models gpt-5,claude-opus-4.8,gemini-3.1-pro
```

Any eval harness that emits per-model correctness works. Nothing in coflight
is specific to the bundled dataset.

If your organisation enforces Windows Application Control or WDAC, the
generated `coflight.exe` launcher may be blocked even though the package
installs cleanly. Run it as a module instead, which needs no launcher:

```bash
python -m coflight.cli certify data/matrix_marketE2.json
```

## What the verdicts mean

| verdict | meaning |
|---|---|
| `DO NOT BUILD` | Every best-model error is a pool-wide co-failure, or the top two are tied. No policy can help. |
| `CAUTION` | Ceiling is open, but the gap is not statistically real, the headroom is too small to matter, or `beta > 2%` caps accuracy anyway. |
| `WORTH BUILDING` | Ceiling is open *and* the gap survives an exact paired test. |

`WORTH BUILDING` is necessary, not sufficient. Oracle gain is what a perfect
router would capture with hindsight. Reaching it means finding the same
queries *before* seeing any answer, which is the part still unsolved -- the
source paper's own routing experiments captured essentially none of the oracle
gain. coflight tells you the ceiling and the headroom. It does not claim you
can reach either.

## Method

- **`beta`, exact.** Clopper-Pearson bounds, not normal-approximation. All-wrong
  events are rare by construction, which is exactly where the normal
  approximation is worst and where an optimistic interval understates risk.
- **Oracle gain** `= P(best wrong) - beta`: the error mass a perfect selector
  could recover, as distinct from the co-failure mass nothing can touch.
- **Exact McNemar** on discordant queries. Two models answering the same
  queries is a paired design; comparing their individual confidence intervals
  is valid but underpowered, and on this data it changes the verdict.
- **Common-shock decomposition** (Marshall-Olkin) in log space, to separate
  co-hard queries from independent error. It reports `identifiable: false`
  rather than returning a confident number when the pool is too small to
  separate the two.
- **Pool-size sweep** with model quality held in a band, because the naive
  sweep is confounded -- ranking by accuracy and adding more means adding
  *weaker* models, which lowers `beta` for reasons unrelated to co-failure.
- **Cost per correct answer**, not accuracy and price separately. A 3x more
  expensive model that is 2x more accurate is not 3x more expensive to
  operate.

## Correctness

Defects found and fixed during development are documented in
[FINDINGS.md](FINDINGS.md) section 4, because each one changed a conclusion.
Two are worth naming here, since they share a shape: no exception, no
obviously wrong number, just a guard silently not running. The significance
result was computed and then never attached to the returned verdict; an
explicitly empty model pool fell through to analysing the entire matrix. Both
are now covered by tests that assert on guard *presence*, not only on the
verdict label.

```bash
pytest -q      # 55 passed
```

## Data

`data/` holds the outcome matrices and price registry from the public dataset
released with the co-failure study ([arXiv 2606.27288](https://arxiv.org/abs/2606.27288),
`josefchen/co-failure-67-models` on Hugging Face). Bundled so every number
here is reproducible offline, with no API calls and no cost.

## Author

Built by [Habib Ur Rehman](https://github.com/habib-analyst) ([@habib-analyst](https://github.com/habib-analyst)).
