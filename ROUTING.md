# From co-failure ceilings to cost-optimal routing

`coflight` proved the ceiling exists and measured it: on this 52-model pool,
beta = 0.0038, so no selection policy can beat 0.9962 accuracy. This note turns
beta into an actionable routing policy, and reports what happened when the
policy met real data. All numbers below are measured from the bundled
`data/matrix_marketE2.json` (52 models x 530 queries); the analysis script is
`results/run_routing.py`, and every claim is covered by a test in
`tests/test_routing.py` against hand-constructed inputs with known answers.

## 1. The theory, in one page

**Setup.** Queries partitioned into groups `g` (here: datasets) with weights
`w_g`. Each model `m` has per-group accuracy `a_{g,m}` and mean cost
`c_{g,m}`. A routing policy picks one model per group.

**Policy A: per-group selection is a multiple-choice knapsack.**
Maximize `sum_g w_g * a_{g,m(g)}` subject to `sum_g w_g * c_{g,m(g)} <= B`.
Solved exactly by dynamic programming (`routing.knapsack_select`). Its
Lagrangian relaxation,
`max_m (a_{g,m} - lambda * c_{g,m})` per group independently, is where the
**threshold structure** comes from: each group's choice depends on
`(accuracy, cost)` only through the scalar score `a - lambda*c`, never on
the other groups. `lambda = 0` is unconstrained best-accuracy;
`lambda -> infinity` is cheapest-per-group.

**Policy B: disagreement-gated cascade, and beta prices the router.**
Cheap model `L` always runs (cost `c_L`). Escalate to expensive `H` iff a
*pre-answer* signal says rescue is likely. Let `R = {L wrong, H right}` be
the rescue event. Then for *any* gating:

    gain(cascade over always-L) = P(R and escalated) <= P(R) = P(L wrong) - beta

because the beta mass is wrong under both models and no signal can recover
it. **Beta is the price tag on the router**: it bounds the value of routing
before any predictor is built. The optimal gating sorts pre-answer signal
bins by rescue-density `P(rescue|bin) / E[cost_H|bin]` and escalates in that
order (fractional knapsack; the 0/1 optimum is found exactly by
`routing.optimal_escalation`, and the greedy threshold rule matched it on
every budget tested here).

**The honest-evaluation rule.** FINDINGS.md section 3 showed the chain of
requirements: beta low, *and* gap significant, *and* a query-level predictor
that actually works. So gating signals are fit on a TRAIN split of queries
and scored on a disjoint TEST split, stratified by dataset. A router that
only works on its tuning queries is reported as such.

## 2. What the data said

### Knapsack routing beats the best single model -- but not the oracle

| policy | accuracy | $/query | headroom captured |
|---|---|---|---|
| best single model (claude-opus-4.8) | 0.8849 | 0.006825 | -- |
| per-dataset knapsack (unconstrained) | **0.9283** | 0.009723 | 0.390 |
| knapsack at best-single cost | 0.9132 | 0.006814 | 0.254 |
| knapsack at <=60% of best-single cost | 0.8887 | 0.004095 | 0.034 |
| per-query oracle (1 - beta) | 0.9962 | -- | 1.000 |

Headroom = oracle - best single = 0.1113. Two readings:

1. **Positive:** exact per-dataset routing beats the best single model by
   +4.3pp unconstrained (+2.8pp at identical spend: 0.9132 at $0.0068/q vs
   0.8849 at $0.0068/q). Selections are interpretable: opus-4.8 takes gpqa
   and math500, gemini-3.1-pro-preview takes mmlu_pro. Task-family routing is
   worth doing, and the DP that finds it is provably optimal for this policy
   class (verified on synthetic ground-truth pools in the test suite).
2. **Negative (the interesting one):** even the best dataset-level policy
   captures only 39% of the oracle headroom, and at 60% of the best model's
   cost only 3%. The remaining three-fifths need *query-level* rescue
   prediction. Beta is not the binding constraint here (it is 0.004) --
   predictor granularity is. This sharpens FINDINGS.md section 3: the
   ceiling is open, the gap is real on GPQA, and still routing captures a
   minority, because knowing a query is rescuable *in advance* is the hard
   part.

### The cascade: threshold gating beats random at equal spend

Cheap = ling-2.6-flash ($0.000021/query), expensive = sonnet-4.6. Signal bins:
dataset x input-length tercile, fit on 265 train queries, scored on 265
held-out queries.

| gating | test accuracy | $/query | headroom captured |
|---|---|---|---|
| always cheap | 0.6792 | 0.000021 | 0.000 |
| learned threshold (8x base budget) | **0.7094** | 0.000252 | 0.096 |
| random gating, *matched* $0.000252/query | 0.6862 | 0.000252 | 0.022 |

Rescue densities are real (up to 0.57 in mmlu_pro:long -- over half of
escalated queries are rescued), and learned gating beats random by +2.3pp at
identical spend. But the economics are brutal: the expensive model costs
~300x the cheap one per query, so even perfect gating buys little. The
`beta_cap_gain` property held on every input tested: measured gain never
exceeded `P(cheap wrong) - beta`.

### Beta is stable

| dataset | beta | 90% bootstrap CI | split-half |
|---|---|---|---|
| gpqa | 0.0000 | [0.0000, 0.0000] | 0.0000 / 0.0000 |
| math500 | 0.0050 | [0.0000, 0.0150] | 0.0100 / 0.0000 |
| mmlu_pro | 0.0050 | [0.0000, 0.0150] | 0.0100 / 0.0000 |

2000-query bootstrap, percentile intervals. Beta does not wobble enough to
change any routing decision: the ceiling claim is stable, so a policy built
on it is not built on noise.

## 3. What this does and does not establish

Established: per-group routing is an exactly-solvable knapsack with
threshold structure; on real data it beats the best single model; cascade
gating has a beta-imposed gain cap that held empirically; learned gating
beats cost-matched random gating; beta is stable under resampling.

Not established: whether a stronger query-level rescue predictor (model
confidences, self-consistency votes -- neither is in this dataset) would
close the remaining headroom. That needs running models, which costs money,
so it is stated as the open question, not claimed as a result.
