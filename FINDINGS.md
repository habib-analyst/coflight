# coflight: findings

All numbers reproduce from the bundled matrices in `data/`, from the public
dataset released with the co-failure study (arXiv 2606.27288,
`josefchen/co-failure-67-models` on Hugging Face). No API calls were made;
total cost of the analysis was $0.

Two matrices are analysed:

- `data/matrix_marketE2.json` -- 598 queries, 67 models, three benchmarks
  (gpqa, math500, mmlu_pro). 52 models answer every query, giving 530 fully
  answered rows.
- `data/matrix_marketCG.json` -- 63 queries, 18 models, code generation. A
  different task family and a different pool, included as a control for the
  format claim in section 1.

---

## 1. The ceiling is open on this pool

| dataset | n | all-wrong | beta | 90% CI | rho | best | gap | oracle gain | verdict |
|---|---|---|---|---|---|---|---|---|---|
| gpqa | 130 | 0 | 0.0000 | [0.0000, 0.0228] | 0.251 | 0.8462 | 0.1154 (p=0.0026) | 0.1538 | WORTH BUILDING |
| math500 | 200 | 1 | 0.0050 | [0.0003, 0.0235] | 0.381 | 0.9800 | 0.0200 (p=0.2891) | 0.0150 | CAUTION |
| mmlu_pro | 200 | 1 | 0.0050 | [0.0003, 0.0235] | 0.293 | 0.9300 | 0.0200 (p=0.4240) | 0.0650 | CAUTION |
| codegen | 63 | 0 | 0.0000 | [0.0000, 0.0464] | 0.220 | 0.9048 | 0.0159 (p=1.0000) | 0.0952 | CAUTION |

The verdicts are driven by the paired significance test in section 3, not by
beta. Only GPQA has a model gap that survives it.

The published paper reports beta = 0.052 on open-ended mathematics for a
17-model pool. On this 52-model pool, restricted to multiple-choice
benchmarks, beta is 0.004. The difference is format, not pool quality: the
paper showed the same effect by re-asking GPQA in free-response form and
watching beta reopen to 0.127.

**beta tracks task format, not subject.** The same 52 models, the same
queries, different answer format: 0.000 on GPQA-MC, 0.005 on MATH-500 and
MMLU-Pro.

## 2. beta decays with pool size even at matched quality

The naive sweep is confounded: ranking models by accuracy and adding more
means adding *weaker* models, and weak models err on easy queries where a
strong model is right. That drives beta down for reasons unrelated to the
common-shock effect.

Restricting to a band of models within 0.10 accuracy of the best holds quality
roughly constant:

| k | beta (MATH-500) | beta (MMLU-Pro) | best | gap | gain |
|---|---|---|---|---|---|
| 3 | 0.0195 [0.0175, 0.0217] | 0.0500 [0.0468, 0.0534] | 0.885 / 0.840 | 0.031 / 0.052 | 0.031 / 0.052 |
| 5 | 0.0125 [0.0109, 0.0143] | 0.0391 [0.0362, 0.0421] | | | 0.035 / 0.049 |
| 8 | 0.0089 [0.0076, 0.0105] | 0.0313 [0.0288, 0.0341] | | | 0.033 / 0.046 |

Pooled across all 530 queries with a 9-model matched band: beta = 0.0723
(k=3) -> 0.0434 (k=5) -> 0.0325 (k=8). The confidence intervals do not
overlap, so this is not sampling noise.

**This is a real tension with the published result.** The paper's Proposition
2 predicts the *ratio* beta(m) / beta_single_factor(m) grows with m, and that a
common-shock atom keeps beta above a floor. Both are consistent with what we
see: beta falls with k, and the *rate of fall* slows as k grows. The
52-model pool sits at 0.004.

One comparison is not available and should not be claimed. The
independence baseline is a product of 52 marginals and comes out at
\(5.0\times10^{-24}\), i.e. zero for every practical purpose, so "beta is far
above independent errors" is unfalsifiable on a pool this large. Independence
is only a useful baseline on small pools. The honest statement of the tension
is the flattening slope in the table above, not a ratio against a degenerate
reference.

## 3. The accuracy gap is usually not statistically real

This is the finding most likely to change a deployment decision, and it was
invisible until the significance test was added.

Comparing the best model to the runner-up on the *same* queries is a paired
design, so each model's own confidence interval is the wrong instrument: it
treats two models measured on identical queries as independent samples. On
MMLU-Pro the intervals overlap heavily and the unpaired reading is that the
0.0200 gap is real.

| dataset | best | runner-up | gap | unpaired CP overlap? | exact McNemar p | significant |
|---|---|---|---|---|---|---|
| gpqa | 0.8462 | 0.7308 | 0.1154 | no (0.784 vs 0.794) | 0.0026 | **yes** |
| math500 | 0.9800 | 0.9600 | 0.0200 | yes | 0.2891 | no |
| mmlu_pro | 0.9300 | 0.9100 | 0.0200 | yes | 0.4240 | no |

The paired test is both the correct instrument and the more powerful one, and
it disagrees with the unpaired reading on MMLU-Pro. Discordant queries on
MMLU-Pro: 9 in the best model's favour, 5 against.

**Practical consequence:** of the three benchmarks, only GPQA supports
building a router. On MATH-500 and MMLU-Pro the leader is indistinguishable
from the runner-up at n=200, so a system that routes between them is routing
on noise. The tool returns CAUTION for both.

For GPQA the ceiling is fully open (beta = 0.0000, ceiling 1.0000) and the
gap is real (p=0.0026, 19 discordant wins to 4), which is the one genuine
WORTH BUILDING in this dataset.

The oracle gain of 0.1538 on GPQA is still not automatically reachable. A
selection policy must identify those 20 queries in advance, from signal
available before seeing any answer. Measured on this data that signal is
weak: the source paper's own routing experiments captured essentially none of
the oracle gain on open-ended tasks. So the correct chain of requirements is
beta low, *and* gap significant, *and* a query-level predictor that actually
works. The first two are necessary, not sufficient.

## 4. Method notes / corrections made during development

Six defects were found and fixed, all recorded because each one changed a
conclusion:

1. **Tuple order in `oracle_gain`.** `co_failure_rate` returns
   `(all_wrong, total)`; the caller unpacked it as `(total, all_wrong)`, so
   beta came out as 1.0 whenever a pool had no co-failure. This reported
   oracle gain of 0.0000 for GPQA, which is wrong -- 20 of 130 GPQA errors
   are rescuable. Guarded by `test_beta_is_zero_when_all_errors_are_rescuable`.

2. **Degenerate common-shock estimator.** The homogeneous
   Marshall-Olkin fit assumed all models share one independent error rate.
   On a real pool whose accuracy spans 0.33 to 0.88, that assumption forces
   alpha0 -> 0 and predicts beta = 0.34 against an observed 0.004. The
   product-of-margins form also underflows to zero by k=52 (~1e-40), which
   returned beta_fit = 1.0. The estimator now computes in log space and
   refuses to report a value when the pool is too small or the moment
   estimator is not identifiable, rather than emitting a confident wrong
   number. Guarded by
   `test_common_shock_survives_large_pool_underflow` and
   `test_common_shock_flags_small_pools_as_unidentifiable`.

3. **Independence baseline is degenerate at scale.** `prod(1 - alpha_i)` over
   the 52-model pool evaluates to 5.0e-24, so the baseline pins to
   beta_independent = 1.0 and any "how many times above independent?" ratio is
   vacuous. Pinned by `test_independence_baseline_saturates_on_wide_pool` so
   the limitation is explicit rather than rediscovered.

4. **The significance result was computed and then dropped.** `paired_gap` was
   implemented, called, and its result stored in a local `gap_test`, but the
   `Verdict(...)` constructor call omitted the field. The default empty dict
   was returned, `if gap_test and not gap_test["significant"]` was falsy, and
   the branch never fired. The `else` branch then formatted
   `gap_test['p_value']`, which would have raised `KeyError` on the first
   genuinely significant pool; every dataset that reached it was in the
   not-significant state, so the CLI kept printing confident verdicts.
   Guarded by `test_insignificant_gap_blocks_the_build` and
   `test_real_headroom_is_worth_building`, which assert on the field itself
   rather than on the label.

5. **Unpaired test on a paired design.** Each model's accuracy was compared
   through its own Clopper-Pearson interval, which is valid but
   underpowered when two models answer the same queries. It reported MMLU-Pro
   as routable. The exact McNemar test on discordant pairs reverses that.

6. **Empty pool silently analysed the whole roster.** `decide` used
   `if models` to detect "no pool given", so an explicitly empty list fell
   through to `models_answering_all()`. A caller who filtered a pool down to
   nothing would have received a confident verdict for all 52 models instead
   of an error. Caught by `test_empty_pool_raises`.

All six were caught by tests written against hand-constructed matrices with
known answers, not against the real data. Bugs 1 and 3 surfaced directly from
the real data. Bug 4 was the dangerous one: it produced no error and no wrong
number, just a guard that silently never ran. It was found only because a
test asserted on the field's presence, and that is an argument for testing
that guards are wired in, not just that they are correct in isolation. Bug 6
is the same shape: a falsy-check where a `None`-check was meant, so a
degenerate input produced a plausible answer rather than a failure.

A second audit pass, after the cost analysis was added, found four more of the
same family. All four are in `tests/test_audit_regressions.py`:

7. **Two different "best models" for the same pool.** `pool_report` reached
   the winner through `max()` over a dict, `decide` through a
   `(accuracy, name)` sort. On tied accuracy they picked different models, so
   the two entry points contradicted each other on the same input. Both now
   go through one `rank_models` helper. Caught by
   `test_ranking_is_shared_so_best_model_cannot_diverge`.

8. **`pool_report` had the same empty-pool bug as `decide`**, and neither had
   a test. Caught by `test_pool_report_raises_on_explicitly_empty_pool`.

9. **A diagnostic that contradicted itself.** When the common-shock moment
   estimate exceeded its identifiability threshold, the code zeroed `pi` and
   then formatted the message from the already-zeroed value, printing
   "moment estimate pi=0.000 exceeds 0.75". The actual estimate, 0.962, was
   discarded. Caught by
   `test_shock_note_reports_the_estimate_not_the_clamped_value`.

10. **Three return annotations were false.** `common_shock_fit`, `paired_gap`
    and `oracle_gain` were all annotated `-> dict[str, float]` while returning
    `bool` and `str` values among the floats. Replaced with `TypedDict`.

A third pass found no further defects, and confirmed two suspicions were
false alarms: `restrict_models` does not corrupt `datasets`, and an empty
`Matrix` handles `models_answering_all` safely. Both were checked rather than
assumed, which is why they are not listed as fixes.

The pattern across all ten is worth stating plainly: every one of them was a
guard that appeared to work. None raised. The tests that caught them assert on
values and on field presence, not on the absence of an exception.

## 5. What this does and does not establish

Established here:

- beta is a function of task format, not of model pool (same pool, beta 0.000
  to 0.005 across formats; confirmed on a second, disjoint codegen pool)
- beta falls with pool size at held quality, consistent with a common-shock
  floor rather than independence
- a co-failure ceiling is usually not the binding constraint
- on 3 of 4 benchmarks the best-vs-runner-up gap is not statistically
  significant, so model leaderboards at n=200 do not support routing
- accuracy and cost disagree sharply: 6 of 52 models are worth considering,
  and the most accurate is 233x the price per correct answer for 1.34x the
  accuracy

Not established, and open:

- whether a query-level predictor of beta is learnable. Answering this needs
  the paper's own open question, and requires running models, which costs
  money.
- whether the cost axis changes the verdict. `cost_usd` is present in the
  public data and unused by any analysis we found, but a budget-normalised
  break-even needs a price assumption to test against.

## 6. Cost: accuracy and price disagree sharply

The outcome matrix carries a per-query `cost` for every model, spanning three
orders of magnitude across the pool. Ranking by accuracy and ranking by
spend per correct answer produce almost unrelated orderings.

Only 6 of 52 models are on the Pareto front of accuracy against cost per
correct answer:

| model | accuracy | $/query | $/correct |
|---|---|---|---|
| inclusionai/ling-2.6-flash | 0.6585 | 0.000022 | 0.000033 |
| qwen/qwen3-235b-a22b-2507 | 0.6660 | 0.000088 | 0.000133 |
| openai/gpt-oss-120b | 0.7925 | 0.000145 | 0.000183 |
| deepseek/deepseek-v3.2 | 0.8208 | 0.000290 | 0.000354 |
| meta-llama/llama-4-maverick | 0.8321 | 0.000377 | 0.000453 |
| anthropic/claude-sonnet-4.6 | 0.8849 | 0.006745 | 0.007622 |

The most accurate model buys 1.34x the accuracy of the cheapest per correct
answer for 233x the price. Cheapest route to each accuracy target:

| target | cheapest per correct answer | price |
|---|---|---|
| >= 0.50 | ling-2.6-flash | $0.000033 |
| >= 0.60 | ling-2.6-flash | $0.000033 |
| >= 0.70 | gpt-oss-120b | $0.000183 |
| >= 0.80 | deepseek-v3.2 | $0.000354 |
| >= 0.85 | claude-sonnet-4.6 | $0.007622 |

Reading: a 0.70 accuracy target costs 42x less per correct answer than a 0.85
one. Whether that trade is worth making is a product decision, but it should
be a decision, and it is invisible from an accuracy leaderboard.

A caution on the raw per-query cost. Provider pricing and token counts differ
across harnesses and dates, so treat the 233x as an order of magnitude, not a
constant. The method transfers; the absolute figures reflect the prices
recorded in the source dataset.
