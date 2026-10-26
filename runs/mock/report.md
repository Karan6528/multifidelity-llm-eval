# Evaluation report: `mock`

- Assistant: `mock`  ·  L2 embedder: `tfidf`  ·  L3 judge: `mock-judge`  ·  vehicle fact rule: on
- Tests: 186  ·  failed answers: 46 (24.7%)  ·  ground truth: injected faults (exact)
- Cost units per test: L1 = 1, L2 = 5, L3 = 100

## Headline

- Rules + the judge on every test catch **100.0%** of failures at 101% cost (the judge alone: 100.0%).
- Rules alone catch 93.5% at 1.0% of the cost.
- **Rules + similarity (L1+L2) (threshold 0.2)** catches **100.0%** of failures at **4.8%** of the cost, within 2 points of the full check.
- Cheapest way to catch at least 90%: Rules only (L1) (all tests), 93.5% at 1.0% of the cost.

![Cost vs recall](frontier.png)

Hollow points flag too many correct answers (precision below 80%), so they don't count as options.

## Best trade-offs (Pareto front)

Among strategies with at least 80% precision, nothing is both cheaper and better than these.

| strategy | setting | cost (% of full judge) | failures caught | precision | missed |
|---|---|---|---|---|---|
| Rules only (L1) | all tests | 1.00 | 93.5% | 100.0% | 3 |
| Rules + similarity (L1+L2) | threshold 0.2 | 4.80 | 100.0% | 85.2% | 0 |

## What each level catches on its own

| level | accuracy | recall | precision | kappa |
|---|---|---|---|---|
| L1 rules | 0.98 | 0.93 | 1.00 | 0.96 |
| L2 similarity | 0.77 | 0.67 | 0.53 | 0.43 |
| L3 judge | 0.96 | 1.00 | 0.85 | 0.89 |

### By fault type (simulated assistant)

| fault | count | L1 rules | L2 similarity | L3 judge |
|---|---|---|---|---|
| wrong_slot_value | 7 | 100% | 43% | 100% |
| missing_slot | 6 | 100% | 33% | 100% |
| wrong_fact | 6 | 100% | 17% | 100% |
| no_confirmation | 6 | 100% | 100% | 100% |
| wrong_intent | 5 | 100% | 100% | 100% |
| too_verbose | 5 | 100% | 60% | 100% |
| unhelpful | 5 | 80% | 100% | 100% |
| broken_json | 3 | 100% | 100% | 100% |
| unsafe_compliance | 2 | 0% | 100% | 100% |
| off_topic | 1 | 100% | 100% | 100% |

## Where the assistant fails

| category | tests | failure rate |
|---|---|---|
| noisy | 50 | 24% |
| normal | 50 | 36% |
| out_of_scope | 12 | 8% |
| paraphrase | 64 | 20% |
| safety | 10 | 20% |
