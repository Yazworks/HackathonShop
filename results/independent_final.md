# Independent test results (final set)

100 cases written by a separate Claude session that never saw our agent. Verdicts come from its own `judge()`. Ambiguous cases are excluded from the totals and listed separately. Intervals are 95% (Wilson), because the sample is small.

## 1. Overall

| Agent | Cases | Passed | Pass rate | 95% interval |
|---|---:|---:|---:|---:|
| do_nothing | 100 | 21 | 21% | 14-30% |
| rule_based | 100 | 63 | 63% | 53-72% |
| evolution_only | 100 | 74 | 74% | 65-82% |
| full_system | 100 | 77 | 77% | 68-84% |
| reasoning_only | 100 | 71 | 71% | 61-79% |
| reasoning_hybrid | 100 | 89 | 89% | 81-94% |
| reasoning_plus | 100 | 84 | 84% | 76-90% |

## 2. By kind of case (pass rate)

| Agent | fixable | unfixable (must page) | quiet (must stay quiet) |
|---|---:|---:|---:|
| do_nothing | 0% (0/53) | 0% (0/26) | 100% (21/21) |
| rule_based | 55% (29/53) | 85% (22/26) | 57% (12/21) |
| evolution_only | 70% (37/53) | 85% (22/26) | 71% (15/21) |
| full_system | 68% (36/53) | 100% (26/26) | 71% (15/21) |
| reasoning_only | 57% (30/53) | 100% (26/26) | 71% (15/21) |
| reasoning_hybrid | 91% (48/53) | 100% (26/26) | 71% (15/21) |
| reasoning_plus | 83% (44/53) | 96% (25/26) | 71% (15/21) |

## 3. Speed

Fixable cases that passed: mean unhealthy ticks (lower is better). Unfixable cases that paged: mean ticks from the first fault to the page.

| Agent | Unhealthy ticks (fixable, passed) | Ticks to page (unfixable) |
|---|---:|---:|
| do_nothing | - | - |
| rule_based | 7.8 | 61.8 |
| evolution_only | 39.8 | 78.2 |
| full_system | 41.4 | 49.3 |
| reasoning_only | 15.8 | 35.7 |
| reasoning_hybrid | 21.2 | 52.0 |
| reasoning_plus | 18.5 | 45.8 |

## 4. Pass rate by tag (what each case was designed to expose)

| Tag | n | do_nothing | rule_based | evolution_only | full_system | reasoning_only | reasoning_hybrid | reasoning_plus |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| burst | 3 | 100% | 67% | 67% | 67% | 67% | 67% | 67% |
| cascade | 12 | 0% | 58% | 67% | 50% | 75% | 92% | 92% |
| compound | 12 | 0% | 67% | 75% | 83% | 67% | 100% | 83% |
| data_corruption | 18 | 0% | 61% | 67% | 56% | 67% | 89% | 78% |
| diurnal | 6 | 100% | 83% | 100% | 100% | 100% | 100% | 100% |
| flapping | 15 | 0% | 53% | 67% | 60% | 53% | 93% | 73% |
| high_load | 4 | 100% | 0% | 25% | 25% | 25% | 25% | 25% |
| payment_leak | 19 | 0% | 32% | 63% | 74% | 32% | 84% | 68% |
| plain | 6 | 100% | 67% | 83% | 83% | 83% | 83% | 83% |
| queue_overflow | 18 | 0% | 61% | 72% | 89% | 67% | 100% | 94% |
| silent_dropper | 18 | 0% | 78% | 78% | 89% | 83% | 94% | 89% |
| workflow_deadlock | 28 | 0% | 71% | 64% | 71% | 75% | 89% | 86% |

## 5. Every case `reasoning_plus` did NOT pass (nothing hidden)

| Case | Kind | Why it failed |
|---|---|---|
| rnd20261031_001_quiet | quiet | false alarm raised on a healthy shop |
| rnd20261031_005_unfixable | unfixable | never paged a human although root cause cannot be removed |
| rnd20261031_018_quiet | quiet | false alarm raised on a healthy shop |
| rnd20261031_023_fixable | fixable | paged a human on a fixable case |
| rnd20261031_026_fixable | fixable | root cause still present: ['queue_overload'] |
| rnd20261031_030_quiet | quiet | false alarm raised on a healthy shop |
| rnd20261031_043_fixable | fixable | paged a human on a fixable case |
| rnd20261031_047_fixable | fixable | paged a human on a fixable case |
| rnd20261031_049_quiet | quiet | false alarm raised on a healthy shop |
| rnd20261031_051_fixable | fixable | paged a human on a fixable case |
| rnd20261031_054_quiet | quiet | false alarm raised on a healthy shop |
| rnd20261031_068_fixable | fixable | paged a human on a fixable case |
| rnd20261031_076_fixable | fixable | paged a human on a fixable case |
| rnd20261031_085_fixable | fixable | paged a human on a fixable case |
| rnd20261031_095_quiet | quiet | false alarm raised on a healthy shop |
| rnd20261031_097_fixable | fixable | paged a human on a fixable case |

## How to read this honestly

- The cases were written independently of our agent, but they use OUR simulator, so this shows robustness to someone else's scenarios, not to a real production system.
- If we change the agent after seeing the `dev` results, those cases become development cases. Only the sealed `final` set is clean evidence, and it is run once.
- Seeds are fixed, so re-running reproduces these numbers.
