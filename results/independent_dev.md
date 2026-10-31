# Independent test results (dev set)

120 cases written by a separate Claude session that never saw our agent. Verdicts come from its own `judge()`. Ambiguous cases are excluded from the totals and listed separately. Intervals are 95% (Wilson), because the sample is small.

## 1. Overall

| Agent | Cases | Passed | Pass rate | 95% interval |
|---|---:|---:|---:|---:|
| do_nothing | 119 | 20 | 17% | 11-25% |
| rule_based | 119 | 88 | 74% | 65-81% |
| evolution_only | 119 | 96 | 81% | 73-87% |
| full_system | 119 | 93 | 78% | 70-85% |
| reasoning_only | 119 | 85 | 71% | 63-79% |
| reasoning_hybrid | 119 | 107 | 90% | 83-94% |
| reasoning_plus | 119 | 107 | 90% | 83-94% |

## 2. By kind of case (pass rate)

| Agent | fixable | unfixable (must page) | quiet (must stay quiet) |
|---|---:|---:|---:|
| do_nothing | 0% (0/68) | 0% (0/31) | 100% (20/20) |
| rule_based | 65% (44/68) | 97% (30/31) | 70% (14/20) |
| evolution_only | 74% (50/68) | 97% (30/31) | 80% (16/20) |
| full_system | 68% (46/68) | 100% (31/31) | 80% (16/20) |
| reasoning_only | 56% (38/68) | 100% (31/31) | 80% (16/20) |
| reasoning_hybrid | 88% (60/68) | 100% (31/31) | 80% (16/20) |
| reasoning_plus | 88% (60/68) | 100% (31/31) | 80% (16/20) |

## 3. Speed

Fixable cases that passed: mean unhealthy ticks (lower is better). Unfixable cases that paged: mean ticks from the first fault to the page.

| Agent | Unhealthy ticks (fixable, passed) | Ticks to page (unfixable) |
|---|---:|---:|
| do_nothing | - | - |
| rule_based | 8.7 | 61.8 |
| evolution_only | 35.3 | 55.0 |
| full_system | 36.6 | 33.3 |
| reasoning_only | 14.5 | 20.6 |
| reasoning_hybrid | 19.0 | 36.8 |
| reasoning_plus | 20.5 | 39.8 |

## 4. Pass rate by tag (what each case was designed to expose)

| Tag | n | do_nothing | rule_based | evolution_only | full_system | reasoning_only | reasoning_hybrid | reasoning_plus |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cascade | 13 | 0% | 85% | 77% | 69% | 69% | 85% | 85% |
| compound | 9 | 0% | 56% | 67% | 56% | 67% | 78% | 78% |
| data_corruption | 16 | 0% | 62% | 81% | 75% | 69% | 81% | 88% |
| escalation | 12 | 0% | 92% | 92% | 100% | 100% | 100% | 100% |
| flapping | 7 | 0% | 100% | 86% | 86% | 71% | 86% | 86% |
| healthy | 6 | 100% | 67% | 67% | 67% | 67% | 67% | 67% |
| intermittent | 3 | 0% | 100% | 100% | 67% | 100% | 100% | 100% |
| mild | 7 | 0% | 43% | 57% | 43% | 43% | 57% | 57% |
| odd | 6 | 100% | 83% | 100% | 100% | 100% | 100% | 100% |
| overlap | 5 | 0% | 60% | 40% | 60% | 40% | 100% | 100% |
| payment_leak | 13 | 0% | 62% | 77% | 77% | 38% | 77% | 77% |
| plain | 3 | 100% | 33% | 67% | 67% | 67% | 67% | 67% |
| profile | 6 | 33% | 83% | 83% | 67% | 83% | 100% | 100% |
| queue_overflow | 10 | 0% | 80% | 70% | 80% | 50% | 100% | 90% |
| recurrence | 6 | 0% | 83% | 67% | 67% | 83% | 83% | 83% |
| sensors | 11 | 36% | 45% | 73% | 64% | 55% | 82% | 82% |
| silent_dropper | 11 | 0% | 82% | 73% | 82% | 64% | 100% | 100% |
| tools | 4 | 0% | 100% | 100% | 100% | 100% | 100% | 100% |
| traffic | 3 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| workflow_deadlock | 20 | 0% | 80% | 95% | 95% | 70% | 95% | 95% |

## 5. Every case `reasoning_plus` did NOT pass (nothing hidden)

| Case | Kind | Why it failed |
|---|---|---|
| armed_cascade_resurfaces | fixable | root cause still present: ['queue_overload'] |
| barely_visible_dropper_1pct | fixable | root cause still present: ['silent_dropper'] |
| healthy_heavy_noise | quiet | false alarm raised on a healthy shop |
| healthy_noisy_sensors_long | quiet | false alarm raised on a healthy shop |
| mild_dropper_3pct | fixable | root cause still present: ['silent_dropper'] |
| overload_slow_creep_1_7x | fixable | root cause still present: ['queue_overload'] |
| rnd12345_001_quiet | quiet | false alarm raised on a healthy shop |
| rnd12345_028_fixable | fixable | paged a human on a fixable case |
| rnd12345_043_fixable | fixable | root cause still present: ['payment_leak', 'queue_overload'] |
| rnd12345_049_fixable | fixable | root cause still present: ['queue_overload'] |
| rnd12345_058_fixable | fixable | paged a human on a fixable case |
| rnd12345_059_quiet | quiet | false alarm raised on a healthy shop |
| tool_returns_after_outage (ambiguous) | fixable | paged a human on a fixable case |

## 6. Ambiguous cases (reasonable agents could disagree)

| Case | Agent | Passed | Reason |
|---|---|---|---|
| tool_returns_after_outage | do_nothing | no | root cause still present: ['silent_dropper'] |
| tool_returns_after_outage | evolution_only | no | paged a human on a fixable case |
| tool_returns_after_outage | full_system | no | paged a human on a fixable case |
| tool_returns_after_outage | reasoning_hybrid | no | paged a human on a fixable case |
| tool_returns_after_outage | reasoning_only | no | paged a human on a fixable case |
| tool_returns_after_outage | reasoning_plus | no | paged a human on a fixable case |
| tool_returns_after_outage | rule_based | no | paged a human on a fixable case |

## How to read this honestly

- The cases were written independently of our agent, but they use OUR simulator, so this shows robustness to someone else's scenarios, not to a real production system.
- If we change the agent after seeing the `dev` results, those cases become development cases. Only the sealed `final` set is clean evidence, and it is run once.
- Seeds are fixed, so re-running reproduces these numbers.
