# HackathonShop - evaluation results

All situations are held out: harsher than tuning settings, shops built from fresh seeds, **3 seed(s) per situation**, every agent starts with an empty memory. Results come from the simulated shop.

## 1. Headline: situations where a fix exists

Success = shop lastingly healthy at the end, root cause really gone, human not paged.

| Agent | Episodes | Success rate | Mean unhealthy ticks (lower is better) | Mean ticks to lasting recovery | Mean actions |
|---|---:|---:|---:|---:|---:|
| do_nothing | 69 | 0% | 245.4 | - | 0.0 |
| rule_based | 69 | 93% | 8.2 | 16.5 | 2.6 |
| random_search | 69 | 77% | 58.8 | 52.2 | 1.8 |
| evolution_only | 69 | 91% | 48.5 | 44.0 | 1.6 |
| no_dry_run | 69 | 100% | 30.2 | 44.3 | 1.6 |
| full_system | 69 | 97% | 34.7 | 47.0 | 1.6 |
| full_plus | 69 | 100% | 24.2 | 39.8 | 1.8 |
| reasoning_only | 69 | 87% | 36.0 | 22.8 | 1.8 |
| reasoning_hybrid | 69 | 100% | 14.6 | 26.5 | 1.9 |
| reasoning_plus | 69 | 100% | 14.9 | 26.8 | 1.9 |

## 2. By kind of situation (success rate / mean unhealthy ticks)

| Situation type | do_nothing | rule_based | random_search | evolution_only | no_dry_run | full_system | full_plus | reasoning_only | reasoning_hybrid | reasoning_plus |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptation | 0% / 254 | 100% / 26 | 33% / 124 | 100% / 41 | 100% / 41 | 100% / 43 | 100% / 27 | 100% / 27 | 100% / 27 | 100% / 27 |
| combo | 0% / 255 | 100% / 12 | 75% / 63 | 100% / 40 | 100% / 40 | 100% / 43 | 100% / 28 | 100% / 9 | 100% / 9 | 100% / 9 |
| new_faults | 0% / 218 | 67% / 8 | 100% / 40 | 100% / 29 | 100% / 32 | 92% / 46 | 100% / 31 | 75% / 62 | 100% / 17 | 100% / 18 |
| noisy_sensors | 0% / 253 | 89% / 6 | 100% / 26 | 100% / 27 | 100% / 27 | 89% / 30 | 100% / 26 | 78% / 36 | 100% / 11 | 100% / 11 |
| novel | 0% / 208 | 100% / 0 | 100% / 20 | 100% / 14 | 100% / 14 | 100% / 15 | 100% / 12 | 100% / 12 | 100% / 12 | 100% / 12 |
| quiet | 100% / 0 | 33% / 0 | 56% / 0 | 56% / 0 | 56% / 0 | 56% / 0 | 56% / 0 | 56% / 0 | 56% / 0 | 56% / 0 |
| second_fault | 0% / 253 | 100% / 6 | 17% / 195 | 0% / 250 | 100% / 35 | 100% / 37 | 100% / 30 | 50% / 104 | 100% / 24 | 100% / 23 |
| single | 0% / 252 | 100% / 3 | 83% / 21 | 100% / 18 | 100% / 18 | 100% / 20 | 100% / 12 | 100% / 10 | 100% / 10 | 100% / 10 |
| unfixable | 0% / 253 | 100% / 253 | 100% / 253 | 100% / 253 | 100% / 253 | 100% / 253 | 100% / 253 | 100% / 253 | 100% / 253 | 100% / 253 |
| untuned_shop | 0% / 254 | 100% / 5 | 78% / 40 | 100% / 31 | 100% / 29 | 100% / 33 | 100% / 23 | 89% / 39 | 100% / 15 | 100% / 16 |

## 3. Every situation (success rate / mean unhealthy ticks)

| Situation | do_nothing | rule_based | random_search | evolution_only | no_dry_run | full_system | full_plus | reasoning_only | reasoning_hybrid | reasoning_plus |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adapt_combo_scale_queue_down | 0% / 258 | 100% / 30 | 33% / 72 | 100% / 51 | 100% / 51 | 100% / 53 | 100% / 18 | 100% / 28 | 100% / 28 | 100% / 28 |
| adapt_overflow_scale_queue_down | 0% / 250 | 100% / 22 | 33% / 177 | 100% / 30 | 100% / 30 | 100% / 33 | 100% / 36 | 100% / 26 | 100% / 26 | 100% / 26 |
| bigshop_dropper | 0% / 257 | 100% / 12 | 67% / 26 | 100% / 27 | 100% / 27 | 100% / 30 | 100% / 15 | 100% / 15 | 100% / 15 | 100% / 15 |
| bigshop_overflow | 0% / 250 | 100% / 1 | 100% / 42 | 100% / 31 | 100% / 27 | 100% / 32 | 100% / 35 | 100% / 15 | 100% / 15 | 100% / 15 |
| combo_corruption_deadlock | 0% / 254 | 100% / 5 | 100% / 45 | 100% / 34 | 100% / 34 | 100% / 36 | 100% / 25 | 100% / 5 | 100% / 5 | 100% / 5 |
| combo_dropper_corruption | 0% / 257 | 100% / 12 | 67% / 46 | 100% / 43 | 100% / 43 | 100% / 46 | 100% / 44 | 100% / 14 | 100% / 14 | 100% / 14 |
| combo_dropper_overflow | 0% / 255 | 100% / 14 | 33% / 114 | 100% / 45 | 100% / 45 | 100% / 48 | 100% / 18 | 100% / 13 | 100% / 13 | 100% / 13 |
| combo_overflow_deadlock | 0% / 254 | 100% / 15 | 100% / 49 | 100% / 37 | 100% / 37 | 100% / 41 | 100% / 24 | 100% / 5 | 100% / 5 | 100% / 5 |
| hardest_noisy_bigshop_cascade | 0% / 250 | 67% / 3 | 100% / 41 | 100% / 39 | 100% / 51 | 100% / 54 | 100% / 47 | 100% / 14 | 100% / 14 | 100% / 14 |
| newfault_cascade | 0% / 247 | 100% / 0 | 100% / 50 | 100% / 34 | 100% / 34 | 100% / 38 | 100% / 35 | 100% / 15 | 100% / 15 | 100% / 15 |
| newfault_flapping | 0% / 168 | 100% / 10 | 100% / 47 | 100% / 22 | 100% / 21 | 67% / 68 | 100% / 14 | 100% / 14 | 100% / 14 | 100% / 14 |
| newfault_payment_leak | 0% / 206 | 0% / 18 | 100% / 23 | 100% / 23 | 100% / 23 | 100% / 25 | 100% / 28 | 0% / 206 | 100% / 25 | 100% / 30 |
| noisy_combo_corruption_deadlock | 0% / 254 | 67% / 10 | 100% / 31 | 100% / 32 | 100% / 34 | 100% / 39 | 100% / 30 | 100% / 4 | 100% / 4 | 100% / 4 |
| noisy_single_dropper | 0% / 256 | 100% / 8 | 100% / 27 | 100% / 26 | 100% / 26 | 67% / 28 | 100% / 26 | 67% / 95 | 100% / 18 | 100% / 18 |
| noisy_single_overflow | 0% / 250 | 100% / 0 | 100% / 20 | 100% / 22 | 100% / 22 | 100% / 24 | 100% / 23 | 67% / 10 | 100% / 10 | 100% / 10 |
| novel_payment_slowdown | 0% / 208 | 100% / 0 | 100% / 20 | 100% / 14 | 100% / 14 | 100% / 15 | 100% / 12 | 100% / 12 | 100% / 12 | 100% / 12 |
| quiet_healthy_shop | 100% / 0 | 100% / 0 | 100% / 0 | 100% / 0 | 100% / 0 | 100% / 0 | 100% / 0 | 100% / 0 | 100% / 0 | 100% / 0 |
| quiet_noisy_sensors | 100% / 0 | 0% / 0 | 67% / 0 | 67% / 0 | 67% / 0 | 67% / 0 | 67% / 0 | 67% / 0 | 67% / 0 | 67% / 0 |
| quiet_very_noisy_sensors | 100% / 0 | 0% / 0 | 0% / 0 | 0% / 0 | 0% / 0 | 0% / 0 | 0% / 0 | 0% / 0 | 0% / 0 | 0% / 0 |
| second_fault_dropper_then_deadlock | 0% / 257 | 100% / 12 | 33% / 182 | 0% / 257 | 100% / 39 | 100% / 41 | 100% / 19 | 100% / 19 | 100% / 19 | 100% / 19 |
| second_fault_overflow_then_corruption | 0% / 250 | 100% / 0 | 0% / 208 | 0% / 243 | 100% / 30 | 100% / 33 | 100% / 41 | 0% / 190 | 100% / 29 | 100% / 28 |
| single_data_corruption | 0% / 248 | 100% / 0 | 67% / 23 | 100% / 12 | 100% / 12 | 100% / 14 | 100% / 4 | 100% / 4 | 100% / 4 | 100% / 4 |
| single_queue_overflow | 0% / 250 | 100% / 0 | 100% / 22 | 100% / 22 | 100% / 22 | 100% / 24 | 100% / 24 | 100% / 15 | 100% / 15 | 100% / 15 |
| single_silent_dropper | 0% / 257 | 100% / 13 | 67% / 25 | 100% / 26 | 100% / 26 | 100% / 28 | 100% / 16 | 100% / 16 | 100% / 16 | 100% / 16 |
| single_workflow_deadlock | 0% / 254 | 100% / 0 | 100% / 12 | 100% / 11 | 100% / 11 | 100% / 13 | 100% / 4 | 100% / 4 | 100% / 4 | 100% / 4 |
| smallshop_combo | 0% / 254 | 100% / 3 | 67% / 51 | 100% / 34 | 100% / 34 | 100% / 36 | 100% / 19 | 67% / 88 | 100% / 16 | 100% / 18 |
| unfixable_all_queue_tools_down | 0% / 250 | 100% / 250 | 100% / 250 | 100% / 250 | 100% / 250 | 100% / 250 | 100% / 250 | 100% / 250 | 100% / 250 | 100% / 250 |
| unfixable_dropper_returns | 0% / 255 | 100% / 255 | 100% / 255 | 100% / 255 | 100% / 255 | 100% / 255 | 100% / 255 | 100% / 255 | 100% / 255 | 100% / 255 |

For `unfixable_*`, success means the agent paged a human (nothing it can do fixes the root cause). For `quiet_*`, success means no incident was raised.

## 4. When nothing can fix it: does the agent hand over to a human, and how fast?

Cells: paged? / mean ticks until the page / mean fixes tried first.

| Situation | do_nothing | rule_based | random_search | evolution_only | no_dry_run | full_system | full_plus | reasoning_only | reasoning_hybrid | reasoning_plus |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unfixable_all_queue_tools_down | NO / - / 0.0 | yes / 59 / 6.0 | yes / 47 / 1.3 | yes / 59 / 2.0 | yes / 59 / 2.0 | yes / 22 / 0.0 | yes / 27 / 0.0 | yes / 14 / 0.0 | yes / 24 / 0.0 | yes / 29 / 0.0 |
| unfixable_dropper_returns | NO / - / 0.0 | yes / 54 / 6.0 | yes / 38 / 1.7 | yes / 29 / 1.0 | yes / 29 / 1.0 | yes / 31 / 1.0 | yes / 36 / 2.0 | yes / 26 / 1.0 | yes / 45 / 2.0 | yes / 57 / 3.0 |

## 5. Learning: the same kind of fault three times in a row

| Fault | Memory | Occurrence | Fix came from | Thinking ticks | Actions | Time to recover |
|---|---|---:|---|---:|---:|---:|
| queue_overflow | on | 1 | evolution | 11 | 1 | 22 |
| queue_overflow | on | 2 | memory | 2 | 1 | 5 |
| queue_overflow | off | 1 | evolution | 11 | 1 | 22 |
| queue_overflow | off | 2 | evolution | 3 | 1 | 7 |
| silent_dropper | on | 1 | evolution | 18 | 1 | 28 |
| silent_dropper | on | 2 | memory | 2 | 1 | 16 |
| silent_dropper | on | 3 | memory | 2 | 1 | 17 |
| silent_dropper | off | 1 | evolution | 18 | 1 | 28 |
| silent_dropper | off | 2 | evolution | 18 | 1 | 32 |
| silent_dropper | off | 3 | evolution | 16 | 1 | 29 |

## How to read this honestly

- The shop is a simulation; real systems are messier.
- Faults are clean and have known fixes in the action menu; the novel-fault row shows what happens when the fault is outside the diagnosis model's labels.
- Evolution uses randomness; seeds are fixed, so re-running reproduces these numbers.
