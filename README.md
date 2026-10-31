# HackathonShop

An agent that notices when a shop is failing, works out why, and fixes it, or admits it can't and calls a human.

This is my submission for Challenge 1, Agentic Recovery of a Failing System. It is an individual project.

## Overview

The challenge asks for a system that can identify a failure, diagnose it, recover from it and verify the recovery. To build and test that, I wrote a small simulated online shop and a way to break it on purpose. The shop takes orders, pushes them through a queue, charges payments, runs a data-sync job and a workflow engine, and reports a single health score from 0 to 100. Anything at 80 or above counts as healthy.

The agent watches that score and the shop's logs. When something goes wrong it decides what the cause is, tries a fix on a copy of the shop first, applies it to the real one, and waits to see whether the shop stays healthy. If it can't find a fix that works, it pages a human and stops trying, which is the behaviour the fifth problem in the challenge is about.

On 100 test cases written by a separate Claude session that never saw my agent, the agent passed 89% (95% interval 81 to 94%). It paged a human on all 26 cases where the root cause could not be removed.

## The five problems

The challenge describes five failures. Each one maps to a fault I can inject into the shop.

| Challenge problem | Fault in the shop | What the agent can do about it |
|---|---|---|
| Payments silently dropping orders | Silent order dropper | Restart the payment service |
| A queue losing messages under load | Queue overflow | Scale the queue or throttle traffic |
| A sync job slowly corrupting data | Data corruption | Roll the sync job back |
| A workflow that hangs forever | Frozen workflow | Kill the stuck job |
| A repair agent that never escalates | Faults that can't be fixed | Nothing, so it has to page a human |

I also added harder variations that the challenge didn't spell out: a slow payment leak, faults that flap on and off, a traffic spike that then corrupts data, noisy or stale sensors, missing logs, broken tools, and shops of different sizes.

## How the agent works

The agent runs one loop: watch the shop, declare an incident, plan a fix, rehearse it, apply it, verify it, and then either learn from it or escalate.

It watches by reading the metrics and logs every tick. When health stays under 80 for two ticks in a row it declares an incident. A trained gradient boosting model, which looks at thirteen features taken from the metrics and logs, proposes a diagnosis, and a diagnosis has to repeat three times in a row before the agent acts on it, so one noisy reading can't trigger a fix.

Once it has a diagnosis it looks for a plan in three places, in order, and each gets one chance per incident. The first is a reasoning layer built on a knowledge base I wrote by hand, much like an on-call runbook. It weighs the evidence for each possible cause, picks the remedy with the best track record, says in advance what that remedy should change, and checks afterwards whether that happened. If the prediction fails, the remedy is dropped and the agent moves on. The second is memory: fixes saved from earlier incidents, which are only trusted once they have worked often enough. The third is evolutionary search, used when the first two have nothing. It starts with a population of 40 candidate strategies, scores each one in simulation on how quickly it recovers, how few actions it takes and whether it removes the root cause, and then breeds and mutates the best for up to 20 generations.

Whatever the plan, it is first run for two ticks on a staging copy of the shop. It has to recover and remove the root cause there before it touches the real shop. After applying it, the agent needs five healthy ticks in a row to call the incident over. A fix that evolution discovered can be saved for next time. If every source of plans fails, the agent pages a human and stops.

I think of the whole thing as a hybrid. The detector is a supervised model trained on simulated incidents, the reasoning layer is an expert system, the memory is case-based, and evolution is the fallback for situations nobody planned for. I wrote the knowledge base myself, so I don't claim it is learned.

## Results

Because the simulator and my own scenarios are both mine, I didn't want to rely on them alone. I tested at three levels.

### My own evaluation

This covers 28 situations (23 with a fix, 3 quiet and 2 unfixable), run at harsher settings than the ones I tuned on, with three random seeds each. The 69 episodes per agent in the table below are the 23 situations with a fix, run three times. Every agent starts with an empty memory. The second column is the share of episodes fixed, and the third is the average number of ticks the shop spent unhealthy, so lower is better.

| Agent | Fixed | Unhealthy ticks |
|---|---:|---:|
| Do nothing | 0% | 245.4 |
| Simple rule-based script | 93% | 8.2 |
| Random search, same budget as evolution | 77% | 58.8 |
| Evolution only | 91% | 48.5 |
| Reasoning only | 87% | 36.0 |
| Earlier version, without the reasoning layer | 97% | 34.7 |
| Submitted agent | 100% | 14.6 |

The rule-based script is faster whenever a fault matches one of its rules, which is why it posts the best time here. It does much worse on the independent cases below.

### Independent test cases

I asked a separate Claude session to write an adversarial test suite. It was given the challenge text and the simulator, and nothing else: it never saw my agent, my knowledge base or my tests. An omniscient oracle then checks that the suite is fair. It must solve every case labelled fixable, must be unable to remove the root cause in any case labelled unfixable, and must find that the quiet cases need no action at all.

I ran one set of cases before the final run, and a second, sealed set for the final result. The development set has 60 hand-written cases and 60 random ones (119 were scored), and the submitted agent passed 90% of them (95% interval 83 to 94%). The final set is 100 random cases generated from a seed I hadn't looked at, and I ran it once, after the agent was frozen.

| Agent | Final set pass rate | 95% interval |
|---|---:|---:|
| Do nothing | 21% | 14 to 30% |
| Rule-based script | 63% | 53 to 72% |
| Reasoning only | 71% | 61 to 79% |
| Evolution only | 74% | 65 to 82% |
| Earlier version, without the reasoning layer | 77% | 68 to 84% |
| Submitted agent plus starter playbooks | 84% | 76 to 90% |
| Submitted agent | 89% | 81 to 94% |

Broken down by the kind of case, the submitted agent passed 48 of 53 fixable cases (91%), all 26 unfixable cases (100%, each one ending in a page to a human), and 15 of 21 quiet cases (71%), where the right answer is to do nothing. That last number is the weakest, and I come back to it under limitations.

### What each part adds

No single part of the agent does the job alone. On the independent final set, evolution on its own fixed 74% of cases and the reasoning layer on its own 71%, while the combination reached 89%. Evolution is also doing real work and not just guessing: given the same evaluation budget as random search, it fixed 91% of my own cases against 77%.

Memory pays off on repeats. In a learning experiment, a repeated silent-dropper fault needed 2 thinking ticks once the agent remembered the fix, against 16 to 18 without memory. For queue overflow the saving was small, 2 ticks against 3.

The staging rehearsal didn't show a measurable gain. The version without it also fixed 100% of my own cases, in 30.2 unhealthy ticks on average. I keep it as a safety check, but I can't claim it improved the numbers.

## Limitations

All of this runs in my own simulator. The independent cases were written without knowledge of my agent, but they still use my simulated shop, so they show robustness to someone else's scenarios and not to a real production system.

The agent misses very mild faults. If a problem never pushes health below 80, for example when a small fraction of orders is lost, no incident is ever declared. On the development set, cases that start mildly passed only 57% of the time.

It also raises false alarms. On the quiet cases in the final set, it stayed quiet in 15 of 21. The six it got wrong came from heavy sensor noise and from sustained high-but-sustainable traffic, which looks a lot like an overload.

On the final set it failed 5 of 53 fixable cases, and the payment leak, the one fault family with no rule in my knowledge base, is the least well covered, though the agent still passed 84% of those cases.

The knowledge base is written by hand, not learned. Evolution fills the gaps, but what it discovers is not folded back into the knowledge base, and the track record of each remedy is not saved between runs. Verification only looks at the health score, so a fault that switches itself off for a while could look like a recovery. And evolution is slow in real time: a hard fault can take roughly 12 to 33 seconds on a laptop.

With more time I would add a trend-based trigger so slow, mild faults are caught before health drops, feed evolved fixes back into the knowledge base, save remedy track records between runs, tell heavy load apart from real overload more carefully, and write an adapter so the agent could read telemetry from a live system.

## Running it

I tested everything on Windows with Python 3.11. From the project folder in PowerShell:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python detector.py
python main.py
```

`detector.py` trains the diagnosis model. A trained copy is already in `models/`, so you can skip it, but it is quick to rebuild. `main.py` starts the console at http://127.0.0.1:8000 and opens it in your browser.

In the console, the buttons under "Break the shop" inject a fault and the page shows what the agent thinks and does in plain language, along with the health line and its decision log. The payment leak is a good one to try, since it has no rule in the knowledge base. Turning on "root cause keeps returning" shows the escalation path: the agent tries, fails, and pages a human instead of looping. `python main.py --keep-memory` keeps the fixes learned in earlier runs, and by default memory is wiped at startup.

One thing to know: the console runs the agent with ten pre-trained starter playbooks loaded and with immediate first action switched on. That is the configuration called "plus" in the results, which scored 84% on the final set. The 89% headline number is from the same agent without those two additions.

Everything in the results can be reproduced, and the seeds are fixed so re-running gives the same numbers.

| Command | What it does |
|---|---|
| `python shop.py` | Self-test of the simulator |
| `python recovery_agent.py demo` | A short scripted demo of the agent in the terminal |
| `python evaluate.py quick` | My own evaluation with one seed |
| `python evaluate.py full` | My own evaluation with three seeds, plus charts (takes a while) |
| `python independent_cases.py` | Checks the independent cases against the oracle |
| `python run_independent.py dev` | Independent development set |
| `python run_independent.py final` | Independent sealed final set, which should be run only once |
| `python make_starter.py` | Regenerates the starter playbooks |

Tables, raw data and charts are written to the `results/` folder.

## Project layout

`shop.py` is the simulated shop, its faults, the health score and the sensor noise. `detector.py` trains the diagnosis model. `knowledge.py` holds the hand-written runbook of causes, evidence and remedies, and `reasoning.py` weighs that evidence, predicts what a fix should change and checks afterwards. `playbooks.py` is the memory, `evolution.py` is the evolutionary search and its fitness function, and `recovery_agent.py` is the loop that ties them together. `make_starter.py` builds the starter playbooks. `server.py` and `main.py` are the console's backend and launcher, and `static/` holds the page with its bundled chart library and fonts. `evaluate.py` runs my own evaluation, `independent_cases.py` is the independently written test suite (unchanged from how I received it), and `run_independent.py` runs the agents against it. The trained model lives in `models/`, the starter playbooks in `memory/`, and all results in `results/`.

## Notes

The independent test cases were written by Claude in a separate session, as described above. I didn't change the agent after seeing the development results. The only later code change was a fix to how thinking time is counted for the random-search comparison, which has no effect on any other agent.

Chart.js (MIT licence) and the fonts Instrument Serif, Figtree and JetBrains Mono (SIL Open Font Licence) are bundled in `static/` so the console works offline.
