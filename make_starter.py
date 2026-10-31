"""
make_starter.py - train the starter playbooks (run once, takes a few minutes).

What this does, in plain words
------------------------------
The agent normally learns a playbook only AFTER it has suffered an incident.
Here we let evolution practise offline, on simulated incidents of every kind
(the single faults and every pair of them), and save the winners to
memory/starter_playbooks.json. A shop that starts with these playbooks can use
a remembered fix on its very first incident instead of thinking from scratch.

To keep this honest:
  * practice uses the MILD "training" fault settings only, never the harsher
    settings used in evaluate.py;
  * practice shops use their own seeds, different from every evaluation shop;
  * each playbook is re-checked on 3 fresh practice shops before it is kept,
    and its pass count is stored as its starting track record.

Run:  python make_starter.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from detector import key_of
from evolution import (EvolutionEngine, TRAIN_PARAMS, dry_run, make_visibly_sick)
from playbooks import PlaybookLibrary, STARTER_PATH
from shop import FAULTS, Shop, run_shop

SINGLES = ["silent_dropper", "queue_overflow", "data_corruption", "workflow_deadlock"]


def practice_shop(faults, seed: int) -> Shop:
    shop = Shop(seed=seed)
    run_shop(shop, 50)
    for f in faults:
        FAULTS[f](shop, **TRAIN_PARAMS[f])
    run_shop(shop, 40)
    make_visibly_sick(shop)
    return shop


def main() -> None:
    t0 = time.perf_counter()
    groups = [(f,) for f in SINGLES] + list(itertools.combinations(SINGLES, 2))
    lib = PlaybookLibrary(None)                    # in-memory while building
    kept = []
    for n, faults in enumerate(groups, 1):
        key = key_of(faults)
        seeds = lib.seed_strategies(list(faults))
        res = EvolutionEngine(seed=100 + n).run(practice_shop(faults, 300 + n), initial=seeds)
        passes = sum(dry_run(res.best, practice_shop(faults, 600 + 10 * n + k)).passed
                     for k in range(3))
        verdict = "kept" if passes >= 2 else "DROPPED"
        print(f"  {key:<38} fitness {res.best_fitness:5.1f}  re-check {passes}/3  {verdict}", flush=True)
        if passes >= 2:
            pb = lib.add(key, res.best, res.best_fitness, res.generations)
            pb.times_used, pb.successes, pb.origin = 3, passes, "starter"
            pb.id = f"st_{len(kept) + 1:03d}"
            kept.append(pb)
    Path(STARTER_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(STARTER_PATH).write_text(json.dumps([asdict(p) for p in kept], indent=2))
    print(f"\nSaved {len(kept)} starter playbooks to {STARTER_PATH} "
          f"({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
