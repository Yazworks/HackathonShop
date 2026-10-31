"""
playbooks.py - the playbook library: fixes the agent has learned and saved.

When the agent recovers a shop using a freshly evolved strategy, the strategy
is saved here as a "playbook" for that kind of incident. Next time the same
kind of incident happens the agent can use the playbook straight away
instead of evolving a new fix from scratch.

Every playbook tracks how often it was used and how often it worked. A
playbook that stops working is no longer trusted (see PlaybookLibrary.find).

Playbooks are keyed by the diagnosis, for example "silent_dropper" or
"queue_overflow+silent_dropper". Playbooks for single problems are also used
as head starts when evolving a fix for a combination that includes them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from evolution import Strategy

PLAYBOOK_PATH = "memory/playbooks.json"
STARTER_PATH = "memory/starter_playbooks.json"   # shipped with the project; see make_starter.py


@dataclass
class Playbook:
    id: str
    key: str                       # diagnosis, e.g. "silent_dropper"
    genes: List[float]             # the winning recovery strategy
    fitness: float                 # score it earned when it was evolved
    generations: int               # how many generations evolution needed
    times_used: int = 1
    successes: int = 1
    created_tick: int = 0
    origin: str = "learned"        # "starter" = trained offline and shipped with the project

    @property
    def success_rate(self) -> float:
        return self.successes / self.times_used if self.times_used else 0.0

    def strategy(self) -> Strategy:
        return Strategy.from_array(self.genes)

    def row(self) -> Dict:
        """Compact form for a dashboard table."""
        return {"id": self.id, "key": self.key, "origin": self.origin,
                "fitness": round(self.fitness, 1),
                "generations": self.generations, "times_used": self.times_used,
                "success_rate": round(self.success_rate, 2)}


class PlaybookLibrary:
    def __init__(self, path: Optional[str] = None, starter_path: Optional[str] = None):
        """path=None keeps everything in memory only (used for tests).
        starter_path: playbooks trained offline; they are loaded but never
        written back to the learned-playbooks file."""
        self.path = Path(path) if path else None
        self.items: List[Playbook] = []
        if starter_path and Path(starter_path).exists():
            for d in json.loads(Path(starter_path).read_text()):
                d["origin"] = "starter"
                self.items.append(Playbook(**d))
        if self.path and self.path.exists():
            for d in json.loads(self.path.read_text()):
                self.items.append(Playbook(**d))

    # ---- lookup -----------------------------------------------------------
    def find(self, key: str, min_rate: float = 0.6) -> Optional[Playbook]:
        """Best trusted playbook for this exact diagnosis, or None."""
        ok = [p for p in self.items if p.key == key and p.success_rate >= min_rate]
        if not ok:
            return None
        return max(ok, key=lambda p: (p.success_rate, p.fitness))

    def seed_strategies(self, causes: List[str]) -> List[Strategy]:
        """Strategies for the individual problems inside a combination. They
        give evolution a head start when it searches for a combined fix."""
        want = set(causes)
        out: List[Strategy] = []
        for p in self.items:
            part = set(p.key.split("+"))
            if p.key != "unknown" and part and part < want:      # proper subset
                out.append(p.strategy())
        return out

    # ---- updates ----------------------------------------------------------
    def add(self, key: str, strategy: Strategy, fitness: float,
            generations: int, tick: int = 0) -> Playbook:
        n = 1 + max((int(p.id.split("_")[1]) for p in self.items
                     if p.id.startswith("pb_")), default=0)
        pb = Playbook(id=f"pb_{n:03d}", key=key,
                      genes=[round(g, 6) for g in strategy.to_array()],
                      fitness=float(fitness), generations=int(generations),
                      created_tick=int(tick))
        self.items.append(pb)
        self._save()
        return pb

    def record_use(self, playbook_id: str, succeeded: bool) -> None:
        for p in self.items:
            if p.id == playbook_id:
                p.times_used += 1
                p.successes += 1 if succeeded else 0
                self._save()
                return

    def rows(self) -> List[Dict]:
        return [p.row() for p in self.items]

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            [asdict(p) for p in self.items if p.origin != "starter"], indent=2))
