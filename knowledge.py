"""
knowledge.py - what the agent KNOWS about how the shop fails and how to fix it.

What this file does, in plain words
-----------------------------------
This is the agent's runbook, written as data instead of prose. For every kind
of problem it records:

  * EVIDENCE   - which symptoms point to this cause (and which symptoms, when
                 absent, argue against it). Each piece of evidence nudges the
                 agent's belief up or down by a weight.
  * REMEDIES   - which actions are known to fix it, in order of preference, and
                 what each action is EXPECTED to change and how quickly. These
                 expectations are what the agent later checks its fix against.
  * RELATIONS  - causes that can set off other causes (a long queue overload
                 can start corrupting data).

This is human expertise encoded for the agent, the same as a real on-call
runbook. It is written by us (the authors); what the agent adds on top is
weighing the evidence, predicting, checking, and revising (reasoning.py).

The numbers (weights) are log-odds: +3 is strong support, -3 strong
evidence against. A belief is  sigmoid(prior + sum of weights that apply),
so two moderate clues can outweigh one weak one, and a missing symptom can
count against a cause.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence


# ----------------------------------------------------------------------------
# A view over the last few readings (what the agent has seen recently)
# ----------------------------------------------------------------------------
class View:
    def __init__(self, snapshots: Sequence):
        self.s = list(snapshots)
        self.last = self.s[-1]

    def mean(self, name: str, n: int = 3) -> float:
        vals = [getattr(x, name) for x in self.s[-n:]]
        return sum(vals) / len(vals)

    def slope(self, name: str, n: int = 6) -> float:
        """average change per reading over the last n readings"""
        vals = [getattr(x, name) for x in self.s[-n:]]
        return (vals[-1] - vals[0]) / max(1, len(vals) - 1)


# ----------------------------------------------------------------------------
# Evidence, remedies, causes
# ----------------------------------------------------------------------------
@dataclass
class Evidence:
    text: str                          # how it reads in the explanation
    weight: float                      # log-odds added when the test is true
    test: Callable[[View], bool]


@dataclass
class Remedy:
    action: str
    signal: str                        # what should change if this fix works
    direction: str                     # "down" (signal should fall) or "up"
    threshold: float                   # absolute level that counts as fixed
    fraction: Optional[float]          # ... or: down to this fraction of the level when we acted
    settle: int                        # ticks allowed before we call it not working
    prior_worked: int = 3              # starting track record (successes, failures)
    prior_failed: int = 1

    def expectation_text(self) -> str:
        if self.direction == "down":
            if self.threshold > 0:
                return f"{self.signal} should fall below {self.threshold:g} within {self.settle} ticks"
            return (f"{self.signal} should fall to {int((self.fraction or 0) * 100)}% of its "
                    f"current level within {self.settle} ticks")
        return f"{self.signal} should rise above {self.threshold:g} within {self.settle} ticks"

    def satisfied(self, current: float, baseline: float) -> bool:
        if self.direction == "down":
            if current <= self.threshold:
                return True
            return self.fraction is not None and current <= self.fraction * baseline
        return current >= self.threshold


@dataclass
class Cause:
    name: str
    label: str
    prior: float                       # starting log-odds (about 12% when -2)
    evidence: List[Evidence]
    remedies: List[Remedy]
    explains: List[str] = field(default_factory=list)   # health-penalty parts this cause explains


CAUSES: Dict[str, Cause] = {
    "silent_dropper": Cause(
        "silent_dropper", "payment service silently dropping orders", -2.0,
        evidence=[
            Evidence("orders are vanishing between the queue and the ledger",
                     1.5, lambda v: v.mean("mismatch_rate") > 0.015),
            Evidence("a large share of orders is lost in payments",
                     3.0, lambda v: v.mean("mismatch_rate") > 0.04),
            Evidence("payment warnings 'acked but no ledger write' in the logs",
                     1.5, lambda v: v.last.log_payment_warn > 0),
            Evidence("no orders are being lost in payments",
                     -2.5, lambda v: v.mean("mismatch_rate") < 0.008),
        ],
        remedies=[Remedy("restart_payments", "mismatch_rate", "down", 0.02, 0.5, 16)],
        explains=["pen_mismatch"],
    ),
    "queue_overflow": Cause(
        "queue_overflow", "queue overloaded and losing messages", -2.0,
        evidence=[
            Evidence("the queue is more than 60% full", 1.0, lambda v: v.last.queue_saturation > 0.6),
            Evidence("the queue is almost full", 2.5, lambda v: v.last.queue_saturation > 0.85),
            Evidence("the queue is dropping messages", 2.0, lambda v: v.mean("drop_rate") > 0.03),
            Evidence("queue errors 'buffer full' in the logs", 1.5, lambda v: v.last.log_queue_error > 0),
            Evidence("the queue is filling up", 0.5, lambda v: v.slope("queue_saturation") > 0.02),
            Evidence("the queue is nearly empty and nothing is dropped", -3.0,
                     lambda v: v.last.queue_saturation < 0.4 and v.mean("drop_rate") < 0.01),
        ],
        remedies=[Remedy("scale_queue", "queue_saturation", "down", 0.6, 0.6, 8),
                  Remedy("throttle_traffic", "orders_per_second", "down", 0.0, 0.7, 14, 2, 1)],
        explains=["pen_drop", "pen_sat"],
    ),
    "data_corruption": Cause(
        "data_corruption", "sync job corrupting records", -2.0,
        evidence=[
            Evidence("some synced records are corrupted", 3.0, lambda v: v.last.integrity_score < 0.97),
            Evidence("many synced records are corrupted", 1.0, lambda v: v.last.integrity_score < 0.9),
            Evidence("checksum warnings from the sync job", 1.5, lambda v: v.last.log_sync_warn > 0),
            Evidence("all synced records are clean", -3.0, lambda v: v.last.integrity_score >= 0.99),
        ],
        remedies=[Remedy("rollback_sync", "integrity_score", "up", 0.97, None, 6)],
        explains=["pen_corr"],
    ),
    "workflow_deadlock": Cause(
        "workflow_deadlock", "workflow job frozen and blocking the pipeline", -2.0,
        evidence=[
            Evidence("a job is stuck", 2.0, lambda v: v.last.jobs_stuck > 0),
            Evidence("it has been stuck for a while", 2.0, lambda v: v.last.stuck_duration >= 5),
            Evidence("workflow errors 'holds lock' in the logs", 1.5, lambda v: v.last.log_workflow_error > 0),
            Evidence("no job is stuck", -4.0, lambda v: v.last.jobs_stuck == 0),
        ],
        remedies=[Remedy("kill_stuck_job", "jobs_stuck", "down", 0.5, None, 4)],
        explains=["pen_stuck"],
    ),
}

# Causes that can trigger others: (cause, effect, extra log-odds for the effect,
# condition on the view, explanation)
RELATIONS = [
    ("queue_overflow", "data_corruption", 1.0,
     lambda v: v.last.queue_saturation > 0.8,
     "a long queue overload can start corrupting the sync job"),
]

# how the health penalty is made up (same formula the shop uses)
def penalty_parts(m) -> Dict[str, float]:
    return {
        "pen_mismatch": min(40.0, m.mismatch_rate * 200),
        "pen_drop": min(35.0, m.drop_rate * 100),
        "pen_sat": max(0.0, m.queue_saturation - 0.8) * 50,
        "pen_corr": min(30.0, (1.0 - m.integrity_score) * 600),
        "pen_stuck": min(30.0, m.stuck_duration * 3),
    }
