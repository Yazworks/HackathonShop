"""
reasoning.py - the agent's reasoning: hypotheses, predictions, checking, revising.

What this file does, in plain words
-----------------------------------
Given what the agent has recently seen (metrics, logs, and the diagnosis
model's opinion) it:

  1. ASSESSES    each possible cause: how likely is it, what supports it, what
                 argues against it, and how much of the shop's pain is left
                 UNEXPLAINED by anything it knows (knowledge.py holds the rules).
  2. DECIDES     which fix to use for each likely cause (the one it trusts most),
                 and says what the fix should change and how soon.
  3. ACTS        and then WATCHES. Did the thing it predicted happen?
        yes -> the fix and the diagnosis gain trust
        no  -> the diagnosis is doubted, it tries the next known remedy for
               that cause; if the symptoms come back it concludes it treated a
               symptom and not the cause, and hands over to evolution
  4. NOTICES     new symptoms that appear mid-fix (a second fault, a cascade)
                 and adds the matching remedy to the plan without starting over.
  5. EXPLAINS    every step in a short readable trace (shown in the console).

When the evidence does not fit anything it knows (a lot of unexplained
symptoms, or all remedies refuted) it says so and the agent falls back to
evolution, which needs no knowledge.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from evolution import HEALTHY, WAIT, Decision
from knowledge import CAUSES, RELATIONS, Remedy, View, penalty_parts

ACTIVE_P = 0.6            # a cause counts as present when its belief is at least this
CONFIDENT_P = 0.7         # needed before we trust the knowledge base to plan
UNKNOWN_MAX = 0.3         # too much unexplained pain = we are out of our depth
STREAK_NEEDED = 2         # a cause must look present on this many readings in a row
RELAPSE_AFTER = 6         # ticks after a confirmed fix before a return of symptoms counts as relapse
MAX_ACTIONS = 6


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _pct(p: float) -> str:
    return f"{int(round(p * 100))}%"


# ----------------------------------------------------------------------------
# Assessment: what does the agent currently believe?
# ----------------------------------------------------------------------------
@dataclass
class Assessment:
    p: Dict[str, float]
    support: Dict[str, List[str]]
    against: Dict[str, List[str]]
    unknown_p: float
    unexplained: List[str]
    health: float

    def active(self, threshold: float = ACTIVE_P) -> List[str]:
        return [c for c in CAUSES if self.p[c] >= threshold]

    def confident(self) -> bool:
        return bool(self.active(CONFIDENT_P)) and self.unknown_p < UNKNOWN_MAX

    def brief(self) -> Dict[str, float]:
        d = {c: round(self.p[c], 2) for c in CAUSES}
        d["unknown"] = round(self.unknown_p, 2)
        return d

    def text(self) -> str:
        parts = []
        for c in sorted(CAUSES, key=lambda c: -self.p[c]):
            if self.p[c] < 0.15 and c not in self.active():
                continue
            why = "; ".join(self.support[c][:3]) or "no supporting evidence"
            con = ("; against: " + "; ".join(self.against[c][:2])) if self.against[c] and self.p[c] < 0.9 else ""
            parts.append(f"{c.replace('_', ' ')} {_pct(self.p[c])} (for: {why}{con})")
        if self.unknown_p >= 0.15:
            parts.append(f"unexplained {_pct(self.unknown_p)} (" + "; ".join(self.unexplained) + ")")
        return " | ".join(parts) if parts else "nothing looks wrong"


# ----------------------------------------------------------------------------
# The engine: weighs evidence and keeps a track record for each remedy
# ----------------------------------------------------------------------------
class ReasoningEngine:
    def __init__(self):
        # (cause, action) -> [worked, failed]
        self.record_book: Dict[Tuple[str, str], List[int]] = {
            (c.name, r.action): [r.prior_worked, r.prior_failed]
            for c in CAUSES.values() for r in c.remedies
        }

    # --- track record of fixes -------------------------------------------
    def trust(self, cause: str, action: str) -> float:
        w, f = self.record_book.get((cause, action), [1, 1])
        return w / (w + f)

    def record(self, cause: str, action: str, worked: bool) -> None:
        book = self.record_book.setdefault((cause, action), [1, 1])
        book[0 if worked else 1] += 1

    def ranked_remedies(self, cause: str, exclude: Sequence[str] = ()) -> List[Remedy]:
        order = {r.action: i for i, r in enumerate(CAUSES[cause].remedies)}
        rs = [r for r in CAUSES[cause].remedies if r.action not in exclude]
        return sorted(rs, key=lambda r: (-self.trust(cause, r.action), order[r.action]))

    # --- the weighing of evidence -----------------------------------------
    def assess(self, view: View, probs: Optional[Dict[str, float]] = None) -> Assessment:
        score: Dict[str, float] = {}
        support: Dict[str, List[str]] = {}
        against: Dict[str, List[str]] = {}
        for name, cause in CAUSES.items():
            s = cause.prior
            sup, con = [], []
            for ev in cause.evidence:
                if ev.test(view):
                    s += ev.weight
                    (sup if ev.weight > 0 else con).append(ev.text)
            if probs is not None and name in probs:
                p = probs[name]
                s += 3.0 * (p - 0.5)
                (sup if p > 0.5 else con).append(f"the diagnosis model says {_pct(p)}")
            score[name], support[name], against[name] = s, sup, con
        first = {c: _sigmoid(s) for c, s in score.items()}
        for src, dst, w, cond, text in RELATIONS:
            if first[src] > 0.7 and cond(view):
                score[dst] += w
                support[dst].append(text)
        p = {c: _sigmoid(s) for c, s in score.items()}

        parts = penalty_parts(view.last)
        total = sum(parts.values())
        unexplained, unexpl_sum = [], 0.0
        for key, val in parts.items():
            if val < 1.0:
                continue
            explainers = [c for c in CAUSES.values() if key in c.explains]
            if not any(p[c.name] >= 0.5 for c in explainers):
                unexpl_sum += val
                unexplained.append(f"{key.replace('pen_', '')} problem nothing I know explains")
        unknown = (unexpl_sum / total) if total >= 8.0 else 0.0
        return Assessment(p, support, against, min(0.95, unknown), unexplained,
                          view.last.health_score)


# ----------------------------------------------------------------------------
# A plan the agent builds from its beliefs and then adapts as it goes
# ----------------------------------------------------------------------------
@dataclass
class Step:
    cause: str
    remedy: Remedy
    status: str = "pending"        # pending | applied | confirmed | refuted
    applied_t: int = -1
    confirmed_t: int = -1
    baseline: float = 0.0


class ReasonedPlan:
    """Has the same interface as an evolved Strategy (decide / describe /
    explain), so the agent can rehearse and run it the same way."""
    name = "reasoned"

    def __init__(self, engine: ReasoningEngine, window: Sequence, assessment: Assessment):
        self.engine = engine
        self.window: List = list(window)[-12:]
        self.steps: List[Step] = []
        self.streak: Dict[str, int] = {c: 0 for c in CAUSES}
        self.tried: Dict[str, List[str]] = {c: [] for c in CAUSES}
        self.refuted: Dict[str, set] = {c: set() for c in CAUSES}
        self.unresolved: List[str] = []
        self.actions = 0
        self.gave_up = False
        self._trace: List[Dict] = []
        self.initial = assessment
        # prime the "looks present on N readings in a row" counters with what
        # the agent has already seen
        for k in range(max(1, len(self.window) - STREAK_NEEDED + 1), len(self.window) + 1):
            a = self.engine.assess(View(self.window[:k][-12:]))
            for c in CAUSES:
                self.streak[c] = self.streak[c] + 1 if a.p[c] >= ACTIVE_P else 0

    # --- interface expected by the agent ----------------------------------
    def fresh(self) -> "ReasonedPlan":
        """A private copy for a dry run, so rehearsing does not change this plan."""
        return copy.deepcopy(self)

    def pop_trace(self) -> List[Dict]:
        out, self._trace = self._trace, []
        return out

    def describe(self) -> Dict:
        return {"approach": "reasoned from evidence",
                "causes": [c for c in CAUSES if self.initial.p[c] >= ACTIVE_P]}

    def explain(self) -> str:
        causes = ", ".join(c.replace("_", " ") for c in self.describe()["causes"]) or "unclear"
        return f"reasoned plan for: {causes} (acts at once, checks each fix against its prediction)"

    # --- helpers -----------------------------------------------------------
    def _say(self, step: str, text: str, **data) -> None:
        self._trace.append({"step": step, "text": text, **data})

    def _add_step(self, cause: str, why: str) -> None:
        # a remedy that did nothing is not repeated; one that helped may be used
        # again (up to twice), but a different known remedy is preferred first
        tried = self.tried[cause]
        rems = [r for r in self.engine.ranked_remedies(cause, exclude=self.refuted[cause])
                if tried.count(r.action) < 2]
        rems.sort(key=lambda r: tried.count(r.action))   # stable: keeps trust order within a tier
        if not rems:
            if cause not in self.unresolved:
                self.unresolved.append(cause)
                self._say("revise", f"no remedy left for {cause.replace('_', ' ')}")
            return
        r = rems[0]
        self.tried[cause].append(r.action)
        self.steps.append(Step(cause, r))
        self._say("decision",
                  f"{why}: for {cause.replace('_', ' ')} I trust {r.action} most "
                  f"({_pct(self.engine.trust(cause, r.action))} track record)",
                  cause=cause, action=r.action)

    def _give_up(self, reason: str) -> Decision:
        self.gave_up = True
        self._say("giveup", reason)
        return Decision("escalate", reason=reason)

    # --- the decision each tick ---------------------------------------------
    def decide(self, m, ctx) -> Decision:
        if self.gave_up:
            return Decision("escalate", reason="reasoning has given up")
        t = ctx.t
        self.window.append(m)
        self.window = self.window[-12:]
        a = self.engine.assess(View(self.window))
        for c in CAUSES:
            self.streak[c] = self.streak[c] + 1 if a.p[c] >= ACTIVE_P else 0
        active = [c for c in CAUSES if self.streak[c] >= STREAK_NEEDED]

        # 1. check the predictions of the fixes already applied
        for s in self.steps:
            if s.status != "applied":
                continue
            current = getattr(m, s.remedy.signal)
            if s.remedy.satisfied(current, s.baseline):
                s.status, s.confirmed_t = "confirmed", t
                self.engine.record(s.cause, s.remedy.action, True)
                self._say("verdict",
                          f"{s.remedy.action} worked as predicted: {s.remedy.signal} went from "
                          f"{s.baseline:.2f} to {current:.2f}. Trust in this diagnosis and fix goes up.",
                          cause=s.cause, action=s.remedy.action, worked=True)
            elif t - s.applied_t >= s.remedy.settle:
                s.status = "refuted"
                self.refuted[s.cause].add(s.remedy.action)
                self.engine.record(s.cause, s.remedy.action, False)
                self._say("verdict",
                          f"{s.remedy.action} did NOT do what I predicted ({s.remedy.signal} is still "
                          f"{current:.2f} after {t - s.applied_t} ticks). Either the fix is not working "
                          f"or I misjudged the cause.",
                          cause=s.cause, action=s.remedy.action, worked=False)
                if s.cause in active or self.initial.p[s.cause] >= ACTIVE_P:
                    self._add_step(s.cause, "trying the next known remedy")

        # 2. symptoms that came back after a fix had worked = treated a symptom
        for s in self.steps:
            if (s.status == "confirmed" and s.cause in active and m.health_score < HEALTHY
                    and t - s.confirmed_t >= RELAPSE_AFTER):
                s.status = "relapsed"
                self.engine.record(s.cause, s.remedy.action, False)
                self._say("revise",
                          f"{s.cause.replace('_', ' ')} is still showing after {s.remedy.action} "
                          f"seemed to work: it was only part of the answer, or a symptom treated "
                          f"instead of the root cause",
                          cause=s.cause, action=s.remedy.action)
                before = len(self.steps)
                self._add_step(s.cause, "adding more help")
                if len(self.steps) == before:
                    return self._give_up(
                        f"{s.cause.replace('_', ' ')} keeps coming back and I have no remedy "
                        f"left to add, so I treated a symptom, not the root cause")

        # 3. new or newly visible problems get their own remedy (second faults,
        #    cascades, symptoms that were slow to show)
        started = {s.cause for s in self.steps} | set(self.unresolved)
        for c in active:
            if c not in started:
                self._add_step(c, "evidence now supports " + c.replace("_", " ")
                               + f" ({_pct(a.p[c])})")

        if m.health_score >= HEALTHY:
            return WAIT
        if self.actions >= MAX_ACTIONS:
            return self._give_up(f"used {MAX_ACTIONS} actions and the shop is still unhealthy")

        # 4. act on the next pending step
        for s in self.steps:
            if s.status == "pending":
                s.status, s.applied_t = "applied", t
                s.baseline = getattr(m, s.remedy.signal)
                self.actions += 1
                self._say("prediction", f"applying {s.remedy.action}. Prediction: "
                          f"{s.remedy.expectation_text()}.", cause=s.cause, action=s.remedy.action)
                return Decision("act", s.remedy.action, 1.0,
                                reason=f"reasoned: {s.cause.replace('_', ' ')}")

        # 5. nothing to do right now
        if any(s.status == "applied" for s in self.steps):
            return WAIT                                  # waiting for a fix to show its effect
        if not active:
            if t >= 6:
                return self._give_up("the shop is unhealthy but no cause I know of fits the "
                                     "evidence; handing over to search")
            return WAIT
        if self.unresolved and all(c in self.unresolved or
                                   any(s.cause == c and s.status in ("refuted", "confirmed")
                                       for s in self.steps) for c in active):
            return self._give_up("every remedy I know for the active problem has been tried "
                                 "without lasting success")
        return WAIT
