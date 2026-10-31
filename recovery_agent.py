"""
recovery_agent.py - the self-healing loop for HackathonShop.

What this file does, in plain words
-----------------------------------
This is the agent. It watches the shop, and when something goes wrong it:

  1. DIAGNOSES   which problem(s) are happening (detector.py: metrics + logs)
  2. CHECKS      the playbook library for a fix it has already learned
  3. EVOLVES     a new fix on staging copies if there is none (evolution.py)
  4. DRY-RUNS    the fix on a fresh staging copy before touching the real shop
  5. APPLIES     the fix to the real shop and keeps watching
  6. VERIFIES    the shop is really healthy again (80+ for 5 ticks in a row)
  7. LEARNS      saves the fix as a playbook for next time (playbooks.py)
  8. ESCALATES   pages a human when it cannot fix the problem safely

It also copes with surprises: if a NEW problem appears while it is mid-fix it
re-diagnoses and re-plans; if a saved playbook stops working it falls back to
evolving a fresh fix; if the detector does not recognise the problem it still
evolves a fix (evolution does not need to know the cause).

Time is simulated. Evolving a fix costs 1 tick per generation (the incident
keeps running while the agent thinks) and a dry run costs 2 ticks. Using a saved
playbook skips the evolution cost, which is why remembered fixes are faster.

Run:  python recovery_agent.py demo      (needs: python detector.py first)
"""
from __future__ import annotations

import sys
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

from detector import Detector, DiagnosisStream, key_of, load_detector
from evolution import (
    HEALTHY,
    MAX_TICKS,
    STABLE_TICKS,
    AgentContext,
    EvolutionEngine,
    Strategy,
    dry_run,
    symptom_severity,
)
from knowledge import View
from playbooks import PlaybookLibrary
from reasoning import ACTIVE_P, ReasonedPlan, ReasoningEngine
from shop import FAULTS, MetricsSnapshot, Shop, apply_action, collect_metrics, run_shop

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
TICKS_PER_GENERATION = 1     # thinking time: one generation of evolution = 1 tick
DRY_RUN_TICKS = 2            # a dry run on a staging copy takes 2 ticks
DECLARE_STREAK = 2           # unhealthy this many ticks in a row = incident
UNEXPLAINED_PATIENCE = 8     # reasoning mode: wait this long on a mild dip nothing explains
UNDIAGNOSED_TICKS = 6        # wait this long for a diagnosis, then treat as unknown
EXEC_TIMEOUT = MAX_TICKS     # give a fix this long before calling it failed
MEMORY_MIN_RATE = 0.6        # only trust playbooks that worked this often
PAGED_CLEAR_STREAK = 10      # if the shop heals itself while paged, resume

MONITOR, EXECUTE, PAGED = "MONITOR", "EXECUTE", "PAGED"


def _act_now(strategy: Strategy) -> Strategy:
    """Same fix, but no waiting around: an incident has already been declared
    (the shop has been unhealthy for a while), so start straight away."""
    g = strategy.to_array()
    g[0] = 0.0        # watch only 1 tick
    g[1] = 1.0        # act whenever health is below the healthy line (80)
    return Strategy.from_array(g)


class RecoveryAgent:
    def __init__(
        self,
        shop: Shop,
        detector: Optional[Detector] = None,
        library: Optional[PlaybookLibrary] = None,
        *,
        use_memory: bool = True,
        use_dry_run: bool = True,
        act_immediately: bool = False,
        use_reasoning: bool = False,
        reasoning_first: bool = True,
        use_evolution: bool = True,
        seed: int = 0,
        evolution_kwargs: Optional[Dict] = None,
        emit: Optional[Callable[[Dict], None]] = None,
        record_events: bool = False,
        before_tick: Optional[Callable[[Shop], None]] = None,
    ):
        self.shop = shop
        self.detector = detector
        self.stream = DiagnosisStream(detector) if detector else None
        self.library = library if library is not None else PlaybookLibrary(None)
        self.use_memory = use_memory
        self.use_dry_run = use_dry_run
        self.act_immediately = act_immediately   # skip a strategy's own 'wait and watch' once an incident is declared
        self.use_reasoning = use_reasoning       # plan from knowledge (reasoning.py) before searching
        self.reasoning_first = reasoning_first   # try reasoning before the playbook memory
        self.use_evolution = use_evolution       # False = never evolve (used to test reasoning alone)
        self.engine = ReasoningEngine()          # keeps the track record of each remedy
        self.recent: Deque = deque(maxlen=14)    # the last readings the agent has seen
        self._last_probs: Optional[Dict[str, float]] = None
        self.assessment = None
        self.seed = seed
        self.evolution_kwargs = evolution_kwargs or {}
        self.emit = emit
        self.record_events = record_events
        self.before_tick = before_tick          # e.g. to keep re-injecting a fault

        self.state = MONITOR
        self.healthy_streak = 0
        self.unhealthy_streak = 0
        self.incidents: List[Dict] = []         # one record per incident
        self.inc: Optional[Dict] = None         # the incident being handled
        self.events: Deque[Dict] = deque(maxlen=3000)
        self.health_trace: Deque[float] = deque(maxlen=500)

        self._policy: Optional[Strategy] = None
        self._saved_strategy: Optional[Strategy] = None
        self._ctx: Optional[AgentContext] = None
        self._exec_ticks = 0
        self._source = ""
        self._playbook = None
        self._plan_fitness = 0.0
        self._plan_generations = 0
        self._last_diag_key = "healthy"

    # ------------------------------------------------------------------ events
    def _ev(self, type_: str, record: bool = True, **data) -> None:
        if self.emit is None and not (self.record_events and record):
            return
        e = {"tick": self.shop.tick, "type": type_, "data": data}
        if self.record_events and record:
            self.events.append(e)
        if self.emit is not None:
            self.emit(e)

    # ------------------------------------------------------------ simulation
    def _tick_shop(self):
        """Advance the real shop by one tick and observe it."""
        if self.before_tick is not None:
            self.before_tick(self.shop)
        run_shop(self.shop, 1)
        m = collect_metrics(self.shop)
        d = None
        if self.stream is not None:
            d = self.stream.update(m)
            if d.key != self._last_diag_key:
                self._last_diag_key = d.key
                self._ev("diagnosis", causes=d.causes, probs=d.probs,
                         confidence=round(d.confidence, 3), unsure=d.unsure)
        self.recent.append(m)
        if d is not None:
            self._last_probs = dict(d.probs)
        if self.use_reasoning:
            self.assessment = self.engine.assess(View(self.recent), self._last_probs)
        if m.health_score >= HEALTHY:
            self.healthy_streak += 1
            self.unhealthy_streak = 0
        else:
            self.healthy_streak = 0
            self.unhealthy_streak += 1
        self.health_trace.append(m.health_score)
        if self.emit is not None:
            self._ev("metrics", record=False, snapshot=m.to_dict(),
                     diagnosis=(d.causes if d else []),
                     beliefs=(self.assessment.brief() if self.assessment else None))
        return m, d

    def _advance(self, ticks: int) -> None:
        """Let time pass (the agent is thinking); the incident keeps running."""
        for _ in range(ticks):
            self._tick_shop()

    # ------------------------------------------------------------- main step
    def tick(self) -> None:
        m, d = self._tick_shop()
        if self.state == MONITOR:
            if self.unhealthy_streak >= DECLARE_STREAK:
                self._declare(m, d)
        elif self.state == EXECUTE:
            self._execute_step(m)
        elif self.state == PAGED:
            if self.healthy_streak >= PAGED_CLEAR_STREAK:
                self._ev("auto_cleared", reason="shop healthy again without intervention")
                self.resolve()

    def run(self, ticks: int) -> None:
        end = self.shop.tick + ticks
        while self.shop.tick < end:
            self.tick()

    def run_until_idle(self, max_ticks: int = 400, min_ticks: int = 15) -> None:
        """Run until the current incident is closed (or a human is paged)."""
        start = self.shop.tick
        while self.shop.tick - start < max_ticks:
            self.tick()
            if self.shop.tick - start >= min_ticks and (
                    self.state == PAGED or (self.state == MONITOR and self.inc is None
                                            and self.healthy_streak >= STABLE_TICKS)):
                break

    def resolve(self) -> None:
        """A human has fixed things: go back to normal monitoring."""
        self.state = MONITOR
        self.inc = None
        self._policy = None
        if self.stream is not None:
            self.stream.reset()

    # --------------------------------------------------------- incident flow
    def _current_causes(self, d) -> List[str]:
        if self.stream is None:
            return []
        conf = self.stream.confirmed()
        return list(conf) if conf else (list(d.causes) if d else [])

    def _declare(self, m: MetricsSnapshot, d) -> None:
        # With reasoning on, a mild dip that NONE of my knowledge explains is
        # treated as possible sensor noise: keep watching, and only declare it
        # if it lasts or gets worse.
        if (self.use_reasoning and self.assessment is not None
                and not self.assessment.active(ACTIVE_P)
                and self.unhealthy_streak < UNEXPLAINED_PATIENCE and m.health_score >= 70.0):
            return
        causes = self._current_causes(d)
        if self.detector is not None and not causes and self.unhealthy_streak < UNDIAGNOSED_TICKS:
            return                               # give the detector time to commit
        self._open_incident(causes, m)

    def _open_incident(self, causes: List[str], m: MetricsSnapshot) -> None:
        key = key_of(causes) if causes else "unknown"
        self.inc = {
            "id": len(self.incidents) + 1, "key": key, "causes": list(causes),
            "start_tick": self.shop.tick - self.unhealthy_streak + 1,
            "start_health": round(m.health_score, 1), "source": "", "attempts": 0,
            "generations": 0, "planning_ticks": 0, "actions": [],
            "outcome": None, "end_tick": None, "time_to_recover": None,
            "tried": [],                     # plan sources that already failed on this incident
            "escalation_reason": "",
        }
        self.incidents.append(self.inc)
        self._ev("incident_declared", id=self.inc["id"], key=key, causes=causes,
                 health=round(m.health_score, 1))
        self._plan(skip_memory=False)

    def _close(self, outcome: str) -> None:
        if self.inc is not None:
            self.inc["outcome"] = outcome
            self.inc["end_tick"] = self.shop.tick
        self.inc = None
        self._policy = None

    # ---------------------------------------------------------------- planning
    def _try_reasoning(self):
        """Weigh the evidence; if it clearly fits what we know, plan from knowledge."""
        if not self.use_reasoning or len(self.recent) < 2:
            return None
        a = self.engine.assess(View(self.recent), self._last_probs)
        self._ev("reasoning", step="beliefs", text="what I believe: " + a.text(),
                 beliefs=a.brief())
        if not a.confident():
            self._ev("reasoning", step="not_confident",
                     text="the evidence does not clearly fit what I know, so I will not trust "
                          "my own knowledge here")
            return None
        self._ev("reasoning", step="plan",
                 text="this fits my knowledge, so I will plan from it instead of searching")
        return ReasonedPlan(self.engine, list(self.recent), a)

    def _choose(self, allow_memory: bool, allow_reasoning: bool = True):
        """Pick a plan: reasoning from knowledge, a saved playbook we trust, or a
        freshly evolved strategy (in an order set by reasoning_first)."""
        inc = self.inc
        key = inc["key"]
        if self.reasoning_first and allow_reasoning:
            plan = self._try_reasoning()
            if plan is not None:
                return plan, "reasoning", None, 0, 0.0
        if self.use_memory and key != "unknown":
            pb = self.library.find(key, MEMORY_MIN_RATE) if allow_memory else None
            if pb is not None:
                self._ev("memory_hit", playbook=pb.id, key=key,
                         success_rate=round(pb.success_rate, 2), times_used=pb.times_used)
                return pb.strategy(), "memory", pb, 0, pb.fitness
            self._ev("memory_miss", key=key)
        if not self.reasoning_first and allow_reasoning:
            plan = self._try_reasoning()
            if plan is not None:
                return plan, "reasoning", None, 0, 0.0
        if not self.use_evolution:
            return None, "none", None, 0, 0.0
        seeds = self.library.seed_strategies(inc["causes"]) if self.use_memory else []

        def on_gen(gen, best, avg, _s):
            self._ev("evolution_update", generation=gen + 1, best=round(best, 1),
                     avg=round(avg, 1))

        engine = EvolutionEngine(seed=self.seed * 1000 + inc["id"] * 10 + inc["attempts"],
                                 **self.evolution_kwargs)
        res = engine.run(self.shop.clone(), on_generation=on_gen, initial=seeds)
        self._ev("evolution_done", generations=res.generations,
                 best_fitness=round(res.best_fitness, 1), seeded_from=len(seeds),
                 strategy=res.best.describe(), seconds=round(res.seconds, 1))
        # thinking time is charged per batch of 40 strategies tested, so a search
        # that tests many strategies at once is not given free thinking time
        gens = max(res.generations, round(res.evaluations / 40))
        inc["generations"] += gens
        return res.best, "evolution", None, gens, res.best_fitness

    def _relabel_if_new(self) -> None:
        """While we were thinking, the detector may have spotted another
        symptom. The staging copy we evolved on already contained it, so we
        just correct the diagnosis (and therefore the playbook's label)."""
        if self.stream is None or self.inc is None:
            return
        conf = self.stream.confirmed()
        new = sorted(set(conf) - set(self.inc["causes"])) if conf else []
        if new:
            self.inc["causes"] = sorted(set(self.inc["causes"]) | set(new))
            self.inc["key"] = key_of(self.inc["causes"])
            self._ev("diagnosis_updated", key=self.inc["key"], added=new)

    def _plan(self, skip_memory: bool, skip_reasoning: bool = False) -> None:
        inc = self.inc
        tried_memory = skip_memory or "memory" in inc["tried"]
        tried_reasoning = skip_reasoning or "reasoning" in inc["tried"]
        while True:
            causes_at_start = list(inc["causes"])
            strategy, source, pb, gens, fit = self._choose(allow_memory=not tried_memory,
                                                           allow_reasoning=not tried_reasoning)
            if strategy is None:
                self._page("no confident plan from reasoning or memory, and searching is switched off")
                return
            original = strategy
            if self.act_immediately and isinstance(strategy, Strategy):
                strategy = _act_now(strategy)
            inc["attempts"] += 1
            inc["source"] = source
            thinking = gens * TICKS_PER_GENERATION
            inc["planning_ticks"] += thinking
            self._advance(thinking)                       # the incident keeps running
            if self._healed_itself():
                return
            self._relabel_if_new()
            passed = True
            if self.use_dry_run:
                rehearsal = strategy.fresh() if hasattr(strategy, "fresh") else strategy
                dr = dry_run(rehearsal, self.shop.clone())
                inc["planning_ticks"] += DRY_RUN_TICKS
                self._ev("dry_run", passed=dr.passed, health_before=round(dr.health_before, 1),
                         health_after=round(dr.health_after, 1), source=source)
                self._advance(DRY_RUN_TICKS)
                if self._healed_itself():
                    return
                self._relabel_if_new()
                passed = dr.passed
            if passed:
                self._begin_execute(strategy, source, pb, fit, gens, original)
                return
            if source == "memory":                        # saved fix no longer works
                self.library.record_use(pb.id, False)
                self._ev("playbook_failed", playbook=pb.id, reason="dry run failed")
                inc["memory_failed"] = True
                # a saved fix often fails because the problem is bigger than we thought;
                # give the detector a few ticks to catch up, then look again
                before = inc["key"]
                self._advance(3)
                inc["planning_ticks"] += 3
                if self._healed_itself():
                    return
                self._relabel_if_new()
                tried_memory = inc["key"] == before       # new diagnosis = new playbook to try
                continue
            if source == "reasoning":                     # rehearsal of a reasoned plan failed
                self._ev("reasoning", step="rehearsal_failed",
                         text="the rehearsal on the staging copy failed, so I will not use this "
                              "plan on the live shop; falling back")
                tried_reasoning = True
                continue
            if inc["causes"] != causes_at_start and inc["attempts"] < 3:
                continue                                  # new information: plan again
            self._page("dry run of the newly evolved fix failed; "
                       "not risking the live shop")
            return

    def _healed_itself(self) -> bool:
        if self.healthy_streak >= STABLE_TICKS:
            self._ev("self_resolved", key=self.inc["key"] if self.inc else "")
            self._close("self_resolved")
            self.state = MONITOR
            if self.stream is not None:
                self.stream.reset()
            return True
        return False

    # --------------------------------------------------------------- execution
    def _begin_execute(self, strategy, source, pb, fitness, gens, original=None) -> None:
        self._policy = strategy
        self._saved_strategy = original if original is not None else strategy
        self._source = source
        self._playbook = pb
        self._plan_fitness = fitness
        self._plan_generations = gens
        self._ctx = AgentContext(health_history=[], severity_history=[])
        self._exec_ticks = 0
        self.state = EXECUTE
        self._ev("plan_ready", source=source, strategy=strategy.describe(),
                 explain=strategy.explain())

    def _execute_step(self, m: MetricsSnapshot) -> None:
        ctx, inc = self._ctx, self.inc
        ctx.health_history.append(m.health_score)
        ctx.severity_history.append(symptom_severity(m))

        if self.healthy_streak >= STABLE_TICKS:
            self._recover()
            return

        if self._source == "reasoning":
            self._relabel_if_new()                        # the reasoned plan adapts by itself
        elif self.stream is not None:                     # a NEW problem appeared?
            conf = self.stream.confirmed()
            if conf and (set(conf) - set(inc["causes"])):
                self._ev("replan", old=inc["causes"], new=list(conf))
                self._close("superseded")
                self.state = MONITOR
                self._open_incident(list(conf), m)
                return

        ctx.t = self._exec_ticks
        self._exec_ticks += 1
        d = self._policy.decide(m, ctx)
        if hasattr(self._policy, "pop_trace"):
            for t in self._policy.pop_trace():
                self._ev("reasoning", **t)
        if d.kind == "act":
            msg = apply_action(self.shop, d.action, d.strength)
            ctx.actions_taken += 1
            ctx.last_action = d.action
            ctx.last_action_tick = ctx.t
            ctx.severity_at_action = ctx.severity_history[-1]
            inc["actions"].append({"tick": self.shop.tick, "action": d.action,
                                   "strength": round(d.strength, 2)})
            self._ev("action_applied", action=d.action, strength=round(d.strength, 2),
                     reason=d.reason, result=msg)
        elif d.kind == "escalate":
            self._exec_failed(d.reason)
        elif self._exec_ticks > EXEC_TIMEOUT:
            self._exec_failed(f"no recovery after {EXEC_TIMEOUT} ticks")

    def _exec_failed(self, reason: str) -> None:
        # each kind of plan gets ONE chance per incident, so the agent cannot
        # loop forever between reasoning, memory and search: it escalates instead
        if self.inc is not None and self._source not in self.inc["tried"]:
            self.inc["tried"].append(self._source)
        if self._source == "memory":                      # the saved fix did not work
            self.library.record_use(self._playbook.id, False)
            self._ev("playbook_failed", playbook=self._playbook.id, reason=reason)
            if self.inc is not None:
                self.inc["memory_failed"] = True
            self.state = MONITOR
            self._plan(skip_memory=True)                  # reason or evolve a fresh one
        elif self._source == "reasoning" and self.use_evolution:
            self._ev("reasoning", step="fallback",
                     text=f"my knowledge was not enough ({reason}); switching to search")
            self.state = MONITOR
            self._plan(skip_memory=False, skip_reasoning=True)
        else:
            self._page(reason)

    def _recover(self) -> None:
        inc = self.inc
        recovered_tick = self.shop.tick - self.healthy_streak + 1
        inc["time_to_recover"] = recovered_tick - inc["start_tick"]
        self._ev("verification", healthy_ticks=self.healthy_streak,
                 health=round(self.health_trace[-1], 1), recovered=True,
                 time_to_recover=inc["time_to_recover"])
        if self.use_memory and inc["key"] != "unknown" and self._source != "reasoning":
            if self._source == "memory":
                self.library.record_use(self._playbook.id, True)
            elif (self.library.find(inc["key"], MEMORY_MIN_RATE) is None
                  or inc.get("memory_failed")):         # no duplicates of a playbook we already trust
                pb = self.library.add(inc["key"], self._saved_strategy, self._plan_fitness,
                                      self._plan_generations, self.shop.tick)
                self._ev("playbook_saved", **pb.row())
        self._close("recovered")
        self.state = MONITOR
        if self.stream is not None:
            self.stream.reset()

    def _page(self, reason: str) -> None:
        inc = self.inc
        inc["escalation_reason"] = reason
        self._ev("escalation", reason=reason, actions_tried=len(inc["actions"]),
                 key=inc["key"])
        self._close("paged")
        self.state = PAGED

    # ----------------------------------------------------------------- summary
    def summary(self) -> Dict:
        return {"state": self.state, "tick": self.shop.tick,
                "health": round(self.health_trace[-1], 1) if self.health_trace else None,
                "incidents": self.incidents, "playbooks": self.library.rows()}


# ----------------------------------------------------------------------------
# Readable one-line descriptions of events (console and dashboard log)
# ----------------------------------------------------------------------------
def format_event(e: Dict) -> Optional[str]:
    t, d, k = e["tick"], e["data"], e["type"]
    p = f"[t={t:04d}] "
    if k == "incident_declared":
        return p + f"INCIDENT #{d['id']} declared: {d['key']} (health {d['health']})"
    if k == "diagnosis":
        return p + f"diagnosis: {d['causes'] or 'looks healthy'} (confidence {d['confidence']})"
    if k == "memory_hit":
        return p + (f"memory HIT: playbook {d['playbook']} for '{d['key']}' "
                    f"(worked {int(d['success_rate'] * 100)}% of {d['times_used']} uses)")
    if k == "memory_miss":
        return p + f"no trusted playbook for '{d['key']}' - evolving a new fix"
    if k == "evolution_done":
        return p + (f"evolution finished: {d['generations']} generations, best fitness "
                    f"{d['best_fitness']}, {d['seconds']}s, seeded from {d['seeded_from']} playbook(s)")
    if k == "dry_run":
        return p + (f"dry run on staging copy ({d['source']}): health {d['health_before']} -> "
                    f"{d['health_after']} {'PASSED' if d['passed'] else 'FAILED'}")
    if k == "plan_ready":
        return p + f"applying fix to the real shop: {d['explain']}"
    if k == "action_applied":
        return p + f"ACTION {d['action']} (strength {d['strength']}): {d['result']}"
    if k == "verification":
        return p + (f"VERIFIED recovered: health {d['health']} for {d['healthy_ticks']} ticks "
                    f"(time to recover {d['time_to_recover']} ticks)")
    if k == "playbook_saved":
        return p + f"playbook saved: {d['id']} for '{d['key']}' (fitness {d['fitness']})"
    if k == "playbook_failed":
        return p + f"playbook {d['playbook']} FAILED: {d['reason']}"
    if k == "diagnosis_updated":
        return p + f"diagnosis updated while planning: now '{d['key']}' (added {d['added']})"
    if k == "replan":
        return p + f"NEW problem appeared mid-fix: {d['old']} -> {d['new']}, re-planning"
    if k == "escalation":
        return p + f"PAGING A HUMAN: {d['reason']} (actions tried: {d['actions_tried']})"
    if k == "self_resolved":
        return p + "shop healed by itself while the agent was planning"
    if k == "reasoning":
        return p + "REASONING: " + d["text"]
    if k == "auto_cleared":
        return p + f"resuming monitoring: {d['reason']}"
    return None                                           # metrics, evolution_update


# ----------------------------------------------------------------------------
# Demo: python recovery_agent.py demo
# ----------------------------------------------------------------------------
def _print_new_events(agent: RecoveryAgent, seen: int) -> int:
    evs = list(agent.events)
    for e in evs[seen:]:
        line = format_event(e)
        if line:
            print("  " + line)
    return len(evs)


def _demo() -> None:
    try:
        det = load_detector()
    except FileNotFoundError:
        print("Detector model not found. Run  python detector.py  first.")
        sys.exit(1)

    shop = Shop(seed=42)
    lib = PlaybookLibrary(None)
    agent = RecoveryAgent(shop, det, lib, seed=1, record_events=True)
    seen = 0

    def section(title: str) -> None:
        print(f"\n=== {title} ===")

    agent.run(60)
    section("1. Silent order dropper strikes for the FIRST time (nothing saved yet)")
    FAULTS["silent_dropper"](shop, drop_rate=0.20)
    agent.run_until_idle()
    seen = _print_new_events(agent, seen)

    agent.run(40)
    section("2. The SAME kind of fault again, but worse (35% of orders lost)")
    FAULTS["silent_dropper"](shop, drop_rate=0.35)
    agent.run_until_idle()
    seen = _print_new_events(agent, seen)

    agent.run(40)
    section("3. Data corruption AND a frozen workflow together (a combination never seen)")
    FAULTS["data_corruption"](shop, corruption_rate=0.10)
    FAULTS["workflow_deadlock"](shop)
    agent.run_until_idle()
    seen = _print_new_events(agent, seen)

    agent.run(40)
    section("4. A problem the agent CANNOT fix (the root cause keeps coming back)")
    FAULTS["silent_dropper"](shop, drop_rate=0.30)
    agent.before_tick = lambda s: FAULTS["silent_dropper"](s, drop_rate=0.30)
    agent.run_until_idle()
    seen = _print_new_events(agent, seen)
    agent.before_tick = None
    apply_action(shop, "restart_payments", 1.0)           # the human fixes the root cause
    agent.resolve()
    print("  (a human fixed the root cause; the agent resumes monitoring)")

    print("\n=== Incident summary ===")
    print(f"  {'#':<3} {'problem':<34} {'fix came from':<14} {'thinking ticks':>14} "
          f"{'actions':>8} {'outcome':<12} {'time to recover':>15}")
    for i in agent.incidents:
        ttr = "-" if i["time_to_recover"] is None else str(i["time_to_recover"])
        print(f"  {i['id']:<3} {i['key']:<34} {i['source'] or '-':<14} {i['planning_ticks']:>14} "
              f"{len(i['actions']):>8} {i['outcome'] or 'open':<12} {ttr:>15}")
    print("\n=== Playbook library ===")
    for r in lib.rows():
        print(f"  {r['id']}  {r['key']:<30} fitness {r['fitness']:>5}  "
              f"used {r['times_used']}x  success rate {int(r['success_rate'] * 100)}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        t0 = time.perf_counter()
        _demo()
        print(f"\nDemo finished in {time.perf_counter() - t0:.0f}s")
    else:
        print("Usage: python recovery_agent.py demo")
