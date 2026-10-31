"""
evolution.py - the brain of the HackathonShop self-healing recovery agent.

What this file does, in plain words
-----------------------------------
When the shop breaks, we do NOT hand-code the fix. Instead we:

  1. Create many random "recovery strategies" (a strategy is a short list of
     settings: how long to watch before acting, which fixes to try, how hard
     to apply them, how many tries to allow, when to page a human).
  2. Test every strategy on COPIES of the broken shop (a staging copy), so the
     real shop is never touched.
  3. Score each one (fitness): did the shop recover, how fast, with how few
     actions, and did it page a human when the problem could not be fixed?
  4. Keep the best strategies, breed them (mix two parents, add small random
     changes), and repeat for a number of generations.

The best strategy that comes out is the fix. This is evolutionary AI.

The file also contains two simple baselines (a do-nothing agent and a
hand-written rule-based fixer) so we can PROVE the evolved agent is better
than simple alternatives, and a dry_run() helper that tests a strategy on a
staging copy before it is applied to the real shop.

Run:  python evolution.py                    (demo on the compound incident)
      python evolution.py all                (demo on all five incidents)
      python evolution.py silent_dropper     (demo on one named incident)
Incidents: silent_dropper, queue_overflow, data_corruption,
           workflow_deadlock, compound
"""
from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from shop import (
    ACTIONS,
    FAULTS,
    WINDOW,
    MetricsSnapshot,
    Shop,
    apply_action,
    collect_metrics,
    inject_data_corruption,
    inject_silent_dropper,
    inject_workflow_deadlock,
    run_shop,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
HEALTHY = 80.0        # health score at or above this counts as "healthy"
STABLE_TICKS = 5      # healthy for this many ticks in a row = really recovered
MAX_TICKS = 80        # how long an agent gets to fix an incident
SETTLE_TICKS = 20     # after a late fault appears, wait this long before
                      # declaring recovery (symptoms take time to show up)


# ----------------------------------------------------------------------------
# What an agent can decide each tick
# ----------------------------------------------------------------------------
@dataclass
class Decision:
    kind: str                 # "wait" | "act" | "escalate"
    action: str = ""
    strength: float = 1.0
    reason: str = ""


WAIT = Decision("wait")


@dataclass
class AgentContext:
    """What the agent remembers about its own behaviour so far."""
    t: int = 0
    actions_taken: int = 0
    last_action: str = ""
    last_action_tick: Optional[int] = None
    health_history: List[float] = field(default_factory=list)
    severity_history: List[float] = field(default_factory=list)
    severity_at_action: float = 0.0

    @property
    def since_last(self) -> int:
        return 10 ** 6 if self.last_action_tick is None else self.t - self.last_action_tick

    @property
    def fix_settling(self) -> bool:
        """True while our last fix deserves more time. The metrics look back
        over a 15-tick window, so right after a fix they can stay flat for
        several ticks even though the fix worked. A fix is therefore judged
        only after one full window has passed, unless the symptoms are
        clearly getting WORSE."""
        if not self.severity_history or self.last_action_tick is None:
            return False
        if self.since_last > WINDOW:
            return False
        return self.severity_history[-1] <= self.severity_at_action + 1.0


def symptom_severity(m: MetricsSnapshot) -> float:
    """Uncapped 'how bad are the raw symptoms' number (0 = none)."""
    return (m.mismatch_rate * 200 + m.drop_rate * 100 + (1.0 - m.integrity_score) * 600
            + m.stuck_duration * 3 + max(0.0, m.queue_saturation - 0.8) * 50)


# ----------------------------------------------------------------------------
# Recovery strategy (the thing that evolves)
# ----------------------------------------------------------------------------
GENE_NAMES = [
    "observe", "trigger", "first", "second", "third", "fourth",
    "strength", "gap", "budget", "escalate_at",
]


def _bucket(gene: float, n: int) -> int:
    return min(int(gene * n), n - 1)


@dataclass
class Strategy:
    """
    10 settings, each a number between 0 and 1. Evolution only ever changes
    these numbers; the properties below translate them into real behaviour.
    """
    observe: float = 0.2       # how long to watch before acting
    trigger: float = 0.5       # how unhealthy the shop must look before acting
    first: float = 0.0         # which fix to try first
    second: float = 0.3        # which fix to try second
    third: float = 0.6         # which fix to try third
    fourth: float = 0.9        # which fix to try fourth (four problems at once)
    strength: float = 1.0      # gentle (0) ... aggressive (1)
    gap: float = 0.5           # patience between attempts
    budget: float = 0.4        # how many attempts before giving up
    escalate_at: float = 0.4   # health level at which to page a human

    # ---- encoding helpers -------------------------------------------------
    def to_array(self) -> List[float]:
        return [getattr(self, n) for n in GENE_NAMES]

    @classmethod
    def from_array(cls, arr) -> "Strategy":
        return cls(*[min(1.0, max(0.0, float(v))) for v in arr])

    @classmethod
    def random(cls, rng: random.Random) -> "Strategy":
        return cls(*[rng.random() for _ in GENE_NAMES])

    # ---- decoded behaviour ------------------------------------------------
    @property
    def observe_ticks(self) -> int:
        return 1 + int(self.observe * 9)                 # 1..10

    @property
    def trigger_health(self) -> float:
        return 55.0 + 25.0 * self.trigger                # 55..80

    @property
    def action_sequence(self) -> List[str]:
        return [ACTIONS[_bucket(g, len(ACTIONS))]
                for g in (self.first, self.second, self.third, self.fourth)]

    @property
    def strength_value(self) -> float:
        return 0.2 + 0.8 * self.strength                 # 0.2..1.0

    @property
    def retry_gap(self) -> int:
        return 3 + int(self.gap * 12)                    # 3..15

    @property
    def max_actions(self) -> int:
        return 1 + int(self.budget * 5)                  # 1..6

    @property
    def escalate_health(self) -> float:
        return self.escalate_at * 50.0                   # 0..50

    def describe(self) -> Dict:
        return {
            "watch_ticks_before_acting": self.observe_ticks,
            "act_when_health_below": round(self.trigger_health, 1),
            "fix_order": self.action_sequence,
            "strength": round(self.strength_value, 2),
            "wait_between_attempts": self.retry_gap,
            "max_attempts": self.max_actions,
            "page_human_if_health_below": round(self.escalate_health, 1),
        }

    def explain(self) -> str:
        d = self.describe()
        return (f"watch {d['watch_ticks_before_acting']} ticks, act when health < "
                f"{d['act_when_health_below']}, try {' then '.join(d['fix_order'][:self.max_actions])} "
                f"at strength {d['strength']}, wait {d['wait_between_attempts']} ticks between "
                f"attempts, max {d['max_attempts']} attempts, page a human if health < "
                f"{d['page_human_if_health_below']} after a failed attempt")

    # ---- the decision rule ------------------------------------------------
    def decide(self, m: MetricsSnapshot, ctx: AgentContext) -> Decision:
        health = m.health_score
        if ctx.t + 1 < self.observe_ticks:
            return WAIT
        if health >= self.trigger_health:
            return WAIT
        if ctx.since_last < self.retry_gap:
            return WAIT                                   # give the last fix time to work
        if ctx.actions_taken >= 1 and ctx.fix_settling:
            return WAIT                                   # give the fix a full window to show
        if ctx.actions_taken >= self.max_actions:
            return Decision("escalate", reason=(
                f"used all {self.max_actions} attempts, health still {health:.0f}"))
        if ctx.actions_taken >= 1 and health < self.escalate_health:
            return Decision("escalate", reason=(
                f"health {health:.0f} is below {self.escalate_health:.0f} after "
                f"{ctx.actions_taken} attempt(s)"))
        action = self.action_sequence[ctx.actions_taken % 4]
        return Decision("act", action, self.strength_value,
                        reason=f"health {health:.0f} below {self.trigger_health:.0f}")


# ----------------------------------------------------------------------------
# Baselines (simple alternatives the evolved agent must beat)
# ----------------------------------------------------------------------------
class DoNothingAgent:
    """Never acts, never escalates."""
    name = "do_nothing"

    def decide(self, m: MetricsSnapshot, ctx: AgentContext) -> Decision:
        return WAIT


class RuleBasedAgent:
    """
    A hand-written if-this-then-that fixer, the way an engineer would script it.
    Looks at the symptoms, picks the matching fix, waits 8 ticks, repeats,
    and pages a human after 6 attempts.
    """
    name = "rule_based"

    def decide(self, m: MetricsSnapshot, ctx: AgentContext) -> Decision:
        if ctx.t < 3 or m.health_score >= HEALTHY:
            return WAIT
        if ctx.since_last < 8:
            return WAIT
        if ctx.actions_taken >= 6:
            return Decision("escalate", reason=f"6 attempts used, health {m.health_score:.0f}")
        if m.jobs_stuck > 0 and m.stuck_duration >= 5:
            action = "kill_stuck_job"
        elif m.integrity_score < 0.97:
            action = "rollback_sync"
        elif m.drop_rate > 0.03 or m.queue_saturation > 0.9:
            action = "throttle_traffic" if ctx.last_action == "scale_queue" else "scale_queue"
        elif m.mismatch_rate > 0.03:
            action = "restart_payments"
        else:
            return WAIT
        return Decision("act", action, 1.0, reason=f"rule matched, health {m.health_score:.0f}")


# ----------------------------------------------------------------------------
# Scenarios: a broken shop plus (optionally) things that happen later
# ----------------------------------------------------------------------------
@dataclass
class Event:
    """A fault that appears `tick` ticks after the agent takes over."""
    tick: int
    fault: str
    kwargs: Dict = field(default_factory=dict)


@dataclass
class Scenario:
    shop: Shop                                   # template, never modified
    events: List[Event] = field(default_factory=list)
    persist: Dict = field(default_factory=dict)  # faults that keep coming back
    fixable: bool = True                         # False = a human is needed
    name: str = ""


def _reapply(shop: Shop, persist: Dict) -> None:
    """Re-create faults whose root cause is never really fixed."""
    if not persist:
        return
    if persist.get("drop_rate"):
        inject_silent_dropper(shop, persist["drop_rate"])
    if persist.get("corruption_rate"):
        inject_data_corruption(shop, persist["corruption_rate"])
    if persist.get("deadlock") and not any(j.stuck for j in shop.workflows.jobs):
        inject_workflow_deadlock(shop)
    if persist.get("flap") and shop.payments.flap_period == 0:
        shop.payments.flap_rate, shop.payments.flap_period = persist["flap"]
    if persist.get("leak") and not shop.payments.leak_active:
        shop.payments.leak_active, shop.payments.leak_every = True, persist["leak"]


@dataclass
class AgentResult:
    recovered: bool
    escalated: bool
    start_health: float
    final_health: float
    ticks_to_recovery: int
    ticks_to_escalation: int
    actions_taken: int
    escalation_reason: str
    action_log: List[str]
    health_trace: List[float]
    ticks_run: int
    residual: bool = False       # health recovered but a root cause is still switched on


def run_agent(policy, scenario: Scenario, max_ticks: int = MAX_TICKS) -> AgentResult:
    """Let one agent try to fix one scenario (on a private copy of the shop)."""
    shop = scenario.shop.clone()
    start_health = collect_metrics(shop).health_score
    pending = sorted(scenario.events, key=lambda e: e.tick)
    last_event_tick = max((e.tick for e in pending), default=0)
    settle_until = last_event_tick + SETTLE_TICKS if pending else 0

    trace: List[float] = []
    severity: List[float] = []
    ctx = AgentContext(health_history=trace, severity_history=severity)
    log: List[str] = []
    streak = 0
    escalated, esc_tick, reason = False, -1, ""

    for t in range(max_ticks):
        while pending and pending[0].tick <= t:
            ev = pending.pop(0)
            FAULTS[ev.fault](shop, **ev.kwargs)
        _reapply(shop, scenario.persist)
        run_shop(shop, 1)

        m = collect_metrics(shop)
        trace.append(m.health_score)
        severity.append(symptom_severity(m))
        streak = streak + 1 if m.health_score >= HEALTHY else 0
        if streak >= STABLE_TICKS and not pending and t >= settle_until:
            break                                       # genuinely recovered

        ctx.t = t
        d = policy.decide(m, ctx)
        if d.kind == "act":
            apply_action(shop, d.action, d.strength)
            ctx.actions_taken += 1
            ctx.last_action = d.action
            ctx.last_action_tick = t
            ctx.severity_at_action = severity[-1]
            log.append(f"t={t} {d.action} (strength {d.strength:.2f}) - {d.reason}")
        elif d.kind == "escalate":
            escalated, esc_tick, reason = True, t, d.reason
            log.append(f"t={t} PAGE HUMAN - {d.reason}")
            break

    recovered = streak >= STABLE_TICKS and not pending and len(trace) > settle_until
    ttr = max(0, (len(trace) - streak) - last_event_tick) if recovered else max_ticks
    return AgentResult(
        recovered=recovered,
        escalated=escalated,
        start_health=start_health,
        final_health=trace[-1] if trace else start_health,
        ticks_to_recovery=ttr,
        ticks_to_escalation=esc_tick if escalated else max_ticks,
        actions_taken=ctx.actions_taken,
        escalation_reason=reason,
        action_log=log,
        health_trace=trace,
        ticks_run=len(trace),
        residual=bool(_persistent_part(shop)),
    )


# ----------------------------------------------------------------------------
# Fitness: how good was this run? (0..100)
# ----------------------------------------------------------------------------
def compute_fitness(r: AgentResult, fixable: bool = True) -> float:
    efficiency = max(0.0, 1.0 - max(0, r.actions_taken - 1) / 4.0)   # 1 action = best
    if fixable:
        if r.recovered:
            speed = max(0.0, 1.0 - r.ticks_to_recovery / MAX_TICKS)
            score = 55.0 + 30.0 * speed + 15.0 * efficiency
            return score - (15.0 if r.residual else 0.0)   # patched over, not truly fixed
        if r.escalated:
            return 10.0 + 10.0 * r.final_health / 100.0           # failed safely, did not fix
        return 25.0 * r.final_health / 100.0                      # quietly left it broken
    # Problem a machine cannot fix: the right move is to try, then page a human.
    if r.escalated and r.actions_taken >= 1:
        speed = max(0.0, 1.0 - r.ticks_to_escalation / MAX_TICKS)
        return 70.0 + 15.0 * speed + 15.0 * efficiency
    if r.escalated:
        return 30.0                                               # paged without even trying
    return 10.0 * r.final_health / 100.0                          # kept going / gave up silently


def evaluate_policy(policy, scenarios: List[Scenario]) -> Tuple[float, List[AgentResult]]:
    results = [run_agent(policy, sc) for sc in scenarios]
    fit = sum(compute_fitness(r, sc.fixable) for r, sc in zip(results, scenarios)) / len(results)
    return fit, results


# ----------------------------------------------------------------------------
# Building training scenarios from a broken shop
# ----------------------------------------------------------------------------
def _jitter(shop: Shop, rng: random.Random) -> None:
    """Make a slightly different version of the same incident (so agents
    cannot just memorise one exact situation)."""
    p, s = shop.payments, shop.sync
    if p.drop_active:
        p.drop_rate = min(0.6, max(0.05, p.drop_rate * rng.uniform(0.6, 2.0)))
    if shop.traffic_multiplier > 1.05:
        shop.traffic_multiplier = max(1.5, shop.traffic_multiplier * rng.uniform(0.7, 1.6))
    if s.corruption_active:
        s.corruption_rate = min(0.3, max(0.03, s.corruption_rate * rng.uniform(0.6, 2.0)))


def make_visibly_sick(shop: Shop, max_extra: int = 80) -> None:
    """Let a fault keep developing until it is clearly visible in the health
    score. Otherwise a barely-sick shop makes 'do nothing' look like a cure."""
    extra = 0
    while collect_metrics(shop).health_score >= 76.0 and extra < max_extra:
        run_shop(shop, 5)
        extra += 5


def _persistent_part(shop: Shop) -> Dict:
    d: Dict = {}
    if shop.payments.drop_active:
        d["drop_rate"] = shop.payments.drop_rate
    if shop.sync.corruption_active:
        d["corruption_rate"] = shop.sync.corruption_rate
    if shop.workflows.deadlock_active:
        d["deadlock"] = True
    if shop.payments.flap_period > 0:
        d["flap"] = (shop.payments.flap_rate, shop.payments.flap_period)
    if shop.payments.leak_active:
        d["leak"] = shop.payments.leak_every
    return d


AGE_CHOICES = (0, 10, 25)   # extra ticks an incident has been running when the fix starts


def make_scenarios(sick_shop: Shop, seed: int, variants: int = 3,
                   include_persistent: bool = True) -> List[Scenario]:
    """
    Training cases built from the broken shop. Each variant differs in fault
    severity, random luck AND age. Age matters: the shop's metrics look back
    over a 15-tick window, so a fix shows its effect differently on a fresh
    incident than on an old one, and a real agent always applies its fix a
    while after the incident started. Training on mixed ages stops evolution
    from tuning itself to one lucky timing.
    """
    rng = random.Random(seed)
    base = sick_shop.clone()
    base.logs.clear()
    scenarios: List[Scenario] = []
    for i in range(variants):
        v = base.clone()
        v.rng = random.Random(rng.randrange(10 ** 9))
        _jitter(v, rng)
        make_visibly_sick(v)
        run_shop(v, AGE_CHOICES[i % len(AGE_CHOICES)])
        scenarios.append(Scenario(v, name=f"variant{i}"))
    persist = _persistent_part(base)
    if include_persistent and persist:
        v = base.clone()
        v.rng = random.Random(rng.randrange(10 ** 9))
        make_visibly_sick(v)
        run_shop(v, rng.choice(AGE_CHOICES[:2]))
        scenarios.append(Scenario(v, persist=persist, fixable=False, name="persistent"))
    return scenarios


# ----------------------------------------------------------------------------
# The evolution engine
# ----------------------------------------------------------------------------
@dataclass
class EvolutionResult:
    best: Strategy
    best_fitness: float
    history: List[Dict]
    generations: int
    evaluations: int
    seconds: float


class EvolutionEngine:
    def __init__(
        self,
        population_size: int = 40,
        survivors: int = 8,
        mutation_rate: float = 0.2,
        mutation_strength: float = 0.25,
        max_generations: int = 20,
        variants: int = 3,
        include_persistent: bool = True,
        early_stop: float = 97.0,
        stall_generations: int = 6,
        min_generations: int = 8,
        seed: int = 0,
    ):
        self.population_size = population_size
        self.survivors = survivors
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.max_generations = max_generations
        self.variants = variants
        self.include_persistent = include_persistent
        self.early_stop = early_stop
        self.stall_generations = stall_generations     # stop if no progress this long
        self.min_generations = min_generations         # but always run at least this many
        self.seed = seed

    def _crossover(self, a: Strategy, b: Strategy, rng: random.Random) -> Strategy:
        return Strategy.from_array(
            [x if rng.random() < 0.5 else y for x, y in zip(a.to_array(), b.to_array())])

    def _mutate(self, s: Strategy, rng: random.Random) -> Strategy:
        arr = s.to_array()
        for i in range(len(arr)):
            if rng.random() < self.mutation_rate:
                if rng.random() < 0.15:
                    arr[i] = rng.random()                         # wild new value
                else:
                    arr[i] += rng.gauss(0, self.mutation_strength)  # small tweak
        return Strategy.from_array(arr)

    def run(
        self,
        sick_shop: Shop,
        on_generation: Optional[Callable[[int, float, float, Strategy], None]] = None,
        initial: Optional[List[Strategy]] = None,
    ) -> EvolutionResult:
        t0 = time.perf_counter()
        rng = random.Random(self.seed)
        scenarios = make_scenarios(sick_shop, rng.randrange(10 ** 9),
                                   self.variants, self.include_persistent)

        population: List[Strategy] = list((initial or [])[: self.population_size // 2])
        while len(population) < self.population_size:
            population.append(Strategy.random(rng))

        cache: Dict[Tuple, float] = {}
        history: List[Dict] = []
        scored: List[Tuple[Strategy, float]] = []

        for gen in range(self.max_generations):
            scored = []
            for s in population:
                key = tuple(s.to_array())
                if key not in cache:
                    cache[key], _ = evaluate_policy(s, scenarios)
                scored.append((s, cache[key]))
            scored.sort(key=lambda x: x[1], reverse=True)

            best_f = scored[0][1]
            avg_f = sum(f for _, f in scored) / len(scored)
            history.append({"generation": gen, "best": round(best_f, 2), "avg": round(avg_f, 2)})
            if on_generation:
                on_generation(gen, best_f, avg_f, scored[0][0])
            if best_f >= self.early_stop:
                break
            if gen + 1 >= self.min_generations and len(history) > self.stall_generations:
                if best_f <= history[-1 - self.stall_generations]["best"] + 0.01:
                    break                                   # converged: no progress lately

            survivors = [s for s, _ in scored[: self.survivors]]
            children: List[Strategy] = []
            while len(survivors) + len(children) < self.population_size:
                a, b = rng.sample(survivors, 2) if len(survivors) >= 2 else (survivors[0], survivors[0])
                children.append(self._mutate(self._crossover(a, b, rng), rng))
            population = survivors + children

        return EvolutionResult(
            best=scored[0][0],
            best_fitness=scored[0][1],
            history=history,
            generations=len(history),
            evaluations=len(cache),
            seconds=time.perf_counter() - t0,
        )


# ----------------------------------------------------------------------------
# Dry run: test a strategy on a staging copy before touching the real shop
# ----------------------------------------------------------------------------
@dataclass
class DryRun:
    passed: bool
    health_before: float
    health_after: float
    result: AgentResult


def dry_run(policy, shop: Shop, max_ticks: int = MAX_TICKS) -> DryRun:
    r = run_agent(policy, Scenario(shop=shop, name="dry run"), max_ticks)
    # On a staging copy we can also check the root cause is really gone.
    return DryRun(passed=r.recovered and not r.residual, health_before=r.start_health,
                  health_after=r.final_health, result=r)


# ----------------------------------------------------------------------------
# Helpers to build broken shops (used by the demo and later by evaluate.py)
# ----------------------------------------------------------------------------
INCIDENTS = ["silent_dropper", "queue_overflow", "data_corruption",
             "workflow_deadlock", "compound"]

TRAIN_PARAMS: Dict[str, Dict] = {
    "silent_dropper": {"drop_rate": 0.20},
    "queue_overflow": {"traffic_multiplier": 3.0},
    "data_corruption": {"corruption_rate": 0.08},
    "workflow_deadlock": {},
    "compound": {"drop_rate": 0.15, "traffic_multiplier": 2.5},
}

HELDOUT_PARAMS: Dict[str, Dict] = {
    "silent_dropper": {"drop_rate": 0.35},
    "queue_overflow": {"traffic_multiplier": 4.0},
    "data_corruption": {"corruption_rate": 0.12},
    "workflow_deadlock": {},
    "compound": {"drop_rate": 0.25, "traffic_multiplier": 3.5},
}


def build_sick_shop(incident: str, seed: int, params: Optional[Dict] = None,
                    warmup: int = 50, develop: int = 40) -> Shop:
    """A healthy shop that runs for a while, breaks, and gets worse."""
    shop = Shop(seed=seed)
    run_shop(shop, warmup)
    FAULTS[incident](shop, **(TRAIN_PARAMS[incident] if params is None else params))
    run_shop(shop, develop)
    make_visibly_sick(shop)
    return shop


# ----------------------------------------------------------------------------
# Demo: python evolution.py [incident|all]
# ----------------------------------------------------------------------------
def _demo(incident: str) -> None:
    print(f"\n=== Incident: {incident} ===")
    sick = build_sick_shop(incident, seed=11)
    print(f"Broken shop health: {collect_metrics(sick).health_score:.1f} / 100")
    print("Evolving recovery strategies on staging copies of the broken shop...")

    def on_gen(gen, best, avg, _s):
        print(f"  generation {gen + 1:>2}: best fitness {best:5.1f}   average {avg:5.1f}")

    engine = EvolutionEngine(seed=1)
    res = engine.run(sick, on_generation=on_gen)
    print(f"Done in {res.seconds:.1f}s, {res.evaluations} strategies tested.")
    print("Winning strategy:", res.best.explain())

    dr = dry_run(res.best, sick)
    print(f"Dry run on staging copy: health {dr.health_before:.0f} -> {dr.health_after:.0f}, "
          f"{'PASSED' if dr.passed else 'FAILED'}")

    # Held-out check: DIFFERENT random seeds and DIFFERENT (harsher) fault settings
    contenders = [("do nothing", DoNothingAgent()), ("rule-based", RuleBasedAgent()),
                  ("evolved", res.best)]
    seeds = [101, 102, 103, 104, 105]
    scen = [Scenario(build_sick_shop(incident, s, HELDOUT_PARAMS[incident]), name=f"heldout{s}")
            for s in seeds]
    print(f"\nHeld-out test ({len(seeds)} unseen shops, harsher fault settings):")
    print(f"  {'agent':<12} {'recovered':>10} {'avg ticks to fix':>18} {'avg actions':>12}")
    for name, pol in contenders:
        results = [run_agent(pol, sc) for sc in scen]
        ok = [r for r in results if r.recovered]
        ticks = (sum(r.ticks_to_recovery for r in ok) / len(ok)) if ok else float("nan")
        acts = sum(r.actions_taken for r in results) / len(results)
        print(f"  {name:<12} {len(ok):>5}/{len(results):<4} {ticks:>18.1f} {acts:>12.1f}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "compound"
    names = INCIDENTS if arg == "all" else [arg]
    for n in names:
        if n not in FAULTS:
            print(f"Unknown incident '{n}'. Choose from: {', '.join(INCIDENTS)} or 'all'.")
            sys.exit(1)
        _demo(n)
